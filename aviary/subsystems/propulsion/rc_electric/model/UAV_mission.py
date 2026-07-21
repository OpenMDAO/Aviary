import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.UAV_performance import \
    Throttle, Battery, ElectronicSpeedController, Motor, PropCoefficients, Propeller, Vectorization
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.dbf_variables import Aircraft, Dynamic


class RCPropMission(om.Group):
    """Calculates the mission performance (ODE) of a single electric RCMotor."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.options.declare(
            'power_balance_mode', default = 'feedforward', values = ['feedforward', 'solver'], desc = 'Choose between feedforward or solver power balance')
        
        self.name = 'rcpropulsion_mission'

    def setup(self):
        nn = self.options['num_nodes']

        user_feedforward = self.options['power_balance_mode'] == 'feedforward'


        # constraint ties the motor to the prop load; in solver mode the solver does.
        motor_load_factor = 1.0

        #in feedforward mode the prop table reads rpm_slack (a bounded
        # optimizer control) instead of the motor RPM, so the lookup can never
        # leave the training data. The rpm_balance comps below force the motor
        # RPM to match it at the optimum.
        if user_feedforward:
            rpm_in = [(Dynamic.Vehicle.Propulsion.RPM, 'rpm_slack')]
            self.set_input_defaults('rpm_slack', val=np.ones(nn) * 60.0, units='rev/s')
        else:
            rpm_in = []  # solver mode: motor RPM is connected straight in below


        self.add_subsystem(
            'throttle',
            Throttle(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
            ],

            promotes_outputs = [
                Dynamic.Vehicle.Propulsion.CURRENT,]
        )

       
        
        self.add_subsystem(
            'battery',
            Battery(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,
                Aircraft.Battery.RESISTANCE,
                Dynamic.Vehicle.Propulsion.CURRENT,
            ]
        )

        self.add_subsystem(
            'esc', 
            ElectronicSpeedController(num_nodes=nn), 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
                Dynamic.Vehicle.Propulsion.CURRENT
            ],
        )

        self.add_subsystem(
            'motor',
            Motor(num_nodes=nn, load_factor=motor_load_factor),
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT, 
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV,
                Dynamic.Vehicle.Propulsion.CURRENT,
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.RPM,
                ]
        )

      
        self.add_subsystem('vectorize_geo', Vectorization(num_nodes=nn), 
            promotes_inputs=[Aircraft.Engine.Propeller.DIAMETER, Aircraft.Engine.Propeller.PITCH],
            promotes_outputs=['temp_diameter', 'temp_pitch']
            )

        

        self.add_subsystem(
            'propco',
            PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                'temp_diameter',
                'temp_pitch',
            ] + rpm_in,
            promotes_outputs=['ct', 'cp']
        )


        self.add_subsystem(
            'prop',
            Propeller(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER,
                'ct',
                'cp',
                
                Dynamic.Atmosphere.DENSITY
                ] + rpm_in,
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER,
                Dynamic.Vehicle.Propulsion.THRUST,
                
                ]
        )

        
        self.add_subsystem(
            'rpm_balance',
            om.ExecComp(
                'rpm_defect = rpm_slack - rpm_motor',
                rpm_defect={'val': np.zeros(nn), 'units': 'rev/s'},
                rpm_slack={'val': np.zeros(nn), 'units': 'rev/s'},
                rpm_motor={'val': np.zeros(nn), 'units': 'rev/s'},
                has_diag_partials=True,
            ),
            promotes_inputs=['rpm_slack'],
        )
        self.connect(Dynamic.Vehicle.Propulsion.RPM, 'rpm_balance.rpm_motor')
       

       
        
       
        

        self.add_subsystem(
            'electric_power',
            om.ExecComp(
                'p_elec = v_batt * current',
                p_elec={'val': np.zeros(nn), 'units': 'W'},
                v_batt={'val': np.zeros(nn), 'units': 'V'},
                current={'val': np.zeros(nn), 'units': 'A'},
            ),
            promotes_inputs=[
                ('current', Dynamic.Vehicle.Propulsion.CURRENT),
            ],
            promotes_outputs=[('p_elec', Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN)],
        )

        self.connect('battery.voltage_out', 'electric_power.v_batt')
        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')

       
        
        

       

        




        """Constraints"""
              # Force commanded cruise RPM to match motor-computed RPM.
        self.add_constraint('rpm_balance.rpm_defect', upper=0.004, lower=-0.004, ref = 1, units='rev/s')
       
        

        self.options['auto_order'] = True

