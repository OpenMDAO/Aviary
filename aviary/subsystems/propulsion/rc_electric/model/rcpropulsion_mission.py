import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.rc_performance import \
    Battery, ElectronicSpeedController, Motor, PropCoefficients, Propeller, PowerResiduals, Vectorization
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic


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
        self.name = 'rcpropulsion_mission'

    def setup(self):
        nn = self.options['num_nodes']

        self.set_input_defaults('full_throttle', val=np.ones(nn), units='unitless')

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
            Motor(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT, 
                Aircraft.Engine.Motor.MAX_CONT_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV,
                Dynamic.Vehicle.Propulsion.CURRENT,
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.RPM,
                ]
        )

        self.add_subsystem(
            'propco', 
            PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn), 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.RPM, 
                Dynamic.Mission.VELOCITY, 
                'temp_diameter', 
                'temp_pitch',
            ],
            promotes_outputs=['ct', 'cp']
        )

        self.add_subsystem(
            'prop', 
            Propeller(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER, 
                Dynamic.Vehicle.Propulsion.RPM, 
                'ct', 
                'cp', 
                # Aircraft.Engine.NUM_ENGINES, 
                Dynamic.Atmosphere.DENSITY
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER, 
                Dynamic.Vehicle.Propulsion.THRUST,
                # Dynamic.Vehicle.Propulsion.THRUST_MAX,
                ]
        )
        self.add_subsystem(
            'power_summation', 
            PowerResiduals(num_nodes=nn), 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER
                ],
            promotes_outputs=[
                'power_net'
            ]
        )

        self.add_subsystem(
            'battery_max', 
            Battery(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,  
                Aircraft.Battery.RESISTANCE,
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
            ]
        )

        self.add_subsystem(
            'esc_max',
            ElectronicSpeedController(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Vehicle.Propulsion.THROTTLE, 'full_throttle'),
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
            ]
            )



        self.add_subsystem(
            'motor_max', 
            Motor(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT, 
                Aircraft.Engine.Motor.MAX_CONT_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV,
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
                ],
            promotes_outputs=[
                (Dynamic.Vehicle.Propulsion.RPM, Dynamic.Vehicle.Propulsion.RPM_MAX), 
                'current_constraint',
                ]
        )

        self.add_subsystem('vectorize_geo', Vectorization(num_nodes=nn), 
            promotes_inputs=[Aircraft.Engine.Propeller.DIAMETER, Aircraft.Engine.Propeller.PITCH],
            promotes_outputs=['temp_diameter', 'temp_pitch']
            )

        self.add_subsystem(
            'propco_max', 
            PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn), 
            promotes_inputs=[
                (Dynamic.Vehicle.Propulsion.RPM, Dynamic.Vehicle.Propulsion.RPM_MAX), #TODO: CHANGE MAX 
                Dynamic.Mission.VELOCITY, 
                'temp_diameter', 
                'temp_pitch',
            ],
            promotes_outputs=[('ct','ct_max'), ('cp', 'cp_max')]
        )



        self.add_subsystem(
            'prop_max', 
            Propeller(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER, 
                (Dynamic.Vehicle.Propulsion.RPM, Dynamic.Vehicle.Propulsion.RPM_MAX), 
                ('ct','ct_max'), 
                ('cp', 'cp_max'),
                # Aircraft.Engine.NUM_ENGINES, 
                Dynamic.Atmosphere.DENSITY
                ],
            promotes_outputs=[
                (Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.PROP_POWER_MAX), 
                (Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Vehicle.Propulsion.THRUST_MAX)
                ]
        )
        
        self.add_subsystem(
            'power_summation_max', 
            PowerResiduals(num_nodes=nn), 
            promotes_inputs=[
                (Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.PROP_POWER_MAX), 
                ],
            promotes_outputs=[
                ('power_net','power_net_max')
            ]
        )
        
        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')

        self.connect('battery_max.voltage_out', 'esc_max.voltage_in')
        self.connect('esc_max.voltage_out', 'motor_max.voltage_in')
        self.connect('esc_max.current_out', 'motor_max.current')
        #TODO Alex from phase builder base import add_control

        self.add_constraint('power_net', equals=0, ref=1e2)
        self.add_constraint('power_net_max', equals=0, ref=1e2)
        
        self.add_constraint('current_constraint', upper=0, ref=1e2)
        self.add_constraint(Dynamic.Vehicle.Propulsion.RPM_MAX, lower=1, upper=7500, ref=1e3, units='rpm')

        #Constraints to prevent ill-fated surrogate model predictions
        self.add_constraint('ct_max', lower=0, upper=0.12, ref=1.0, units='unitless')
        self.add_constraint('cp_max', lower=0.0034, upper=0.08, ref=1.0, units='unitless')

        self.connect('battery.power', 'power_summation.power_batt')
        self.connect('esc.power', 'power_summation.power_esc')
        self.connect('motor.power', 'power_summation.power_motor')

        self.connect('battery_max.power', 'power_summation_max.power_batt')
        self.connect('esc_max.power', 'power_summation_max.power_esc')
        self.connect('motor_max.power', 'power_summation_max.power_motor')

        # self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # self.nonlinear_solver.options["maxiter"] = 15
        # self.nonlinear_solver.options["err_on_non_converge"] = False
        # self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        # self.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
        # self.nonlinear_solver.linesearch.options["print_bound_enforce"] = False
        # self.linear_solver = om.DirectSolver(assemble_jac=True)#, rhs_checking =True)

        self.options['auto_order'] = True

        # self.add_constraint(Dynamic.Vehicle.Propulsion.CURRENT, lower=0)