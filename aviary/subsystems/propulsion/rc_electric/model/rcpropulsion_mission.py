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
            'aviary_inputs',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'rcpropulsion_mission'

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(
            'battery', 
            Battery(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,  
                Aircraft.Battery.RESISTANCE
            ]
        )

        self.add_subsystem(
            'esc', 
            ElectronicSpeedController(num_nodes=nn), 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE
            ]
        )

        self.add_subsystem('motor', Motor(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT, 
                Aircraft.Engine.Motor.PEAK_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV,
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
                Aircraft.Engine.NUM_ENGINES, 
                Dynamic.Atmosphere.DENSITY
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER, 
                Dynamic.Vehicle.Propulsion.THRUST
                ]
        )

        self.add_subsystem(
            'power_net', 
            PowerResiduals(num_nodes=nn), 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER
                ]
            )

        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        # self.connect('esc.current_out', 'motor.current')


        self.connect('battery.power', 'power_net.power_batt')
        self.connect('esc.power', 'power_net.power_esc')
        self.connect('motor.power', 'power_net.power_motor')
        self.connect('power_net.current', ['battery.current', 'esc.current_in', 'motor.current'])

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["maxiter"] = 30
        self.nonlinear_solver.options["err_on_non_converge"] = False
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
        self.nonlinear_solver.linesearch.options["print_bound_enforce"] = True
        self.linear_solver = om.DirectSolver(assemble_jac=True)#, rhs_checking =True)