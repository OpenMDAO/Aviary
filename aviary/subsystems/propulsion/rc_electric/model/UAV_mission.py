import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.UAV_performance import \
    Battery, ElectronicSpeedController, Motor, PropCoefficients, Propeller, PowerImplicit, Vectorization, PowerResiduals
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
                Aircraft.Engine.Motor.MAX_CONT_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV,
                Dynamic.Vehicle.Propulsion.CURRENT,
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.RPM,
                ('current_constraint', 'current_constraint_nominal'),
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
                # Aircraft.Engine.NUM_ENGINES,
                Dynamic.Atmosphere.DENSITY
                ] + rpm_in,
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER,
                Dynamic.Vehicle.Propulsion.THRUST,
                # Dynamic.Vehicle.Propulsion.THRUST_MAX,
                ]
        )

        if user_feedforward:
            # rpm_defect = optimizer rpm - motor rpm, constrained to 0 below
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
        else:
            # solver mode: motor RPM drives the prop table directly
            self.connect(
                Dynamic.Vehicle.Propulsion.RPM,
                ['propco.' + Dynamic.Vehicle.Propulsion.RPM, 'prop.' + Dynamic.Vehicle.Propulsion.RPM]
            )

        if user_feedforward:
            self.add_subsystem(
                'power_net',
                PowerResiduals(num_nodes=nn),
                promotes_inputs=[
                    Dynamic.Vehicle.Propulsion.PROP_POWER,
                ],
                promotes_outputs=[
                    'power_net',
                ]
            )
        else:
            self.add_subsystem(
                'power_net',
                PowerImplicit(num_nodes=nn),
                promotes_inputs=[
                    Dynamic.Vehicle.Propulsion.PROP_POWER,
                ],
                promotes_outputs=[
                    Dynamic.Vehicle.Propulsion.CURRENT,
                ]
            )
        

        self.add_subsystem(
            'electric_power',
            om.ExecComp(
                'p_elec = v_batt * current',
                p_elec={'val': np.zeros(nn), 'units': 'W'},
                v_batt={'val': 22.2, 'units': 'V'},
                current={'val': np.zeros(nn), 'units': 'A'},
            ),
            promotes_inputs=[
                ('v_batt', Aircraft.Battery.VOLTAGE),
                ('current', Dynamic.Vehicle.Propulsion.CURRENT),
            ],
            promotes_outputs=[('p_elec', Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN)],
        )




        # Max-capability path removed: it was duplicating the nominal motor/prop
        # equations without providing distinct runtime behavior in this example.
        # Keeping these lines commented out avoids the extra branch and the extra
        # max controls/constraints that were not changing the solution.
        #
        # self.add_subsystem(
        #     'battery_max',
        #     Battery(num_nodes=nn),
        #     promotes_inputs=[
        #         Aircraft.Battery.VOLTAGE,
        #         Aircraft.Battery.RESISTANCE,
        #         #(Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
        #         Dynamic.Vehicle.Propulsion.CURRENT,
        #     ]
        # )
        #
        # self.add_subsystem(
        #     'esc_max',
        #     ElectronicSpeedController(num_nodes=nn),
        #     promotes_inputs=[
        #         (Dynamic.Vehicle.Propulsion.THROTTLE, 'full_throttle'),
        #         # (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
        #         Dynamic.Vehicle.Propulsion.CURRENT
        #     ]
        # )
        #
        # self.add_subsystem(
        #     'motor_max',
        #     Motor(num_nodes=nn, load_factor=motor_load_factor),
        #     promotes_inputs=[
        #         Aircraft.Engine.Motor.IDLE_CURRENT,
        #         Aircraft.Engine.Motor.MAX_CONT_CURRENT,
        #         Aircraft.Engine.Motor.RESISTANCE,
        #         Aircraft.Engine.Motor.KV,
        #         # (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
        #         Dynamic.Vehicle.Propulsion.CURRENT,
        #         ],
        #     promotes_outputs=[
        #         (Dynamic.Vehicle.Propulsion.RPM, Dynamic.Vehicle.Propulsion.RPM_MAX),
        #         ('current_constraint', 'current_constraint_max'),
        #         ]
        # )
        #
        # self.add_subsystem(
        #     'propco_max',
        #     PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn),
        #     promotes_inputs=[
        #         Dynamic.Mission.VELOCITY,
        #         'temp_diameter',
        #         'temp_pitch',
        #     ] + rpm_in_max,
        #     promotes_outputs=[('ct', 'ct_max'), ('cp', 'cp_max')]
        # )
        #
        # self.add_subsystem(
        #     'prop_max',
        #     Propeller(num_nodes=nn),
        #     promotes_inputs=[
        #         Aircraft.Engine.Propeller.DIAMETER,
        #         ('ct', 'ct_max'),
        #         ('cp', 'cp_max'),
        #         Dynamic.Atmosphere.DENSITY
        #         ] + rpm_in_max,
        #     promotes_outputs=[
        #         (Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.PROP_POWER_MAX),
        #         (Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Vehicle.Propulsion.THRUST_MAX)
        #         ]
        # )
        #
        # if user_feedforward:
        #     # same idea as rpm_balance, but for the max-power chain
        #     self.add_subsystem(
        #         'rpm_balance_max',
        #         om.ExecComp(
        #             'rpm_defect = rpm_slack_max - rpm_motor',
        #             rpm_defect={'val': np.zeros(nn), 'units': 'rev/s'},
        #             rpm_slack_max={'val': np.zeros(nn), 'units': 'rev/s'},
        #             rpm_motor={'val': np.zeros(nn), 'units': 'rev/s'},
        #             has_diag_partials=True,
        #         ),
        #         promotes_inputs=['rpm_slack_max'],
        #     )
        #     self.connect(Dynamic.Vehicle.Propulsion.RPM_MAX, 'rpm_balance_max.rpm_motor')
        #
        # if user_feedforward:
        #     self.add_subsystem(
        #         'power_net_max',
        #         PowerResiduals(num_nodes=nn),
        #         promotes_inputs=[
        #             (Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.PROP_POWER_MAX),
        #         ],
        #         promotes_outputs=[
        #             ('power_net','power_net_max'),
        #         ]
        #
        #
        #     )
        #     self.connect('battery_max.power', 'power_net_max.power_batt')
        #     self.connect('esc_max.power', 'power_net_max.power_esc')
        #     self.connect('motor_max.power', 'power_net_max.power_motor')
        #
        #
        #
        #
        # self.connect('battery_max.voltage_out', 'esc_max.voltage_in')
        # self.connect('esc_max.voltage_out', 'motor_max.voltage_in')
        # self.connect('esc_max.current_out', 'motor_max.current')
        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')

       
        
        #TODO Alex from phase builder base import add_control

        self.connect('battery.power', 'power_net.power_batt')
        self.connect('esc.power', 'power_net.power_esc')
        self.connect('motor.power', 'power_net.power_motor')

        





        if user_feedforward:
            # Keep electrical power mismatch near zero 
            self.add_constraint('power_net', equals=0.0, units='W')

            # Enforce motor continuous-current limit on nominal branch.
            self.add_constraint('current_constraint_nominal', upper=0, ref=1e2)

            # Force commanded cruise RPM to match motor-computed RPM.
            self.add_constraint('rpm_balance.rpm_defect', equals=0.0)
            # Max-capability constraints removed with the max branch.
        else:
        # NonlinearBlockGS (fixed-point) is used rather than Newton+DirectSolver
        # because the propeller metamodel (PropCoefficients) has flat / extrapolated
        # regions with zero gradient. A DirectSolver factorization of the
        # battery<->esc<->motor<->prop<->power_net current cycle goes singular there,
        # whereas this fixed-point iteration tolerates it.
            self.nonlinear_solver = om.NonlinearBlockGS()
            self.nonlinear_solver.options["maxiter"] = 40
            self.nonlinear_solver.options["use_aitken"] = True
            self.nonlinear_solver.options["err_on_non_converge"] = False

            self.linear_solver = om.LinearBlockGS()

            # Enforce motor continuous-current limit on nominal branch in solver mode.
            self.add_constraint('current_constraint_nominal', upper=0, ref=1)

        
        self.add_constraint('prop.rpm_constraint', upper=0.0, ref=1e2, units='rev/s')
        # self.add_constraint('prop_max.rpm_constraint', upper=0.0, ref=1e2, units='rev/s')

        self.options['auto_order'] = True

        # Newton + DirectSolver alternative (kept for reference): converges the
        # current_flow state in fewer iterations when the metamodel is well inside
        # its trained region, but goes singular on flat extrapolated regions.
        # self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # self.nonlinear_solver.options["maxiter"] = 15
        # self.nonlinear_solver.options["err_on_non_converge"] = False
        # self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        # self.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
        # self.linear_solver = om.DirectSolver(assemble_jac=True)

        # # self.add_constraint(Dynamic.Vehicle.Propulsion.CURRENT, lower=0)
