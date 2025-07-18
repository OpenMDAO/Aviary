import numpy as np
import openmdao.api as om

from aviary.mission.gasp_based.ode.climb_eom import ClimbRates
from aviary.mission.gasp_based.ode.constraints.flight_constraints import FlightConstraints
from aviary.mission.gasp_based.ode.constraints.speed_constraints import SpeedConstraints
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilderBase
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import AlphaModes, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic


class ClimbODE(TwoDOFODE):
    """ODE for quasi-steady climb.

    This ODE has a ``KSComp`` which allows for the switching of obeying an EAS
    constraint or a cruise Mach constraint, whichever is being violated.
    """

    def initialize(self):
        super().initialize()
        self.options.declare(
            'input_speed_type',
            default=SpeedType.EAS,
            types=SpeedType,
            desc='Whether the speed is given as a equivalent airspeed, true airspeed, or Mach number',
        )
        self.options.declare(
            'alt_trigger_units',
            default='ft',
            desc='The units that the altitude trigger is provided in',
        )
        self.options.declare(
            'speed_trigger_units',
            default='kn',
            desc='The units that the speed trigger is provided in.',
        )
        self.options.declare(
            'input_speed_type',
            default=SpeedType.EAS,
            types=SpeedType,
            desc='Whether the speed is given as a equivalent airspeed, true airspeed, or Mach number',
        )
        self.options.declare('EAS_target', desc='target climbing EAS in knots')
        self.options.declare('mach_cruise', default=0, desc='targeted cruise Mach number')

    def setup(self):
        self.options['auto_order'] = True
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']
        subsystem_options = self.options['subsystem_options']
        input_speed_type = self.options['input_speed_type']

        if input_speed_type is SpeedType.EAS:
            speed_inputs = ['EAS']
            speed_outputs = ['mach', Dynamic.Mission.VELOCITY]
        elif input_speed_type is SpeedType.MACH:
            speed_inputs = ['mach']
            speed_outputs = ['EAS', Dynamic.Mission.VELOCITY]

        # TODO: paramport
        self.add_subsystem('params', ParamPort(), promotes=['*'])

        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.ALTITUDE],
            promotes_outputs=[
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                Dynamic.Atmosphere.TEMPERATURE,
                Dynamic.Atmosphere.STATIC_PRESSURE,
                'viscosity',
            ],
        )

        EAS_target = self.options['EAS_target']
        mach_cruise = self.options['mach_cruise']

        mach_balance_group = self.add_subsystem(
            'mach_balance_group', subsys=om.Group(), promotes=['*']
        )

        mach_balance_group.nonlinear_solver = om.NewtonSolver()
        mach_balance_group.nonlinear_solver.options['solve_subsystems'] = True
        mach_balance_group.nonlinear_solver.options['iprint'] = 0
        mach_balance_group.nonlinear_solver.options['atol'] = 1e-7
        mach_balance_group.nonlinear_solver.options['rtol'] = 1e-7
        mach_balance_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        mach_balance_group.linear_solver = om.DirectSolver(assemble_jac=True)
        mach_balance_group.add_subsystem(
            'speeds',
            SpeedConstraints(num_nodes=nn, EAS_target=EAS_target, mach_cruise=mach_cruise),
            promotes_inputs=['EAS', Dynamic.Atmosphere.MACH],
            promotes_outputs=['speed_constraint'],
        )
        mach_balance_group.add_subsystem(
            'ks',
            om.KSComp(width=2, vec_size=nn, units='unitless'),
            promotes_inputs=[('g', 'speed_constraint')],
            promotes_outputs=['KS'],
        )
        speed_bal = om.BalanceComp(
            name='EAS',
            val=EAS_target * np.ones(nn),
            units='kn',
            lhs_name='KS',
            rhs_val=0.0,
            eq_units='unitless',
            upper=350,
            lower=0,
        )
        mach_balance_group.add_subsystem(
            'speed_bal',
            speed_bal,
            promotes_inputs=['KS'],
            promotes_outputs=['EAS'],
        )

        lift_balance_group = self.add_subsystem(
            'lift_balance_group', subsys=om.Group(), promotes=['*']
        )
        flight_condition_group = mach_balance_group

        flight_condition_group.add_subsystem(
            name='flight_conditions',
            subsys=FlightConditions(num_nodes=nn, input_speed_type=input_speed_type),
            promotes_inputs=[
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
            ]
            + speed_inputs,
            promotes_outputs=[Dynamic.Atmosphere.DYNAMIC_PRESSURE] + speed_outputs,
        )

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options, 'method': 'cruise'}
        # collect the propulsion group names for later use with
        for subsystem in core_subsystems:
            # check if subsystem_options has entry for a subsystem of this name
            if subsystem.name in subsystem_options:
                kwargs.update(subsystem_options[subsystem.name])
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                if isinstance(subsystem, AerodynamicsBuilderBase):
                    lift_balance_group.add_subsystem(
                        subsystem.name,
                        system,
                        promotes_inputs=subsystem.mission_inputs(**kwargs),
                        promotes_outputs=subsystem.mission_outputs(**kwargs),
                    )
                else:
                    self.add_subsystem(
                        subsystem.name,
                        system,
                        promotes_inputs=subsystem.mission_inputs(**kwargs),
                        promotes_outputs=subsystem.mission_outputs(**kwargs),
                    )

        self.add_external_subsystems()

        # maybe replace this with the solver in add_alpha_control?
        lift_balance_group.nonlinear_solver = om.NewtonSolver()
        lift_balance_group.nonlinear_solver.options['solve_subsystems'] = True
        lift_balance_group.nonlinear_solver.options['iprint'] = 0
        lift_balance_group.nonlinear_solver.options['atol'] = 1e-7
        lift_balance_group.nonlinear_solver.options['rtol'] = 1e-7
        lift_balance_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        lift_balance_group.linear_solver = om.DirectSolver(assemble_jac=True)

        lift_balance_group.add_subsystem(
            'climb_eom',
            ClimbRates(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            promotes_outputs=[
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                'required_lift',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ],
        )

        self.add_alpha_control(
            alpha_group=lift_balance_group,
            alpha_mode=AlphaModes.REQUIRED_LIFT,
            add_default_solver=False,
            num_nodes=nn,
        )

        self.add_subsystem(
            'constraints',
            FlightConstraints(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Atmosphere.DENSITY,
                'CL_max',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.VELOCITY,
            ]
            + ['aircraft:*'],
            promotes_outputs=['theta', 'TAS_violation'],
        )

        # the last two subsystems will also be used for constraints
        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults('CL_max', val=5 * np.ones(nn), units='unitless')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=500 * np.ones(nn), units='ft')
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=174000 * np.ones(nn), units='lbm')
        self.set_input_defaults(Dynamic.Atmosphere.MACH, val=0 * np.ones(nn), units='unitless')

        self.set_input_defaults(Aircraft.Wing.AREA, val=1.0, units='ft**2')
