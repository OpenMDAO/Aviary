import numpy as np
import openmdao.api as om

from aviary.mission.gasp_based.ode.constraints.flight_constraints import FlightConstraints
from aviary.mission.gasp_based.ode.constraints.speed_constraints import SpeedConstraints
from aviary.mission.gasp_based.ode.descent_eom import DescentRates
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.time_integration_base_classes import add_SGM_required_inputs
from aviary.mission.gasp_based.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilderBase
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import AlphaModes, AnalysisScheme, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic


class DescentODE(TwoDOFODE):
    """ODE for quasi-steady descent.

    This ODE has a ``KSComp`` which allows for the switching of obeying an EAS
    constraint cruise Mach constraint, whichever is being violated. It is only included
    if the option to target the cruise speed is set to true, otherwise it is assumed
    that the equivalent airspeed is an input to this ODE and is set further up in the
    model.
    """

    def initialize(self):
        super().initialize()
        self.options.declare(
            'input_speed_type',
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
        self.options.declare('mach_cruise', default=0, desc='targeted cruise Mach number')
        self.options.declare('EAS_limit', default=0, desc='maximum descending EAS in knots')

    def setup(self):
        self.options['auto_order'] = True
        nn = self.options['num_nodes']
        analysis_scheme = self.options['analysis_scheme']
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']
        subsystem_options = self.options['subsystem_options']
        input_speed_type = self.options['input_speed_type']

        flight_condition_group = self

        if analysis_scheme is AnalysisScheme.SHOOTING:
            add_SGM_required_inputs(
                self,
                {
                    't_curr': {'units': 's'},
                    Dynamic.Mission.DISTANCE: {'units': 'ft'},
                    'alt_trigger': {'units': self.options['alt_trigger_units'], 'val': 10e3},
                    'speed_trigger': {'units': self.options['speed_trigger_units'], 'val': 100},
                },
            )
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

        if analysis_scheme is AnalysisScheme.COLLOCATION:
            EAS_limit = self.options['EAS_limit']
            mach_cruise = self.options['mach_cruise']

            # Add a group to contain the balance

            if input_speed_type is SpeedType.MACH:
                mach_balance_group = self.add_subsystem(
                    'mach_balance_group', subsys=om.Group(), promotes=['*']
                )

                mach_balance_group.options['auto_order'] = True
                mach_balance_group.nonlinear_solver = om.NewtonSolver()
                mach_balance_group.nonlinear_solver.options['solve_subsystems'] = True
                mach_balance_group.nonlinear_solver.options['iprint'] = 0
                mach_balance_group.nonlinear_solver.options['atol'] = 1e-7
                mach_balance_group.nonlinear_solver.options['rtol'] = 1e-7
                mach_balance_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
                mach_balance_group.linear_solver = om.DirectSolver(assemble_jac=True)

                speed_bal = om.BalanceComp(
                    name=Dynamic.Atmosphere.MACH,
                    val=mach_cruise * np.ones(nn),
                    units='unitless',
                    lhs_name='KS',
                    rhs_val=0.0,
                    eq_units='unitless',
                    upper=1000,
                    lower=0,
                )
                mach_balance_group.add_subsystem(
                    'speed_bal',
                    speed_bal,
                    promotes_inputs=['KS'],
                    promotes_outputs=[Dynamic.Atmosphere.MACH],
                )

                mach_balance_group.add_subsystem(
                    'speeds',
                    SpeedConstraints(
                        num_nodes=nn,
                        mach_cruise=mach_cruise,
                        EAS_target=EAS_limit,
                    ),
                    promotes_inputs=['EAS', Dynamic.Atmosphere.MACH],
                    promotes_outputs=['speed_constraint'],
                )
                mach_balance_group.add_subsystem(
                    'ks',
                    om.KSComp(width=2, vec_size=nn, units='unitless'),
                    promotes_inputs=[('g', 'speed_constraint')],
                    promotes_outputs=['KS'],
                )
                flight_condition_group = mach_balance_group

            lift_balance_group = self.add_subsystem(
                'lift_balance_group', subsys=om.Group(), promotes=['*']
            )

        elif analysis_scheme is AnalysisScheme.SHOOTING:
            lift_balance_group = self

        flight_condition_group.add_subsystem(
            name='flight_conditions',
            subsys=FlightConditions(num_nodes=nn, input_speed_type=input_speed_type),
            promotes_inputs=['*'],  # + speed_inputs,
            promotes_outputs=['*'],  # [Dynamic.Atmosphere.DYNAMIC_PRESSURE] + speed_outputs,
        )

        # maybe replace this with the solver in add_alpha_control?
        lift_balance_group.nonlinear_solver = om.NewtonSolver()
        lift_balance_group.nonlinear_solver.options['solve_subsystems'] = True
        lift_balance_group.nonlinear_solver.options['iprint'] = 0
        lift_balance_group.nonlinear_solver.options['atol'] = 1e-7
        lift_balance_group.nonlinear_solver.options['rtol'] = 1e-7
        lift_balance_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        lift_balance_group.linear_solver = om.DirectSolver(assemble_jac=True)

        lift_balance_group.add_subsystem(
            'descent_eom',
            DescentRates(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ],
            promotes_outputs=[
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                'required_lift',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ],
        )

        self.add_subsystem(
            'constraints',
            FlightConstraints(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Atmosphere.DENSITY,
                'CL_max',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
            ]
            + ['aircraft:*'],
            promotes_outputs=['theta', 'TAS_violation'],
        )

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options, 'method': 'cruise'}
        # collect the propulsion group names for later use
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

        self.add_alpha_control(
            alpha_group=lift_balance_group,
            alpha_mode=AlphaModes.REQUIRED_LIFT,
            add_default_solver=False,
            num_nodes=nn,
        )

        # the last two subsystems will also be used for constraints
        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=37500 * np.ones(nn), units='ft')
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=147000 * np.ones(nn), units='lbm')
        self.set_input_defaults(Dynamic.Atmosphere.MACH, val=0 * np.ones(nn), units='unitless')
        self.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THROTTLE, val=0 * np.ones(nn), units='unitless'
        )

        self.set_input_defaults(Aircraft.Wing.AREA, val=1.0, units='ft**2')
