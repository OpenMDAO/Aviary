import numpy as np
import openmdao.api as om

from aviary.mission.two_dof.ode.constraints.flight_constraints import FlightConstraints
from aviary.mission.two_dof.ode.constraints.speed_constraints import SpeedConstraints
from aviary.mission.two_dof.ode.flight_eom import EOMRates
from aviary.mission.two_dof.ode.params import ParamPort
from aviary.mission.two_dof.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilder
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import AlphaModes, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic


class FlightODE(TwoDOFODE):
    """ODE for quasi-steady flight in GASP 2-dof. This replaces ClimbODE and DescentODE.

    This ODE has a ``KSComp`` which aggregates the maximum EAS and mach number constraints.
    This allows a single constraint to whichever of the two constraints would be active at
    each trajectory point. The KSComp is only included if the input_speed_type is set to
    SpeedType.EAS for this phase.
    """

    def initialize(self):
        super().initialize()
        self.options.declare(
            'input_speed_type',
            default=SpeedType.MACH,
            types=SpeedType,
            desc='Whether the speed is given as a equivalent airspeed, true airspeed, or Mach number',
        )
        self.options.declare('mach_target', default=0, desc='Targeted cruise Mach number')
        self.options.declare('EAS_target', desc='Targeted EAS in knots')

    def setup(self):
        self.options['auto_order'] = True
        nn = self.options['num_nodes']
        aviary_options = self.options['aviary_options']
        subsystems = self.options['subsystems']
        subsystem_options = self.options['subsystem_options']
        input_speed_type = self.options['input_speed_type']

        if input_speed_type is SpeedType.EAS:
            speed_inputs = ['EAS']
            speed_outputs = ['mach', Dynamic.Mission.VELOCITY]
        elif input_speed_type is SpeedType.MACH:
            speed_inputs = ['mach']
            speed_outputs = ['EAS', Dynamic.Mission.VELOCITY]

        # TODO: Let's get rid of this paramport
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
                Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
            ],
        )

        EAS_target = self.options['EAS_target']
        mach_target = self.options['mach_target']

        # Either set a target or a limit.

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
                val=mach_target * np.ones(nn),
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
                    mach_target=mach_target,
                    EAS_target=EAS_target,
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

        else:
            flight_condition_group = self

        lift_balance_group = self.add_subsystem(
            'lift_balance_group', subsys=om.Group(), promotes=['*']
        )

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
            'flight_eom',
            EOMRates(num_nodes=nn),
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
        for subsystem in subsystems:
            # check if subsystem_options has entry for a subsystem of this name
            if subsystem.name in subsystem_options:
                kwargs.update(subsystem_options[subsystem.name])
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                if isinstance(subsystem, AerodynamicsBuilder):
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

        self.add_alpha_control(
            alpha_group=lift_balance_group,
            alpha_mode=AlphaModes.REQUIRED_LIFT,
            add_default_solver=False,
            num_nodes=nn,
        )

        # the last two subsystems will also be used for constraints
        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='ft')
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=np.ones(nn), units='lbm')
        self.set_input_defaults(Dynamic.Atmosphere.MACH, val=0 * np.ones(nn), units='unitless')
        self.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THROTTLE, val=0 * np.ones(nn), units='unitless'
        )

        self.set_input_defaults(Aircraft.Wing.AREA, val=1.0, units='ft**2')
