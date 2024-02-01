import numpy as np
import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.climb_eom import ClimbRates
from aviary.mission.gasp_based.ode.constraints.flight_constraints import \
    FlightConstraints
from aviary.mission.gasp_based.ode.constraints.speed_constraints import SpeedConstraints
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilderBase
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilderBase
from aviary.variable_info.enums import AnalysisScheme, AlphaModes, SpeedType
from aviary.variable_info.variables import Dynamic
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.mission.ode.altitude_rate import AltitudeRate


class ClimbODE(BaseODE):
    """ODE for quasi-steady climb.

    This ODE has a ``KSComp`` which allows for the switching of obeying an EAS
    constraint or a cruise Mach constraint, whichever is being violated.
    """

    def initialize(self):
        super().initialize()
        self.options.declare("input_speed_type", default=SpeedType.EAS, types=SpeedType,
                             desc="Whether the speed is given as a equivalent airspeed, true airspeed, or mach number")
        self.options.declare("alt_trigger_units", default='ft',
                             desc='The units that the altitude trigger is provided in')
        self.options.declare("speed_trigger_units", default='kn',
                             desc='The units that the speed trigger is provided in.')
        self.options.declare("input_speed_type", default=SpeedType.EAS, types=SpeedType,
                             desc="Whether the speed is given as a equivalent airspeed, true airspeed, or mach number")
        self.options.declare("EAS_target", desc="target climbing EAS in knots")
        self.options.declare(
            "mach_cruise", default=0, desc="targeted cruise mach number"
        )

    def setup(self):
        nn = self.options["num_nodes"]
        analysis_scheme = self.options["analysis_scheme"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']
        input_speed_type = self.options["input_speed_type"]

        if input_speed_type is SpeedType.EAS:
            speed_inputs = ["EAS"]
            speed_outputs = ["mach", "TAS"]
        elif input_speed_type is SpeedType.MACH:
            speed_inputs = ["mach"]
            speed_outputs = ["EAS", "TAS"]

        # TODO: paramport
        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_subsystem(
            "USatm",
            USatm1976Comp(
                num_nodes=nn),
            promotes_inputs=[
                ("h",
                 Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                "rho",
                ("sos",
                 Dynamic.Mission.SPEED_OF_SOUND),
                ("temp",
                 Dynamic.Mission.TEMPERATURE),
                ("pres",
                 Dynamic.Mission.STATIC_PRESSURE),
                "viscosity"],
        )

        if analysis_scheme is AnalysisScheme.COLLOCATION:
            EAS_target = self.options["EAS_target"]
            mach_cruise = self.options["mach_cruise"]

            constraint_args = {}
            integration_states = []
            constraint_inputs = []

            mach_balance_group = self.add_subsystem(
                "mach_balance_group", subsys=om.Group(), promotes=["*"]
            )

            mach_balance_group.nonlinear_solver = om.NewtonSolver()
            mach_balance_group.nonlinear_solver.options["solve_subsystems"] = True
            mach_balance_group.nonlinear_solver.options["iprint"] = 0
            mach_balance_group.nonlinear_solver.options["atol"] = 1e-7
            mach_balance_group.nonlinear_solver.options["rtol"] = 1e-7
            mach_balance_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
            mach_balance_group.linear_solver = om.DirectSolver(assemble_jac=True)
            mach_balance_group.add_subsystem(
                "speeds",
                SpeedConstraints(
                    num_nodes=nn, EAS_target=EAS_target, mach_cruise=mach_cruise
                ),
                promotes_inputs=["EAS", Dynamic.Mission.MACH],
                promotes_outputs=["speed_constraint"],
            )
            mach_balance_group.add_subsystem(
                "ks",
                om.KSComp(width=2, vec_size=nn, units="unitless"),
                promotes_inputs=[("g", "speed_constraint")],
                promotes_outputs=["KS"],
            )
            speed_bal = om.BalanceComp(
                name="EAS",
                val=EAS_target * np.ones(nn),
                units="kn",
                lhs_name="KS",
                rhs_val=0.0,
                eq_units='unitless',
                upper=350,
                lower=0,
            )
            mach_balance_group.add_subsystem(
                "speed_bal",
                speed_bal,
                promotes_inputs=["KS"],
                promotes_outputs=["EAS"],
            )

            lift_balance_group = self.add_subsystem(
                "lift_balance_group", subsys=om.Group(), promotes=["*"]
            )
            flight_condition_group = mach_balance_group

        elif analysis_scheme is AnalysisScheme.SHOOTING:
            constraint_args = {'analysis_scheme': AnalysisScheme.SHOOTING,
                               'alt_trigger_units': self.options["alt_trigger_units"],
                               'speed_trigger_units': self.options["speed_trigger_units"]}

            integration_states = ["t_curr", Dynamic.Mission.DISTANCE]
            constraint_inputs = ["alt_trigger", "speed_trigger"]

            lift_balance_group = self
            flight_condition_group = self

        flight_condition_group.add_subsystem(
            "fc",
            FlightConditions(num_nodes=nn, input_speed_type=input_speed_type),
            promotes_inputs=["rho", Dynamic.Mission.SPEED_OF_SOUND] + speed_inputs,
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE,] + speed_outputs,
        )

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'cruise'}
        # collect the propulsion group names for later use with
        prop_groups = []
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                if isinstance(subsystem, AerodynamicsBuilderBase):
                    lift_balance_group.add_subsystem(subsystem.name,
                                                     system,
                                                     promotes_inputs=subsystem.mission_inputs(
                                                         **kwargs),
                                                     promotes_outputs=subsystem.mission_outputs(**kwargs))
                else:
                    if isinstance(subsystem, PropulsionBuilderBase):
                        prop_groups.append(subsystem.name)

                    self.add_subsystem(subsystem.name,
                                       system,
                                       promotes_inputs=subsystem.mission_inputs(
                                           **kwargs),
                                       promotes_outputs=subsystem.mission_outputs(**kwargs))

        # maybe replace this with the solver in AddAlphaControl?
        lift_balance_group.nonlinear_solver = om.NewtonSolver()
        lift_balance_group.nonlinear_solver.options["solve_subsystems"] = True
        lift_balance_group.nonlinear_solver.options["iprint"] = 0
        lift_balance_group.nonlinear_solver.options["atol"] = 1e-7
        lift_balance_group.nonlinear_solver.options["rtol"] = 1e-7
        lift_balance_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        lift_balance_group.linear_solver = om.DirectSolver(assemble_jac=True)

        lift_balance_group.add_subsystem(
            "eom",
            ClimbRates(
                num_nodes=nn,
                analysis_scheme=analysis_scheme),
            promotes_inputs=[
                Dynamic.Mission.MASS,
                "TAS",
                Dynamic.Mission.DRAG,
                Dynamic.Mission.THRUST_TOTAL,] +
            integration_states,
            promotes_outputs=[
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.DISTANCE_RATE,
                "required_lift",
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ],
        )

        self.AddAlphaControl(
            alpha_group=lift_balance_group,
            alpha_mode=AlphaModes.REQUIRED_LIFT,
            add_default_solver=False,
            num_nodes=nn)

        self.add_subsystem(
            "constraints",
            FlightConstraints(num_nodes=nn, **constraint_args),
            promotes_inputs=[
                "alpha",
                "rho",
                "CL_max",
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.MASS,
                "TAS",
            ]
            + ["aircraft:*"]
            + constraint_inputs,
            promotes_outputs=["theta", "TAS_violation"],
        )

        # the last two subsystems will also be used for constraints
        self.add_subsystem(
            name='SPECIFIC_ENERGY_RATE_EXCESS',
            subsys=SpecificEnergyRate(num_nodes=nn),
            promotes_inputs=[(Dynamic.Mission.VELOCITY, "TAS"), Dynamic.Mission.MASS,
                             (Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.THRUST_MAX_TOTAL),
                             Dynamic.Mission.DRAG],
            promotes_outputs=[(Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                               Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS)]
        )

        self.add_subsystem(
            name='ALTITUDE_RATE_MAX',
            subsys=AltitudeRate(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                 Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS),
                (Dynamic.Mission.VELOCITY_RATE, "TAS_rate"),
                (Dynamic.Mission.VELOCITY, "TAS")],
            promotes_outputs=[
                (Dynamic.Mission.ALTITUDE_RATE,
                 Dynamic.Mission.ALTITUDE_RATE_MAX)])

        if analysis_scheme is AnalysisScheme.COLLOCATION:
            self.set_order(['params', 'USatm', 'mach_balance_group'] + prop_groups +
                           ['lift_balance_group', 'constraints', 'SPECIFIC_ENERGY_RATE_EXCESS', 'ALTITUDE_RATE_MAX'])

        ParamPort.set_default_vals(self)
        self.set_input_defaults("CL_max", val=5 * np.ones(nn), units="unitless")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE,
                                val=500 * np.ones(nn), units='ft')
        self.set_input_defaults(Dynamic.Mission.MASS,
                                val=174000 * np.ones(nn), units='lbm')
        self.set_input_defaults(Dynamic.Mission.MACH,
                                val=0 * np.ones(nn), units="unitless")
