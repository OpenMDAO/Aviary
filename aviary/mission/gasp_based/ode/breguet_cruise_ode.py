import numpy as np
import openmdao.api as om

from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.breguet_cruise_eom import RangeComp
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.mission.ode.altitude_rate import AltitudeRate
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilderBase
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class BreguetCruiseODESolution(BaseODE):
    """The GASP based cruise ODE"""

    def setup(self):
        nn = self.options["num_nodes"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        # TODO: paramport
        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=nn, input_speed_type=SpeedType.MACH),
            promotes=['*'],
        )

        self.add_subsystem(
            "calc_weight",
            MassToWeight(num_nodes=nn),
            promotes_inputs=["mass"],
            promotes_outputs=["weight"]
        )

        prop_group = om.Group()

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'cruise', 'output_alpha': True}
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                if isinstance(subsystem, PropulsionBuilderBase):
                    prop_group.add_subsystem(subsystem.name,
                                             system,
                                             promotes_inputs=subsystem.mission_inputs(
                                                 **kwargs),
                                             promotes_outputs=subsystem.mission_outputs(**kwargs))
                else:
                    self.add_subsystem(subsystem.name,
                                       system,
                                       promotes_inputs=subsystem.mission_inputs(
                                           **kwargs),
                                       promotes_outputs=subsystem.mission_outputs(**kwargs))

        bal = om.BalanceComp(
            name=Dynamic.Mission.THROTTLE,
            val=np.ones(nn),
            upper=1.0,
            lower=0.0,
            units="unitless",
            lhs_name=Dynamic.Mission.THRUST_TOTAL,
            rhs_name=Dynamic.Mission.DRAG,
            eq_units="lbf",
        )

        prop_group.add_subsystem(
            "thrust_balance",
            subsys=bal,
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        prop_group.linear_solver = om.DirectSolver()

        prop_group.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True,
            maxiter=20,
            rtol=1e-12,
            atol=1e-12,
            err_on_non_converge=False,
        )
        prop_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()

        prop_group.nonlinear_solver.options["iprint"] = 2
        prop_group.linear_solver.options["iprint"] = 2

        self.add_subsystem(
            'prop_group',
            subsys=prop_group,
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        #
        # collect initial/final outputs
        #
        self.add_subsystem(
            "breguet_eom",
            RangeComp(num_nodes=nn),
            promotes_inputs=[
                ("cruise_distance_initial", "initial_distance"),
                ("cruise_time_initial", "initial_time"),
                "mass",
                Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                ("TAS_cruise", Dynamic.Mission.VELOCITY),
            ],
            promotes_outputs=[("cruise_range", Dynamic.Mission.DISTANCE),
                              ("cruise_time", "time")],
        )

        self.add_subsystem(
            name='SPECIFIC_ENERGY_RATE_EXCESS',
            subsys=SpecificEnergyRate(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS,
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
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.VELOCITY],
            promotes_outputs=[
                (Dynamic.Mission.ALTITUDE_RATE,
                 Dynamic.Mission.ALTITUDE_RATE_MAX)])

        ParamPort.set_default_vals(self)
        self.set_input_defaults(
            Dynamic.Mission.ALTITUDE,
            val=37500 * np.ones(nn),
            units="ft")
        self.set_input_defaults("mass", val=np.linspace(
            171481, 171581 - 10000, nn), units="lbm")
