import numpy as np

from aviary.mission.gasp_based.ode.accel_eom import AccelerationRates
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.variable_info.enums import AnalysisScheme, AnalysisScheme
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.mission.gasp_based.ode.time_integration_base_classes import add_SGM_required_inputs, add_SGM_required_outputs


class AccelODE(BaseODE):
    """ODE for level acceleration.

    In level acceleration, there are only nonzero net forces in the direction of motion.
    There is a balance component to solve for the angle of attack necessary to make lift
    equal to weight. Acceleration results from engine thrust in excess of drag.
    """

    def setup(self):
        nn = self.options["num_nodes"]
        analysis_scheme = self.options["analysis_scheme"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        if analysis_scheme is AnalysisScheme.SHOOTING:
            add_SGM_required_inputs(self, {
                't_curr': {'units': 's'},
                Dynamic.Mission.DISTANCE: {'units': 'ft'},
            })
            add_SGM_required_outputs(self, {
                Dynamic.Mission.ALTITUDE_RATE: {'units': 'ft/s'},
            })

        # TODO: paramport
        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_atmosphere(nn)

        self.add_subsystem(
            "calc_weight",
            MassToWeight(num_nodes=nn),
            promotes_inputs=[("mass", Dynamic.Mission.MASS)],
            promotes_outputs=["weight"]
        )

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'cruise', 'output_alpha': True}
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))

        self.add_subsystem(
            "accel_eom",
            AccelerationRates(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.DRAG,
                Dynamic.Mission.THRUST_TOTAL, ],
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.DISTANCE_RATE, ],
        )

        self.add_excess_rate_comps(nn)

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Dynamic.Mission.MASS, val=14e4 *
                                np.ones(nn), units="lbm")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE,
                                val=500 * np.ones(nn), units="ft")
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=200*np.ones(nn),
                                units="m/s")  # val here is nominal
