import numpy as np
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.ode.accel_eom import AccelerationRates
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.subsystems.mass.mass_to_weight import MassToWeight
from aviary.variable_info.enums import AnalysisScheme, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.mission.ode.altitude_rate import AltitudeRate


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

        self.add_flight_conditions(nn)

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

        sgm_inputs = [
            't_curr', Dynamic.Mission.DISTANCE] if analysis_scheme is AnalysisScheme.SHOOTING else []
        sgm_outputs = [
            Dynamic.Mission.ALTITUDE_RATE] if analysis_scheme is AnalysisScheme.SHOOTING else []

        self.add_subsystem(
            "accel_eom",
            AccelerationRates(
                num_nodes=nn,
                analysis_scheme=analysis_scheme),
            promotes_inputs=[
                Dynamic.Mission.MASS,
                Dynamic.Mission.VELOCITY,
                Dynamic.Mission.DRAG,
                Dynamic.Mission.THRUST_TOTAL, ]
            + sgm_inputs,
            promotes_outputs=[
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.DISTANCE_RATE, ]
            + sgm_outputs,
        )

        self.add_excess_rate_comps(nn)
        if analysis_scheme is AnalysisScheme.SHOOTING:
            from aviary.utils.functions import create_printcomp
            dummy_comp = create_printcomp(
                all_inputs=[
                    Dynamic.Mission.DISTANCE,
                    Dynamic.Mission.THROTTLE,
                    Dynamic.Mission.THRUST_TOTAL,
                    Dynamic.Mission.DRAG,
                    Dynamic.Mission.ALTITUDE,
                    Dynamic.Mission.FLIGHT_PATH_ANGLE,
                    Dynamic.Mission.LIFT,
                    Dynamic.Mission.MASS,
                ],
                input_units={
                    Dynamic.Mission.FLIGHT_PATH_ANGLE: 'deg',
                })
            self.add_subsystem(
                "dummy_comp",
                dummy_comp(),
                promotes_inputs=["*"],)
            self.set_input_defaults(
                Dynamic.Mission.DISTANCE, val=0, units='NM')

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Dynamic.Mission.MASS, val=14e4 *
                                np.ones(nn), units="lbm")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE,
                                val=500 * np.ones(nn), units="ft")
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=200*np.ones(nn),
                                units="m/s")  # val here is nominal
        from aviary.mission.gasp_based.ode.time_integration_base_classes import killer_comp
        self.add_subsystem('accel_killer', killer_comp())
