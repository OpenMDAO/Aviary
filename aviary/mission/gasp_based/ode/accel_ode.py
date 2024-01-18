import numpy as np
import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.flight_conditions import FlightConditions
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

        self.add_subsystem(
            "flight_conditions",
            FlightConditions(num_nodes=nn, input_speed_type=SpeedType.TAS),
            promotes_inputs=["rho", Dynamic.Mission.SPEED_OF_SOUND, "TAS"],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE,
                              Dynamic.Mission.MACH, "EAS"],
        )

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
            "eom",
            AccelerationRates(
                num_nodes=nn,
                analysis_scheme=analysis_scheme),
            promotes_inputs=[
                Dynamic.Mission.MASS,
                "TAS",
                Dynamic.Mission.DRAG,
                Dynamic.Mission.THRUST_TOTAL, ]
            + sgm_inputs,
            promotes_outputs=[
                "TAS_rate",
                "distance_rate", ]
            + sgm_outputs,
        )

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

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Dynamic.Mission.MASS, val=14e4 *
                                np.ones(nn), units="lbm")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE,
                                val=500 * np.ones(nn), units="ft")
        self.set_input_defaults("TAS", val=200*np.ones(nn),
                                units="m/s")  # val here is nominal
