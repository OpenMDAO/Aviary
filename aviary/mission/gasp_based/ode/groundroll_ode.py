import numpy as np
import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.groundroll_eom import GroundrollEOM
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import LowSpeedAero
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.subsystems.aerodynamics.aerodynamics_builder import AerodynamicsBuilderBase


class GroundrollODE(BaseODE):
    """ODE for takeoff ground roll.

    This phase begins at the point when the aircraft begins accelerating down the runway
    to takeoff, and runs until the aircraft begins to rotate its front tire off the
    runway.
    """

    def setup(self):
        nn = self.options["num_nodes"]
        analysis_scheme = self.options["analysis_scheme"]
        aviary_options = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        # TODO: paramport
        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_subsystem(
            "USatm", USatm1976Comp(
                num_nodes=nn), promotes_inputs=[
                ("h", Dynamic.Mission.ALTITUDE)], promotes_outputs=[
                "rho", ("sos", Dynamic.Mission.SPEED_OF_SOUND), ("temp", Dynamic.Mission.TEMPERATURE), ("pres", Dynamic.Mission.STATIC_PRESSURE), "viscosity"], )

        self.add_subsystem(
            "fc",
            FlightConditions(num_nodes=nn),
            promotes_inputs=["rho", Dynamic.Mission.SPEED_OF_SOUND, "TAS"],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE,
                              Dynamic.Mission.MACH, "EAS"],
        )
        # broadcast scalar i_wing to alpha for aero
        self.add_subsystem("init_alpha",
                           om.ExecComp("alpha = i_wing",
                                       i_wing={"units": "deg", "val": 1.1},
                                       alpha={"units": "deg", "val": 1.1*np.ones(nn)},),
                           promotes=[("i_wing", Aircraft.Wing.INCIDENCE),
                                     "alpha"])

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'low_speed'}
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))
            if type(subsystem) is AerodynamicsBuilderBase:
                self.promotes(
                    subsystem.name,
                    inputs=["alpha"],
                    src_indices=np.zeros(nn, dtype=int),
                )

        self.add_subsystem("eoms", GroundrollEOM(num_nodes=nn, analysis_scheme=analysis_scheme),
                           promotes=["*"])

        self.add_subsystem("exec", om.ExecComp("over_a = TAS / TAS_rate",
                                               TAS_rate={"units": "kn/s",
                                                         "val": np.ones(nn)},
                                               TAS={"units": "kn", "val": np.ones(nn)},
                                               over_a={"units": "s", "val": np.ones(nn)},
                                               has_diag_partials=True,
                                               ),
                           promotes=["*"])

        self.add_subsystem("exec2", om.ExecComp("dt_dv = 1 / TAS_rate",
                                                TAS_rate={"units": "kn/s",
                                                          "val": np.ones(nn)},
                                                dt_dv={"units": "s/kn",
                                                       "val": np.ones(nn)},
                                                has_diag_partials=True,
                                                ),
                           promotes=["*"])

        self.add_subsystem(
            "exec3",
            om.ExecComp(
                "dmass_dv = mass_rate * dt_dv",
                mass_rate={
                    "units": "lbm/s",
                    "val": np.ones(nn)},
                dt_dv={
                    "units": "s/kn",
                    "val": np.ones(nn)},
                dmass_dv={
                    "units": "lbm/kn",
                    "val": np.ones(nn)},
                has_diag_partials=True,
            ),
            promotes_outputs=[
                "dmass_dv",
            ],
            promotes_inputs=[
                ("mass_rate",
                 Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL),
                "dt_dv"])

        ParamPort.set_default_vals(self)
        self.set_input_defaults("t_init_flaps", val=100.)
        self.set_input_defaults("t_init_gear", val=100.)
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.)
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                val=np.zeros(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units="ft")
        self.set_input_defaults("TAS", val=np.zeros(nn), units="kn")
        self.set_input_defaults("TAS_rate", val=np.zeros(nn), units="kn/s")
        self.set_input_defaults("t_curr", val=np.zeros(nn), units="s")
