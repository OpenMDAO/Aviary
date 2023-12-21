import numpy as np
import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.rotation_eom import RotationEOM
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import LowSpeedAero
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class RotationODE(BaseODE):
    """ODE for takeoff rotation.

    This phase spans the time from when the aircraft is touching the runway but has
    begun to rotate to liftoff.
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

        kwargs = {'num_nodes': nn, 'aviary_inputs': aviary_options,
                  'method': 'low_speed'}
        for subsystem in core_subsystems:
            system = subsystem.build_mission(**kwargs)
            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))

        if analysis_scheme is AnalysisScheme.SHOOTING:
            alpha_comp = om.ExecComp(
                'alpha=rotation_rate*(t_curr-start_rotation)+alpha_init',
                alpha=dict(val=0., units='deg'),
                rotation_rate=dict(val=10.0/3.0, units='deg/s'),
                t_curr=dict(val=0., units='s'),
                start_rotation=dict(val=0., units='s'),
                alpha_init=dict(val=0., units='deg'),
            )
            alpha_comp_inputs = ["rotation_rate", "t_curr", "start_rotation",
                                 ("alpha_init", Aircraft.Wing.INCIDENCE)]
            self.add_subsystem("alpha_comp",
                               alpha_comp,
                               promotes_inputs=alpha_comp_inputs,
                               promotes_outputs=["alpha"],
                               )

        self.add_subsystem("eoms", RotationEOM(
            num_nodes=nn, analysis_scheme=analysis_scheme), promotes=["*"])

        ParamPort.set_default_vals(self)
        self.set_input_defaults("t_init_flaps", val=47.5)
        self.set_input_defaults("t_init_gear", val=37.3)
        self.set_input_defaults("alpha", val=np.ones(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                val=np.zeros(nn), units="deg")
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.zeros(nn), units="ft")
        self.set_input_defaults("TAS", val=np.zeros(nn), units="kn")
        self.set_input_defaults("t_curr", val=np.zeros(nn), units="s")
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=1.)
