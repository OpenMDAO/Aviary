from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.phases.landing_components import (
    GlideConditionComponent, LandingAltitudeComponent,
    LandingGroundRollComponent)
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import LowSpeedAero
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LandingSegment(BaseODE):
    def setup(self):
        # TODO: paramport
        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_subsystem(
            "approach_alt_comp",
            LandingAltitudeComponent(),
            promotes_inputs=[
                Mission.Landing.OBSTACLE_HEIGHT,
                Mission.Landing.AIRPORT_ALTITUDE,
            ],
            promotes_outputs=[Mission.Landing.INITIAL_ALTITUDE],
        )

        self.add_subsystem(
            "USatm_app",
            USatm1976Comp(num_nodes=1),
            promotes_inputs=[("h", Mission.Landing.INITIAL_ALTITUDE)],
            promotes_outputs=[
                ("rho", "rho_app"),
                ("sos", "sos_app"),
                ("temp", "T_app"),
                ("pres", "P_app"),
                ("viscosity", "viscosity_app"),
            ],
        )

        self.add_subsystem(
            "fc_app",
            FlightConditions(num_nodes=1, input_speed_type=SpeedType.MACH),
            promotes_inputs=[
                ("rho", "rho_app"),
                (Dynamic.Mission.SPEED_OF_SOUND, "sos_app"),
                (Dynamic.Mission.MACH, Mission.Landing.INITIAL_MACH),
            ],
            promotes_outputs=[(Dynamic.Mission.DYNAMIC_PRESSURE, "q_app")],
        )

        propulsion_mission = self.add_subsystem(
            name='propulsion',
            subsys=PropulsionMission(
                num_nodes=1,
                aviary_options=self.options['aviary_options']),
            promotes_inputs=["*", (Dynamic.Mission.ALTITUDE, Mission.Landing.INITIAL_ALTITUDE),
                             (Dynamic.Mission.MACH, Mission.Landing.INITIAL_MACH)],
            promotes_outputs=[(Dynamic.Mission.THRUST_TOTAL, "thrust_idle")])
        propulsion_mission.set_input_defaults(Dynamic.Mission.THROTTLE, 0.0)

        # alpha input not needed, only used for CL_max
        self.add_subsystem(
            "aero_app",
            LowSpeedAero(num_nodes=1),
            promotes_inputs=[
                "*",
                (Dynamic.Mission.ALTITUDE, Mission.Landing.INITIAL_ALTITUDE),
                ("rho", "rho_app"),
                (Dynamic.Mission.SPEED_OF_SOUND, "sos_app"),
                ("viscosity", "viscosity_app"),
                ("airport_alt", Mission.Landing.AIRPORT_ALTITUDE),
                (Dynamic.Mission.MACH, Mission.Landing.INITIAL_MACH),
                (Dynamic.Mission.DYNAMIC_PRESSURE, "q_app"),
                ("flap_defl", Aircraft.Wing.FLAP_DEFLECTION_LANDING),
                ("t_init_flaps", "t_init_flaps_app"),
                ("t_init_gear", "t_init_gear_app"),
                ("CL_max_flaps", Mission.Landing.LIFT_COEFFICIENT_MAX),
                (
                    "dCL_flaps_model",
                    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
                ),
                (
                    "dCD_flaps_model",
                    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
                ),
            ],
            promotes_outputs=["CL_max"],
        )

        self.add_subsystem(
            "glide",
            GlideConditionComponent(),
            promotes_inputs=[
                "rho_app",
                Mission.Landing.MAXIMUM_SINK_RATE,
                Dynamic.Mission.MASS,
                Aircraft.Wing.AREA,
                Mission.Landing.GLIDE_TO_STALL_RATIO,
                "CL_max",
                Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
                Mission.Landing.TOUCHDOWN_SINK_RATE,
                Mission.Landing.INITIAL_ALTITUDE,
                Mission.Landing.BRAKING_DELAY,
            ],
            promotes_outputs=[
                Mission.Landing.INITIAL_VELOCITY,
                Mission.Landing.STALL_VELOCITY,
                "TAS_touchdown",
                "density_ratio",
                "wing_loading_land",
                "theta",
                "glide_distance",
                "tr_distance",
                "delay_distance",
                "flare_alt",
            ],
        )

        self.add_subsystem(
            "USatm_td",
            USatm1976Comp(num_nodes=1),
            promotes_inputs=[("h", Mission.Landing.AIRPORT_ALTITUDE)],
            promotes_outputs=[
                ("rho", "rho_td"),
                ("sos", "sos_td"),
                ("viscosity", "viscosity_td"),
            ],
        )

        self.add_subsystem(
            "fc_td",
            FlightConditions(num_nodes=1),
            promotes_inputs=[
                ("rho", "rho_td"),
                (Dynamic.Mission.SPEED_OF_SOUND, "sos_td"),
                ("TAS", "TAS_touchdown"),
            ],
            promotes_outputs=[(Dynamic.Mission.DYNAMIC_PRESSURE, "q_td"),
                              (Dynamic.Mission.MACH, "mach_td")],
        )

        self.add_subsystem(
            "aero_td",
            LowSpeedAero(num_nodes=1, retract_flaps=True, retract_gear=False),
            promotes_inputs=[
                "*",
                (Dynamic.Mission.ALTITUDE, Mission.Landing.AIRPORT_ALTITUDE),
                ("rho", "rho_td"),
                (Dynamic.Mission.SPEED_OF_SOUND, "sos_td"),
                ("viscosity", "viscosity_td"),
                ("airport_alt", Mission.Landing.AIRPORT_ALTITUDE),
                (Dynamic.Mission.MACH, "mach_td"),
                (Dynamic.Mission.DYNAMIC_PRESSURE, "q_td"),
                ("alpha", Aircraft.Wing.INCIDENCE),
                ("flap_defl", Aircraft.Wing.FLAP_DEFLECTION_LANDING),
                ("CL_max_flaps", Mission.Landing.LIFT_COEFFICIENT_MAX),
                (
                    "dCL_flaps_model",
                    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
                ),
                (
                    "dCD_flaps_model",
                    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
                ),
                ("t_init_flaps", "t_init_flaps_td"),
                ("t_init_gear", "t_init_gear_td"),
            ],
            promotes_outputs=[("CD", "touchdown_CD"), ("CL", "touchdown_CL")],
        )
        # GASP seems to run groundroll with flaps up and gear down (IWLD=2)
        self.set_input_defaults("t_init_flaps_td", 1e10)  # never deploy
        self.set_input_defaults("t_init_gear_td", 1e10)  # ensure gear down

        self.add_subsystem(
            "landinggroundroll",
            LandingGroundRollComponent(),
            promotes_inputs=[
                "touchdown_CD",
                "touchdown_CL",
                "TAS_touchdown",
                "thrust_idle",
                "density_ratio",
                "wing_loading_land",
                "glide_distance",
                "tr_distance",
                "delay_distance",
                "CL_max",
                Dynamic.Mission.MASS,
                'mission:*'
            ],
            promotes_outputs=[
                "ground_roll_distance", "average_acceleration", 'mission:*'],
        )

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Mission.Landing.INITIAL_MACH, val=0.1)
        # landing doesn't change flap or gear position
        self.set_input_defaults("t_init_flaps_app", val=1e10)
        self.set_input_defaults("t_init_gear_app", val=1e10)
        self.set_input_defaults(
            Mission.Landing.INITIAL_ALTITUDE, val=50, units="ft")
        self.set_input_defaults('aero_ramps.flap_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.gear_factor:final_val', val=1.)
        self.set_input_defaults('aero_ramps.flap_factor:initial_val', val=0.)
        self.set_input_defaults('aero_ramps.gear_factor:initial_val', val=0.)
