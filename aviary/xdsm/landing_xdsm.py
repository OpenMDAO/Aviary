from pyxdsm.XDSM import FUNC, GROUP, METAMODEL, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

show_outputs = True

# Create subsystem components
x.add_system("landing_alt", FUNC, [r"\textbf{LandingAlt}"])
x.add_system("atmos_approach", FUNC, ["ApproachAtmosphere"])
x.add_system("fc_approach", FUNC, ["ApproachFlightConditions"])
x.add_system("prop_approach", GROUP, ["ApproachPropulsion"])
x.add_system("aero_approach", GROUP, ["ApproachLowSpeedAero", "(alpha~in)"])
x.add_system("glide", FUNC, [r"\textbf{GlideCondition}"])
x.add_system("atmos_touchdown", FUNC, ["TouchdownAtmosphere"])
x.add_system("fc_touchdown", FUNC, ["TouchdownFlightConditions"])
x.add_system("aero_touchdown", GROUP, ["TouchdownLowSpeedAero"])
x.add_system("groundroll", FUNC, [r"\textbf{GroundRoll}"])

# create inputs
x.add_input("landing_alt", [
    Mission.Landing.OBSTACLE_HEIGHT,
    Mission.Landing.AIRPORT_ALTITUDE,
])
x.add_input("fc_approach", [Mission.Landing.INITIAL_MACH])
x.add_input("prop_approach", [
    Dynamic.Mission.THROTTLE,
    Mission.Landing.INITIAL_MACH,
])
x.add_input("aero_approach", [
    "aero_app.alpha",
    "t_init_flaps",
    "t_init_gear",
    Mission.Landing.INITIAL_MACH,
    Mission.Landing.AIRPORT_ALTITUDE,
    Mission.Design.GROSS_MASS,
    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,  # dCL_flaps_model
    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,  # dCD_flaps_model
    # "flap_defl",  # Aircraft.Wing.FLAP_DEFLECTION_LANDING
    "aircraft:*",
])
x.add_input("glide", [
    Mission.Landing.MAXIMUM_SINK_RATE,
    Mission.Landing.GLIDE_TO_STALL_RATIO,
    Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
    Mission.Landing.TOUCHDOWN_SINK_RATE,
    Mission.Landing.BRAKING_DELAY,
    Dynamic.Mission.MASS,
    Aircraft.Wing.AREA,
])
x.add_input("atmos_touchdown", [Mission.Landing.AIRPORT_ALTITUDE])
x.add_input("aero_touchdown", [
    Aircraft.Wing.INCIDENCE,
    Mission.Landing.AIRPORT_ALTITUDE,
    Aircraft.Wing.AREA,
    Aircraft.Wing.HEIGHT,
    Aircraft.Wing.SPAN,
    Mission.Design.GROSS_MASS,
    Aircraft.Wing.FLAP_DEFLECTION_LANDING,  # flap_defl
])
x.add_input("groundroll", ["mass"])

# make connections
x.connect("landing_alt", "atmos_approach", [Mission.Landing.INITIAL_ALTITUDE])
x.connect("landing_alt", "prop_approach", [Mission.Landing.INITIAL_ALTITUDE])
x.connect("landing_alt", "aero_approach", [Mission.Landing.INITIAL_ALTITUDE])
x.connect("landing_alt", "glide", [Mission.Landing.INITIAL_ALTITUDE])

x.connect("atmos_approach", "fc_approach", [
    "rho_app",
    "speed_of_sound_app",  # Dynamic.Mission.SPEED_OF_SOUND
])
# x.connect("atmos_approach", "prop_approach", ["T_app", "P_app"])
x.connect("atmos_approach", "glide", ["rho_app"])

x.connect("fc_approach", "aero_approach", [
    "dynamic_pressure_app",  # Dynamic.Mission.DYNAMIC_PRESSURE
])

x.connect("prop_approach", "groundroll", ["thrust_idle"])

x.connect("aero_approach", "glide", ["CL_max"])
x.connect("aero_approach", "groundroll", ["CL_max"])

x.connect("glide", "groundroll", [
    Mission.Landing.STALL_VELOCITY,
    "TAS_touchdown",
    "density_ratio",
    "wing_loading_land",
    "glide_distance",
    "tr_distance",  # transition distance
    "delay_distance",
])
x.connect("glide", "fc_touchdown", ["TAS_touchdown"])

x.connect("atmos_touchdown", "fc_touchdown", [
    "rho_touchdown",
    "speed_of_sound_touchdown",
])

x.connect("fc_touchdown", "aero_touchdown", [
    "mach_touchdown",
    "dynamic_pressure_touchdown",
])

x.connect("aero_touchdown", "groundroll", ["touchdown_CD", "touchdown_CL"])

# create outputs
if show_outputs:
    x.add_output("glide", [
        Mission.Landing.INITIAL_VELOCITY,
        "theta",
        "flare_alt",
    ], side="right")
    x.add_output("groundroll", [
        # Mission.Landing.INITIAL_VELOCITY,
        "ground_roll_distance",
        Mission.Landing.GROUND_DISTANCE,
        "average_acceleration",  # ABAR
    ], side="right")

x.write("landing_xdsm")
x.write_sys_specs("landing_specs")
