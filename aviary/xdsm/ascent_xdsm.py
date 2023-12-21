from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

# Create subsystem components
x.add_system("atmos", GROUP, ["AtmosphereModel"])
x.add_system("fc", FUNC, ["FlightConditions"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("aero", GROUP, ["LowSpeedAero", "(alpha~in)"])
x.add_system("eom", GROUP, [r"\textbf{AscentEOM}"])

# create inputs
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
x.add_input("fc", ["TAS"])
x.add_input("prop", [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.THROTTLE,
    Aircraft.Engine.SCALE_FACTOR,])
x.add_input("aero", [
    "alpha",
    "t_curr",
    "dt_flaps",
    "dt_gear",
    "t_init_flaps",
    "t_init_gear",
    "aircraft:*",
    # Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF,
    # Aircraft.Wing.HEIGHT,
    # Aircraft.Wing.SPAN,
    # Aircraft.Wing.AREA,
    Dynamic.Mission.ALTITUDE,
    Mission.Takeoff.AIRPORT_ALTITUDE,
    Mission.Design.GROSS_MASS,
    # 'aero_ramps:flap_factor:initial_val',
    # 'aero_ramps:gear_factor:initial_val',
    # 'aero_ramps:flap_factor:final_val',
    # 'aero_ramps:gear_factor:final_val',
],)
x.add_input("eom", [
    "alpha",
    "TAS",
    Dynamic.Mission.MASS,
    Dynamic.Mission.FLIGHT_PATH_ANGLE,
    Aircraft.Wing.INCIDENCE,
],)

# make connections
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
# x.connect("atmos", "prop", [Dynamic.Mission.TEMPERATURE, Dynamic.Mission.STATIC_PRESSURE])
x.connect("fc", "prop", [Dynamic.Mission.MACH])
x.connect("fc", "aero", [Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MACH])
x.connect("prop", "eom", [Dynamic.Mission.THRUST_TOTAL])
x.connect("aero", "eom", [Dynamic.Mission.LIFT, Dynamic.Mission.DRAG])

# create outputs
x.add_output("eom", [
    "TAS_rate",
    Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
    Dynamic.Mission.ALTITUDE_RATE,
    Dynamic.Mission.DISTANCE_RATE,
    "alpha_rate",
    "normal_force",
    "fuselage_pitch",
    "load_factor",
], side="right",)

x.write("ascent_xdsm")
x.write_sys_specs("ascent_specs")
