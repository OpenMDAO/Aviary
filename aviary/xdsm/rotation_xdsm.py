from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

simplified = False

# Create subsystem components
x.add_system("atmos", FUNC, ["USatm"])
x.add_system("fc", FUNC, ["FlightConditions"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("aero", GROUP, [r"\textbf{Aerodynamics}"])
x.add_system("eom", GROUP, [r"\textbf{EOM}"])

# create inputs
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
x.add_input("fc", ["TAS"])

if simplified:
    x.add_input("prop", ["InputValues"])
else:
    x.add_input("prop", [
        Dynamic.Mission.ALTITUDE,
        Dynamic.Mission.MACH,
        Dynamic.Mission.THROTTLE
    ])

if simplified:
    x.add_input("aero", ["InputValues"])
else:
    x.add_input("aero", [
        "airport_alt",
        Dynamic.Mission.ALTITUDE,
        Aircraft.Wing.HEIGHT,
        Aircraft.Wing.SPAN,
        "alpha",
        "flap_defl",
        Mission.Design.GROSS_MASS,
        Aircraft.Wing.AREA,
        "t_init_gear",
        "t_curr",
        "dt_gear",
        "t_init_flaps",
        "dt_flaps",
    ])

if simplified:
    x.add_input("eom", ["InputValues"])
else:
    x.add_input("eom", [
        Dynamic.Mission.MASS,
        "TAS",
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        Aircraft.Wing.INCIDENCE,
        "alpha",
    ])

# make connections
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
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
], side="right")

x.write("rotation_xdsm")
x.write_sys_specs("rotation_specs")
