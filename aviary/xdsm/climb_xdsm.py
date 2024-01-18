from pyxdsm.XDSM import FUNC, GROUP, IFUNC, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

simplified = True

# input_speed_type = "EAS" for both high altitude and low altitude
# analysis_scheme = COLLOCATION

# Create subsystem components
x.add_system("atmos", FUNC, ["AtmosphereModel"])
x.add_system("speeds", FUNC, [r"\textbf{SpeedConstraints}"])
x.add_system("ks", FUNC, ["KSComp"])
x.add_system("balance_speed", IFUNC, ["BalanceSpeed"])
x.add_system("fc", FUNC, [r"\textbf{FlightConditions}"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("aero", GROUP, ["CruiseAero", "(alpha~in)"])
x.add_system("eom", GROUP, [r"\textbf{ClimbEOM}"])
x.add_system("balance_lift", IFUNC, ["BalanceLift"])
x.add_system("constraints", FUNC, [r"\textbf{Constraints}"])

# create inputs
x.add_input("speeds", [Dynamic.Mission.MACH])
x.add_input("constraints", [
    Dynamic.Mission.MASS,
    Aircraft.Wing.AREA,
    Aircraft.Wing.INCIDENCE
])
x.add_input("balance_speed", ["EAS"])
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
if simplified:
    x.add_input("aero", ["alpha", "aircraft:*"])
else:
    x.add_input("aero", [
        # Aircraft.Wing.AREA,
        "aircraft:wing:*",
        "aircraft:horizontal_tail:*",
        "aircraft:vertical_tail:*",
        "aircraft:fuselage:*",
        "aircraft:design:*",
        "aircraft:nacelle:*",
        "aircraft:strut:*",
        Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
    ])
x.add_input("prop", [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.THROTTLE,
    Aircraft.Engine.SCALE_FACTOR,
])
x.add_input("eom", [Dynamic.Mission.MASS])

# make connections
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
# x.connect("atmos", "prop", [Dynamic.Mission.TEMPERATURE, Dynamic.Mission.STATIC_PRESSURE])
x.connect("atmos", "constraints", ["rho"])
x.connect("fc", "prop", [Dynamic.Mission.MACH])
x.connect("fc", "aero", [Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MACH])
x.connect("fc", "eom", ["TAS"])
x.connect("fc", "constraints", ["TAS"])
x.connect("fc", "speeds", [Dynamic.Mission.MACH])
x.connect("speeds", "ks", ["speed_constraint"])
x.connect("ks", "balance_speed", ["KS"])
x.connect("balance_speed", "fc", ["EAS"])
x.connect("balance_speed", "speeds", ["EAS"])
x.connect("aero", "eom", [Dynamic.Mission.DRAG])
x.connect("aero", "constraints", ["CL_max"])
x.connect("prop", "eom", [Dynamic.Mission.THRUST_TOTAL])
x.connect("eom", "constraints", [Dynamic.Mission.FLIGHT_PATH_ANGLE])
x.connect("aero", "balance_lift", [Dynamic.Mission.LIFT])
x.connect("eom", "balance_lift", ["required_lift"])
x.connect("balance_lift", "constraints", ["alpha"])
x.connect("balance_lift", "aero", ["alpha"])

# create outputs
x.add_output("eom", [
    Dynamic.Mission.ALTITUDE_RATE,
    Dynamic.Mission.DISTANCE_RATE,
    "required_lift",
], side="right")
x.add_output("constraints", ["theta", "TAS_violation"], side="right")

x.add_output("aero", [
    Dynamic.Mission.LIFT,
    Dynamic.Mission.DRAG,
    "CL_max",
], side="right")

x.write("climb_xdsm")
x.write_sys_specs("climb_specs")
