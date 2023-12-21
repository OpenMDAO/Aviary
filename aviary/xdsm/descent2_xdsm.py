"""XDSM for below 10 kft"""

from pyxdsm.XDSM import FUNC, GROUP, IFUNC, IGROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

# Create subsystem components
x.add_system("atmos", FUNC, ["AtmosphereModel"])
x.add_system("fc", FUNC, ["FlightConditions"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("eom", GROUP, ["EOM"])
x.add_system("balance_lift", FUNC, ["BalanceLift"])
x.add_system("aero", IGROUP, ["CruiseAero", "(alpha~in)"])
x.add_system("pitch", FUNC, ["Constraints"])

# create inputs
# x.add_input("fc", ["EAS"])
# x.add_input("aero", [Aircraft.Wing.AREA])
x.add_input("aero", [r"aircraft:*", Dynamic.Mission.ALTITUDE])
x.add_input("prop", [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.THROTTLE,
    Aircraft.Engine.SCALE_FACTOR,
])
x.add_input("eom", ["mass"])
x.add_input("pitch", [
    Dynamic.Mission.MASS,
    Aircraft.Wing.AREA,
    Aircraft.Wing.INCIDENCE,
])

# make connections
x.add_input("fc", ["EAS"])
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
x.connect("atmos", "aero", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
x.connect("atmos", "pitch", ["rho"])
x.connect("fc", "aero", [Dynamic.Mission.MACH, Dynamic.Mission.DYNAMIC_PRESSURE])
x.connect("fc", "prop", [Dynamic.Mission.MACH])
x.connect("fc", "eom", ["TAS"])
x.connect("fc", "pitch", ["TAS"])
x.connect("prop", "eom", [Dynamic.Mission.THRUST_TOTAL])
x.connect("prop", "aero", [Dynamic.Mission.THRUST_TOTAL])
x.connect("eom", "pitch", [Dynamic.Mission.FLIGHT_PATH_ANGLE])
x.connect("eom", "balance_lift", ["required_lift"])
x.connect("balance_lift", "aero", ["alpha"])
x.connect("balance_lift", "eom", ["alpha"])
x.connect("balance_lift", "pitch", ["alpha"])
x.connect("aero", "eom", [Dynamic.Mission.DRAG])
x.connect("aero", "pitch", ["CL_max"])
x.connect("aero", "balance_lift", [Dynamic.Mission.LIFT])

# create outputs
x.add_output("eom", [
    Dynamic.Mission.ALTITUDE_RATE,
    "distance_rate",
    "required_lift"
], side="right")

x.write("descent2_xdsm")
x.write_sys_specs("descent2_specs")
