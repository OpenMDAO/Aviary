"""XDSM for above 10 kft"""

from pyxdsm.XDSM import FUNC, GROUP, IFUNC, IGROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

# input_speed_type = "MACH"
# analysis_scheme = "COLLOCATION"

# Create subsystem components
x.add_system("atmos", FUNC, ["AtmosphereModel"])
x.add_system("balance_speed", IFUNC, ["BalanceSpeed"])
x.add_system("speeds", FUNC, [r"\textbf{SpeedConstraints}"])
x.add_system("fc", FUNC, ["FlightConditions"])
x.add_system("ks", FUNC, ["KSComp"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("eom", GROUP, [r"\textbf{EOM}"])
x.add_system("balance_lift", FUNC, ["BalanceLift"])
x.add_system("aero", IGROUP, ["CruiseAero", "(alpha~in)"])
x.add_system("pitch", FUNC, ["Constraints"])
x.add_system('specific_energy', FUNC, ["specific_energy"])
x.add_system('alt_rate', FUNC, ["alt_rate"])

# create inputs
# x.add_input("fc", [Dynamic.Mission.MACH])
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
x.add_input("balance_speed", ["rhs_mass"])
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
x.add_input("specific_energy", [Dynamic.Mission.MASS])
x.add_input("alt_rate", ["TAS_rate"])

# make connections
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
x.connect("atmos", "aero", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
x.connect("atmos", "pitch", ["rho"])
x.connect("balance_speed", "fc", [Dynamic.Mission.MACH])
x.connect("balance_speed", "speeds", [Dynamic.Mission.MACH])
x.connect("balance_speed", "aero", [Dynamic.Mission.MACH])
x.connect("balance_speed", "prop", [Dynamic.Mission.MACH])
x.connect("speeds", "ks", ["speed_constraint"])
x.connect("fc", "aero", [Dynamic.Mission.DYNAMIC_PRESSURE])
x.connect("fc", "eom", ["TAS"])
x.connect("fc", "speeds", ["EAS"])
x.connect("fc", "pitch", ["TAS"])
x.connect("ks", "balance_speed", ["KS"])
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
x.connect("fc", "specific_energy", ["TAS"])
x.connect("aero", "specific_energy", [Dynamic.Mission.DRAG])
x.connect("prop", "specific_energy", [Dynamic.Mission.THRUST_MAX_TOTAL])
x.connect("specific_energy", "alt_rate", [Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS])
x.connect("fc", "alt_rate", ["TAS"])

# create outputs
x.add_output("eom", [
    Dynamic.Mission.ALTITUDE_RATE,
    Dynamic.Mission.DISTANCE_RATE,
    "required_lift"
], side="right")

x.add_output("alt_rate", [Dynamic.Mission.ALTITUDE_RATE_MAX], side="right")

x.write("descent1_xdsm")
x.write_sys_specs("descent1_specs")
