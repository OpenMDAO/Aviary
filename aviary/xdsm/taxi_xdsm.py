from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

# Create subsystem components
x.add_system("atmos", FUNC, ["USatm"])
x.add_system("prop", GROUP, ["prop"])
x.add_system("fuel", GROUP, [r"\textbf{taxifuel}"])

# create inputs
x.add_input("atmos", ["airport_alt"])
x.add_input("prop", [
    Dynamic.Mission.THROTTLE,
    Dynamic.Mission.ALTITUDE,
])
x.add_input("fuel", [Mission.Summary.GROSS_MASS])

# make connections
x.connect("atmos", "prop", [
    Dynamic.Mission.TEMPERATURE,
    Dynamic.Mission.STATIC_PRESSURE
])
x.connect("prop", "fuel", [Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL])

# create output
x.add_output("fuel", ["mass"], side="right")

x.write("taxi_xdsm")
x.write_sys_specs("taxi_specs")
