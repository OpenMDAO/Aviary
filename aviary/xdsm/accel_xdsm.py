from pyxdsm.XDSM import FUNC, GROUP, IFUNC, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

simplified = True

# Create subsystem components
x.add_system("atmos", FUNC, [r"AtmosphereModel"])
x.add_system("fc", FUNC, [r"\textbf{FlightConditions}"])
x.add_system("weight", FUNC, ["MassToWeight"])
x.add_system("aero", GROUP, ["CruiseAero", "(alpha~in)"])
x.add_system("prop", GROUP, [r"Propulsion"])
x.add_system("eom", FUNC, [r"\textbf{AccelEOM}"])

# create inputs
if simplified:
    x.add_input("aero", ["aircraft:*"])
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

# create outputs
x.add_output("fc", ["EAS"], side="right")
x.add_output("aero", [Dynamic.Mission.LIFT, Dynamic.Mission.DRAG, "alpha"], side="right")
# for accel_ode, CruiseAero(output_alpha=True). So alpha is not an input.
x.add_output("eom", ["TAS_rate", Dynamic.Mission.DISTANCE_RATE], side="right")

# make connections
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])
x.add_input("fc", ["TAS"])
x.add_input("prop", [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.THROTTLE,
    Aircraft.Engine.SCALE_FACTOR,
])
x.add_input("weight", [Dynamic.Mission.MASS])
x.add_input("eom", [Dynamic.Mission.MASS, "TAS"])
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
# x.connect("atmos", "prop", [Dynamic.Mission.TEMPERATURE, Dynamic.Mission.STATIC_PRESSURE])
x.connect("fc", "aero", [Dynamic.Mission.MACH, Dynamic.Mission.DYNAMIC_PRESSURE])
x.connect("fc", "prop", [Dynamic.Mission.MACH])
x.connect("weight", "aero", ["weight/lift_req"])
x.connect("aero", "eom", [Dynamic.Mission.DRAG])
# x.connect("aero", "balance", [Dynamic.Mission.LIFT])
x.connect("prop", "eom", [Dynamic.Mission.THRUST_TOTAL])

# Do not run spec test on aero.
x.write("accel_xdsm")
x.write_sys_specs("accel_specs")
