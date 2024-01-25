from pyxdsm.XDSM import FUNC, GROUP, IFUNC, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

simplified = True

# Create subsystem components
x.add_system("atmos", FUNC, ["AtmosphereModel"])
x.add_system("fc", FUNC, [r"\textbf{FlightConditions}"])
x.add_system("weight", FUNC, ["MassToWeight"])
x.add_system("aero", GROUP, ["CruiseAero", "(alpha~in)"])
x.add_system("prop", GROUP, ["Propulsion"])
x.add_system("thrust_bal", IFUNC, ["BalanceThrust"])
x.add_system("breguet_eom", FUNC, [r"\textbf{BreguetEOM}"])

# create inputs

if simplified:
    x.add_input("aero", [Dynamic.Mission.MACH, "aircraft:*"])
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
x.add_input("atmos", [Dynamic.Mission.ALTITUDE])  # alt_cruise
x.add_input("fc", [Dynamic.Mission.MACH])
x.add_input("prop", [
    Dynamic.Mission.ALTITUDE,
    Dynamic.Mission.MACH,
    Dynamic.Mission.THROTTLE,
    Aircraft.Engine.SCALE_FACTOR,
])
x.add_input("breguet_eom", [
    Dynamic.Mission.MASS,
    "cruise_distance_initial",
    "cruise_time_initial",
])
x.add_input("weight", ["mass"])

# make connections
x.connect("fc", "aero", [Dynamic.Mission.DYNAMIC_PRESSURE])
x.connect("atmos", "fc", ["rho", Dynamic.Mission.SPEED_OF_SOUND])
x.connect("weight", "aero", ["weight/lift_req"])
x.connect("aero", "prop", ["drag/thrust_req"])
x.connect("aero", "thrust_bal", ["thrust_req"])
x.connect("fc", "breguet_eom", ["TAS"])  # TAS_cruise
x.connect("prop", "breguet_eom", [Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL])
x.connect("prop", "thrust_bal", [Dynamic.Mission.THRUST_TOTAL])
x.connect("thrust_bal", "prop", ["thrust_req"])

# create outputs
x.add_output("breguet_eom", [Dynamic.Mission.DISTANCE, "cruise_time"], side="right")
# x.add_output("fc", ["EAS"], side="right")

x.write("cruise_xdsm")
x.write_sys_specs("cruise_specs")
