from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

show_outputs = True

# Create subsystem components
x.add_system("speeds", FUNC, ["LoadSpeeds"])
x.add_system("params", FUNC, ["LoadParameters"])
x.add_system("slope", FUNC, ["LiftCurveSlope", "(at Cruise)"])
x.add_system("factors", FUNC, ["LoadFactors"])

# create inputs
x.add_input("speeds", [
    Aircraft.Design.MAX_STRUCTURAL_SPEED,
    Aircraft.Wing.LOADING,
])
x.add_input("slope", [
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.SWEEP, Mission.Design.MACH
])
x.add_input("factors", [Aircraft.Wing.AVERAGE_CHORD])

# make connections
x.connect("speeds", "params", ["max_airspeed", "vel_c"])
x.connect("speeds", "factors", ["max_maneuver_factor", "min_dive_vel"])
x.connect("slope", "factors", [Aircraft.Design.LIFT_CURVE_SLOPE])
x.connect("params", "factors", ["density_ratio", "V9"])

### add outputs ###
if show_outputs is True:
    # params
    x.add_output("params", ["max_mach"], side="right")
    # factors
    x.add_output("factors", [Aircraft.Wing.ULTIMATE_LOAD_FACTOR], side="right")


x.write("design_load_xdsm")
x.write_sys_specs("design_load_specs")
