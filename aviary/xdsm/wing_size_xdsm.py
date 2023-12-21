from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

# DIMENSIONAL_LOCATION_SPECIFIED = False
# This file assume HAS_FOLD = False and HAS_STRUT = False
# Otherwise, see wing_size_with_strut_fold_xdsm.py

# Create subsystem components
x.add_system("wing_size", FUNC, ["WingSize"])
x.add_system("wing_parameters", FUNC, ["WingParameters"])

# x.add_input("wing_size", [r"gross_wt_initial", r"wing_loading"])
x.add_input("wing_size", [
    Mission.Design.GROSS_MASS,
    Aircraft.Wing.LOADING,
    Aircraft.Wing.ASPECT_RATIO,
])

# x.add_input("wing_parameters", [r"sweep_c4", r"(fuel_vol_frac)"])
x.add_input("wing_parameters", [
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
    Aircraft.Fuselage.AVG_DIAMETER,
])

x.connect("wing_size", "wing_parameters", [Aircraft.Wing.AREA, Aircraft.Wing.SPAN])

x.add_output("wing_parameters", [
    Aircraft.Wing.CENTER_CHORD,
    Aircraft.Wing.AVERAGE_CHORD,
    Aircraft.Wing.ROOT_CHORD,
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
    Aircraft.Wing.LEADING_EDGE_SWEEP,
], side="right")

x.write("wing_size_xdsm")
# x.write_sys_specs("wing_size_specs")
