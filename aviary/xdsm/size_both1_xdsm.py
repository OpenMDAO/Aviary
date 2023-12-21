"""
This XDSM is for the case with a fold and a strut, fold location based on strut
location. It also is the case where tail volume coefficients are NOT computed.
"""

from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

# Do not change the following flags. Otherwise, unittest will fail.
HAS_FOLD = True
HAS_STRUT = True
compute_volume_coeff = False  # see empennage_size_xdsm.py

x.add_system("fuselage", GROUP, ["FuselageGroup"])
x.add_input("fuselage", [
    Aircraft.Fuselage.DELTA_DIAMETER,
    Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
    Aircraft.Fuselage.NOSE_FINENESS,
    Aircraft.Fuselage.TAIL_FINENESS,
    Aircraft.Fuselage.WETTED_AREA_FACTOR,
])
x.add_output("fuselage", [
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.TailBoom.LENGTH,
    "cabin_height",
    "cabin_len",
    "nose_height",
], side="right")

x.add_system("wing", GROUP, [r"\textbf{WingGroup}"])
wing_inputs = [
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
    Mission.Design.GROSS_MASS,
    Aircraft.Wing.LOADING,
    Aircraft.Fuel.WING_FUEL_FRACTION,
]
if HAS_FOLD:
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Wing.FOLDED_SPAN+"}")
if HAS_STRUT:
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Strut.ATTACHMENT_LOCATION+"}")
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Strut.AREA_RATIO+"}")
x.add_input("wing", wing_inputs)
wing_outputs = [
    Aircraft.Wing.CENTER_CHORD,
    Aircraft.Wing.ROOT_CHORD,
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
    Aircraft.Wing.LEADING_EDGE_SWEEP,
    Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
]
if HAS_FOLD:
    wing_outputs.append("fold.nonfolded_taper_ratio")
    wing_outputs.append(Aircraft.Wing.FOLDING_AREA)
    wing_outputs.append("fold.nonfolded_wing_area")
    wing_outputs.append("fold.tc_ratio_mean_folded")
    wing_outputs.append("fold.nonfolded_AR")
if HAS_STRUT:
    wing_outputs.append(Aircraft.Strut.LENGTH)
    wing_outputs.append(Aircraft.Strut.CHORD)
x.add_output("wing", wing_outputs, side="right")

x.add_system("empennage", GROUP, [r"\textbf{EmpennageSize}"])

emp_inputs = [
    Aircraft.HorizontalTail.MOMENT_RATIO,
    Aircraft.VerticalTail.MOMENT_RATIO,
    Aircraft.VerticalTail.ASPECT_RATIO,
    Aircraft.HorizontalTail.ASPECT_RATIO,
    Aircraft.VerticalTail.TAPER_RATIO,
    Aircraft.HorizontalTail.TAPER_RATIO,
]
if compute_volume_coeff:
    emp_inputs.append(
        r"\textcolor{gray}{"+Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION+"}")
else:
    emp_inputs.append(
        r"\textcolor{gray}{"+Aircraft.HorizontalTail.VOLUME_COEFFICIENT+"}")
    emp_inputs.append(
        r"\textcolor{gray}{"+Aircraft.VerticalTail.VOLUME_COEFFICIENT+"}")
x.add_input("empennage", emp_inputs)
x.add_output("empennage", [
    Aircraft.VerticalTail.AREA,
    Aircraft.HorizontalTail.AREA,
    Aircraft.VerticalTail.SPAN,
    Aircraft.HorizontalTail.SPAN,
    Aircraft.VerticalTail.ROOT_CHORD,
    Aircraft.HorizontalTail.ROOT_CHORD,
    Aircraft.VerticalTail.AREA,
    Aircraft.VerticalTail.AVERAGE_CHORD,
    Aircraft.HorizontalTail.AVERAGE_CHORD,
    Aircraft.VerticalTail.MOMENT_ARM,
    Aircraft.HorizontalTail.MOMENT_ARM,
], side="right")
x.connect("wing", "empennage", [
    Aircraft.Wing.SPAN,
    Aircraft.Wing.AREA,
    Aircraft.Wing.AVERAGE_CHORD
])
if compute_volume_coeff:
    x.connect("fuselage", "empennage", [
        Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.AVG_DIAMETER])

x.add_system("engine", FUNC, ["EngineSize"])
x.add_input("engine", [
    Aircraft.Engine.REFERENCE_DIAMETER,
    Aircraft.Engine.SCALE_FACTOR,
    Aircraft.Nacelle.CORE_DIAMETER_RATIO,
    Aircraft.Nacelle.FINENESS,
])
x.add_output("engine", [
    Aircraft.Nacelle.AVG_DIAMETER,
    Aircraft.Nacelle.AVG_LENGTH,
    Aircraft.Nacelle.SURFACE_AREA,
], side="right")

x.add_system("cable_size", FUNC, ["CableSize"])
x.add_input("cable_size", [Aircraft.Engine.WING_LOCATIONS])
x.add_output("cable_size", [Aircraft.Electrical.HYBRID_CABLE_LENGTH], side="right")

x.connect("fuselage", "wing", [Aircraft.Fuselage.AVG_DIAMETER])
x.connect("fuselage", "cable_size", [Aircraft.Fuselage.AVG_DIAMETER])
x.connect("wing", "cable_size", [Aircraft.Wing.SPAN])

x.write("size_both1_xdsm")
x.write_sys_specs("size_both1_specs")
