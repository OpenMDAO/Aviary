from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft

x = XDSM()

compute_volume_coeff = True

if compute_volume_coeff:
    x.add_system("hvc", FUNC, [r"\textcolor{gray}{htail_vc\, (TailVolCoef)}"])
    x.add_system("vvc", FUNC, [r"\textcolor{gray}{vtail_vc\, (TailVolCoef)}"])
x.add_system("htail", FUNC, [r"htail\, (TailSize)"])
x.add_system("vtail", FUNC, [r"vtail\, (TailSize)"])

# move to EmpennageSize inputs
# x.add_input("htail", [r"chord_hor_arm", r"h_ar", r"h_tr", r"(htail_vol_coef)"])
# x.add_input("vtail", [r"span_ver_arm", r"v_ar", r"v_tr", r"(vtail_vol_coef)"])

if compute_volume_coeff:
    x.add_input("hvc", [
        r"\textcolor{gray}{"+Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION+"}",
        r"\textcolor{gray}{"+Aircraft.Fuselage.LENGTH+"}",
        r"\textcolor{gray}{"+Aircraft.Fuselage.AVG_DIAMETER+"}",
        r"\textcolor{gray}{"+Aircraft.Wing.AREA+"}",
        r"\textcolor{gray}{"+Aircraft.Wing.AVERAGE_CHORD+"}",
    ])
    x.add_input("vvc", [
        r"\textcolor{gray}{"+Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION+"}",
        r"\textcolor{gray}{"+Aircraft.Fuselage.LENGTH+"}",
        r"\textcolor{gray}{"+Aircraft.Fuselage.AVG_DIAMETER+"}",
        r"\textcolor{gray}{"+Aircraft.Wing.AREA+"}",
        r"\textcolor{gray}{"+Aircraft.Wing.SPAN+"}",
    ])
in_htail_array = [
    Aircraft.HorizontalTail.MOMENT_RATIO,
    Aircraft.HorizontalTail.ASPECT_RATIO,
    Aircraft.HorizontalTail.TAPER_RATIO,
    Aircraft.Wing.AREA,
    Aircraft.Wing.AVERAGE_CHORD,
]
if not compute_volume_coeff:
    in_htail_array.append(
        r"\textcolor{gray}{"+Aircraft.HorizontalTail.VOLUME_COEFFICIENT+"}")
x.add_input("htail", in_htail_array)
in_vtail_array = [
    Aircraft.VerticalTail.MOMENT_RATIO,
    Aircraft.VerticalTail.ASPECT_RATIO,
    Aircraft.VerticalTail.TAPER_RATIO,
    Aircraft.Wing.AREA,
    Aircraft.Wing.SPAN,
]
if not compute_volume_coeff:
    in_vtail_array.append(
        r"\textcolor{gray}{"+Aircraft.VerticalTail.VOLUME_COEFFICIENT+"}")
x.add_input("vtail", in_vtail_array)

# make connections
if compute_volume_coeff:
    x.connect("hvc", "htail", [
        r"\textcolor{gray}{"+Aircraft.HorizontalTail.VOLUME_COEFFICIENT+"}"])
    x.connect("vvc", "vtail", [
        r"\textcolor{gray}{"+Aircraft.VerticalTail.VOLUME_COEFFICIENT+"}"])

# create outputs
x.add_output("htail", [
    Aircraft.HorizontalTail.AREA,
    Aircraft.HorizontalTail.SPAN,
    Aircraft.HorizontalTail.ROOT_CHORD,
    Aircraft.HorizontalTail.AVERAGE_CHORD,
    Aircraft.HorizontalTail.MOMENT_ARM,
], side="right")
x.add_output("vtail", [
    Aircraft.VerticalTail.AREA,
    Aircraft.VerticalTail.SPAN,
    Aircraft.VerticalTail.ROOT_CHORD,
    Aircraft.VerticalTail.AVERAGE_CHORD,
    Aircraft.VerticalTail.MOMENT_ARM
], side="right")

x.write("empennage_size_xdsm")
# x.write_sys_specs("empennage_size_specs")
