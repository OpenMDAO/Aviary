from pyxdsm.XDSM import FUNC, GROUP, IGROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic

x = XDSM()

simplified = False  # for testflo run, it must be False

# Create subsystem components
x.add_system("basic", GROUP, [r"\textbf{BasicFlapsCalculations}"])
x.add_system("CL_max", FUNC, [r"\textbf{CLmaxCalculation}"])
x.add_system("tables", IGROUP, [r"\textbf{LookupTables}"])
x.add_system("increments", FUNC, [r"\textbf{LiftAndDragIncrements}"])

# create inputs
if simplified is True:
    x.add_input("basic", ["InputValues"])
    x.add_input("CL_max", ["InputValues"])
    x.add_input("tables", ["InputValues"])
    x.add_input("increments", ["InputValues"])
else:
    x.add_input("basic", [
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.FLAP_CHORD_RATIO,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.CENTER_CHORD,
        Aircraft.Fuselage.AVG_DIAMETER,
        Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
        Aircraft.Wing.SPAN,
        Aircraft.Wing.SLAT_CHORD_RATIO,
        "slat_defl",
        Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION,
        "flap_defl",
        Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION,
        Aircraft.Wing.ROOT_CHORD,
        Aircraft.Fuselage.LENGTH,
        Aircraft.Wing.LEADING_EDGE_SWEEP,
    ])
    x.add_input("CL_max", [
        Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
        Aircraft.Wing.LOADING,
        Dynamic.Mission.STATIC_PRESSURE,
        Aircraft.Wing.AVERAGE_CHORD,
        Aircraft.Wing.MAX_LIFT_REF,
        Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM,
        "fus_lift",  # fuselage lift increment
        "kinematic_viscosity",
        Dynamic.Mission.TEMPERATURE,
        Dynamic.Mission.SPEED_OF_SOUND,
    ])
    x.add_input("tables", [
        Aircraft.Wing.FLAP_CHORD_RATIO,
        Aircraft.Wing.FLAP_SPAN_RATIO,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
        "flap_defl",
    ])
    x.add_input("increments", [
        Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
        Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
    ])

# make connections
x.connect("basic", "increments", ["VLAM8", "VDEL4", "VDEL5"])
x.connect("basic", "CL_max", ["VLAM8", "VLAM9", "VLAM12"])
x.connect("basic", "tables", [
    "slat_defl_ratio",
    "flap_defl_ratio",
    "body_to_span_ratio",
    Aircraft.Wing.SLAT_SPAN_RATIO,
    "chord_to_body_ratio",
])
x.connect("CL_max", "tables", ["reynolds", Dynamic.Mission.MACH])
x.connect("tables", "increments", [
    "VDEL1",
    "VDEL2",
    "VDEL3",
    "VLAM3",
    "VLAM4",
    "VLAM5",
    "VLAM6",
    "VLAM7",
    "VLAM13",
    "VLAM14",
])
x.connect("tables", "CL_max", [
    "VLAM1",
    "VLAM2",
    "VLAM3",
    "VLAM4",
    "VLAM5",
    "VLAM6",
    "VLAM7",
    "VLAM10",
    "VLAM11",
    "VLAM13",
    "VLAM14",
    "fus_lift",
])

# create outputs
x.add_output("increments", ["delta_CL", "delta_CD"], side="right")
x.add_output("CL_max", ["CL_max"], side="right")

x.write("flaps_xdsm")
x.write_sys_specs("flaps_specs")
