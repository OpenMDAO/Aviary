from pyxdsm.XDSM import FUNC, GROUP, IFUNC, IGROUP, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

# src = "subsystems/flops_based/computed_aero_group.py"
simplified = False
show_outputs = True

# Create subsystem components
x.add_system("mux", FUNC, ["MUX"])
x.add_system("dynPress", FUNC, ["DynamicPressure"])
x.add_system("lt_eq_wt", FUNC, ["LiftEqualsWeight"])
x.add_system("deptDrag", FUNC, ["LiftDependentDrag"])
x.add_system("induDrag", FUNC, ["InducedDrag"])
x.add_system("compDrag", FUNC, ["CompressibilityDrag"])
x.add_system("skinFric", FUNC, ["SkinFriction"])
x.add_system("skinDrag", FUNC, ["SkinFrictionDrag"])
x.add_system("cmpdDrag", GROUP, ["ComputedDrag"])
x.add_system("buffLift", FUNC, ["BuffetLift"])

if simplified:
    x.add_input("mux", ["InputValues"])
    x.add_input("dynPress", ["InputValues"])
    x.add_input("lt_eq_wt", ["InputValues"])
    x.add_input("deptDrag", ["InputValues"])
    x.add_input("induDrag", ["InputValues"])
    x.add_input("compDrag", ["InputValues"])
    x.add_input("skinFric", ["InputValues"])
    x.add_input("skinDrag", ["InputValues"])
    x.add_input("cmpdDrag", ["InputValues"])
    x.add_input("buffLift", ["InputValues"])
else:
    # Wing
    x.add_input("mux", [
        Aircraft.Wing.WETTED_AREA,
        Aircraft.Wing.FINENESS,
        Aircraft.Wing.CHARACTERISTIC_LENGTH,
        Aircraft.Wing.LAMINAR_FLOW_UPPER,
        Aircraft.Wing.LAMINAR_FLOW_LOWER,
        # Tail
        Aircraft.HorizontalTail.WETTED_AREA,
        Aircraft.HorizontalTail.FINENESS,
        Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
        Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER,
        Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER,
        # Vertical Tail
        Aircraft.VerticalTail.WETTED_AREA,
        Aircraft.VerticalTail.FINENESS,
        Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
        Aircraft.VerticalTail.LAMINAR_FLOW_UPPER,
        Aircraft.VerticalTail.LAMINAR_FLOW_LOWER,
        # Fuselage
        Aircraft.Fuselage.WETTED_AREA,
        Aircraft.Fuselage.FINENESS,
        Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
        Aircraft.Fuselage.LAMINAR_FLOW_UPPER,
        Aircraft.Fuselage.LAMINAR_FLOW_LOWER,
        # Engine
        Aircraft.Nacelle.WETTED_AREA,
        Aircraft.Nacelle.FINENESS,
        Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
        Aircraft.Nacelle.LAMINAR_FLOW_UPPER,
        Aircraft.Nacelle.LAMINAR_FLOW_LOWER,
    ])

    x.add_input("dynPress", [Dynamic.Mission.MACH, Dynamic.Mission.STATIC_PRESSURE])
    x.add_input("lt_eq_wt", [
        Aircraft.Wing.AREA,
        Dynamic.Mission.MASS,
        # Dynamic.Mission.DYNAMIC_PRESSURE,
    ])
    x.add_input("deptDrag", [
        Dynamic.Mission.MACH,
        # Dynamic.Mission.LIFT,
        Dynamic.Mission.STATIC_PRESSURE,
        Mission.Design.MACH,
        Mission.Design.LIFT_COEFFICIENT,
        Aircraft.Wing.AREA,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.THICKNESS_TO_CHORD,
    ])
    x.add_input("induDrag", [
        Dynamic.Mission.MACH,
        # Dynamic.Mission.LIFT,
        Dynamic.Mission.STATIC_PRESSURE,
        Aircraft.Wing.AREA,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.SPAN_EFFICIENCY_FACTOR,
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.TAPER_RATIO,
    ])
    x.add_input("compDrag", [
        Dynamic.Mission.MACH,
        Mission.Design.MACH,
        Aircraft.Design.BASE_AREA,
        Aircraft.Wing.AREA,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.THICKNESS_TO_CHORD,
        Aircraft.Fuselage.CROSS_SECTION,
        Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
        Aircraft.Fuselage.LENGTH_TO_DIAMETER,
    ])
    x.add_input("skinFric", [
        Dynamic.Mission.MACH,
        Dynamic.Mission.STATIC_PRESSURE,
        Dynamic.Mission.TEMPERATURE,
        # "characteristic_lengths",
    ])
    x.add_input("skinDrag", [
        # "skin_friction_coeff",
        # "Re",
        Aircraft.Wing.AREA,
    ])
    x.add_input("cmpdDrag", [
        # Dynamic.Mission.DYNAMIC_PRESSURE,
        Dynamic.Mission.MACH,
        Aircraft.Wing.AREA,
        Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
        Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
        Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
        Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
    ])
    x.add_input("buffLift", [
        Dynamic.Mission.MACH,
        Mission.Design.MACH,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.THICKNESS_TO_CHORD,
    ])

# make connections
x.connect("mux", "skinFric", ["characteristic_lengths"])

x.connect("mux", "skinDrag", [
    "wetted_areas",
    "fineness_ratios",
    "laminar_fractions_upper",
    "laminar_fractions_lower",
])

x.connect("dynPress", "lt_eq_wt", [Dynamic.Mission.DYNAMIC_PRESSURE])
x.connect("dynPress", "cmpdDrag", [Dynamic.Mission.DYNAMIC_PRESSURE])
x.connect("lt_eq_wt", "deptDrag", [Dynamic.Mission.LIFT])
x.connect("lt_eq_wt", "induDrag", [Dynamic.Mission.LIFT])

x.connect("skinFric", "skinDrag", ["skin_friction_coeff", "Re"])
x.connect("deptDrag", "cmpdDrag", ["CD/pressure_drag_coeff"])
x.connect("induDrag", "cmpdDrag", ["induced_drag_coeff"])
x.connect("compDrag", "cmpdDrag", ["compress_drag_coeff"])
x.connect("skinDrag", "cmpdDrag", ["skin_friction_drag_coeff"])

# create outputs
if show_outputs:
    x.add_output("lt_eq_wt", ["CL", Dynamic.Mission.LIFT], side="right")
    # x.add_output("skinFric", ["cf_iter",  "wall_temp"], side="right")  # don't see why needed
    x.add_output("cmpdDrag", [
        "CDI",
        "CD0",
        "CD/drag_coefficient",
        Dynamic.Mission.DRAG,
    ], side="right")
    x.add_output("buffLift", ["delta_CL_Before"], side="right")  # DELCLB

x.write("computed_aero_xdsm")
x.write_sys_specs("computed_aero_specs")
