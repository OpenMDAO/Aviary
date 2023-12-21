"""
This XDSM is for the case with both a fold and a strut, volume coefficients input, no
augmented system, no propeller, and a low wing
"""

from pyxdsm.XDSM import FUNC, GROUP, IGROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

simplified = False
HAS_FOLD = True
HAS_STRUT = True
# HAS_PROPELLERS = False
# has_hybrid_system = False  # see fixed_mass_xdsm.py

# Create subsystem components
x.add_system("size", GROUP, [r"\textbf{SizeGroup}"])
x.add_system("design_load", GROUP, ["DesignLoadGroup"])
x.add_system("fixed_mass", GROUP, [r"\textbf{FixedMassGroup}"])
x.add_system("equip", FUNC, ["EquipAndUsefulMass"])
x.add_system("wing_mass", IGROUP, [r"\textbf{WingMassGroup}"])
x.add_system("fuel_mass", IGROUP, ["FuelMassGroup"])

### make input connections ###

size_inputs = [
    # connect inputs:
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Mission.Design.GROSS_MASS,
    Aircraft.Wing.LOADING,
    Aircraft.VerticalTail.ASPECT_RATIO,
    Aircraft.HorizontalTail.TAPER_RATIO,
    Aircraft.Fuel.WING_FUEL_FRACTION,
    Aircraft.VerticalTail.TAPER_RATIO,
    # direct inputs:
    Aircraft.Fuselage.DELTA_DIAMETER,
    Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
    Aircraft.Fuselage.NOSE_FINENESS,
    Aircraft.Fuselage.TAIL_FINENESS,
    Aircraft.Fuselage.WETTED_AREA_FACTOR,
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
    Aircraft.HorizontalTail.MOMENT_RATIO,
    Aircraft.VerticalTail.MOMENT_RATIO,
    Aircraft.HorizontalTail.ASPECT_RATIO,
    Aircraft.Engine.REFERENCE_DIAMETER,
    Aircraft.Nacelle.CORE_DIAMETER_RATIO,
    Aircraft.Nacelle.FINENESS,
    Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
    Aircraft.VerticalTail.VOLUME_COEFFICIENT,
    # Aircraft.Engine.WING_LOCATIONS,
]
if HAS_FOLD:
    size_inputs.append(
        r"\textcolor{gray}{"+Aircraft.Engine.SCALE_FACTOR+"}")  # connect inputs
    size_inputs.append(
        r"\textcolor{gray}{"+Aircraft.Wing.FOLDED_SPAN+"}")  # connect inputs
if HAS_STRUT:
    size_inputs.append(
        r"\textcolor{gray}{"+Aircraft.Strut.ATTACHMENT_LOCATION+"}")  # connect inputs
    size_inputs.append(
        r"\textcolor{gray}{"+Aircraft.Strut.AREA_RATIO+"}")  # direct inputs
if simplified:
    x.add_input("size", ["InputValues"])
else:
    x.add_input("size", size_inputs)
if simplified:
    x.add_input("design_load", ["InputValues"])
else:
    x.add_input("design_load", [
        # connect input:
        Aircraft.Wing.LOADING,
        # direct input:
        Aircraft.Design.MAX_STRUCTURAL_SPEED,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.SWEEP,
        Mission.Design.MACH
    ])
fixed_input = [
    # HL connect inputs:
    "density",
    Aircraft.Wing.LOADING,
    Aircraft.Fuselage.AVG_DIAMETER,
    Aircraft.Wing.FLAP_CHORD_RATIO,
    Aircraft.Wing.SLAT_CHORD_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Aircraft.Wing.CENTER_CHORD,
    Aircraft.Wing.FLAP_SPAN_RATIO,
    Aircraft.Wing.SLAT_SPAN_RATIO,
    Mission.Landing.LIFT_COEFFICIENT_MAX,
    # connect inputs:
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.SWEEP,
    Mission.Design.GROSS_MASS,
    Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
    Aircraft.VerticalTail.ASPECT_RATIO,
    Aircraft.HorizontalTail.TAPER_RATIO,
    Aircraft.Engine.SCALED_SLS_THRUST,
    Aircraft.Engine.WING_LOCATIONS,
    Aircraft.VerticalTail.TAPER_RATIO,
    # direct inputs:
    Aircraft.CrewPayload.CARGO_MASS,
    Aircraft.VerticalTail.SWEEP,
    Aircraft.HorizontalTail.MASS_COEFFICIENT,
    Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
    Aircraft.VerticalTail.MASS_COEFFICIENT,
    Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
    Aircraft.VerticalTail.THICKNESS_TO_CHORD,
    Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
    Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
    Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
    Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER,
    Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
    Aircraft.Controls.CONTROL_MASS_INCREMENT,
    Aircraft.LandingGear.MASS_COEFFICIENT,
    Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT,
    Aircraft.Engine.MASS_SPECIFIC,
    Aircraft.Nacelle.MASS_SPECIFIC,
    Aircraft.Engine.PYLON_FACTOR,
    Aircraft.Engine.ADDITIONAL_MASS_FRACTION,
    Aircraft.Engine.MASS_SCALER,
    Aircraft.Propulsion.MISC_MASS_SCALER,
    Aircraft.LandingGear.MAIN_GEAR_LOCATION,
    # Aircraft.Engine.SCALED_SLS_THRUST,
    Aircraft.Wing.MOUNTING_TYPE,
    Aircraft.Nacelle.CLEARANCE_RATIO,
]
if HAS_STRUT:
    fixed_input.append(
        r"\textcolor{gray}{"+Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS+"}")
if simplified:
    x.add_input("fixed_mass", ["InputValues"])
else:
    x.add_input("fixed_mass", fixed_input)
if simplified:
    x.add_input("equip", ["InputValues"])
else:
    x.add_input("equip", [
        Mission.Design.GROSS_MASS,
        Aircraft.Engine.SCALED_SLS_THRUST,
        Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
        Aircraft.Fuel.WING_FUEL_FRACTION,
        Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
        # direct input:
        Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
    ])
wing_inputs = [
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Mission.Design.GROSS_MASS,
    # direct input:
    Aircraft.Wing.MASS_COEFFICIENT,
]
if HAS_FOLD:
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Wing.FOLD_MASS_COEFFICIENT+"}")
if HAS_STRUT:
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Strut.MASS_COEFFICIENT+"}")
if simplified:
    x.add_input("wing_mass", ["InputValues"])
else:
    x.add_input("wing_mass", wing_inputs)
if simplified:
    x.add_input("fuel_mass", ["InputValues"])
else:
    x.add_input("fuel_mass", [
        Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
        Mission.Design.GROSS_MASS,
        # direct input
        Aircraft.Fuel.FUEL_MARGIN,
        Aircraft.Fuselage.MASS_COEFFICIENT,
        "pylon_len",
        "MAT",
        Aircraft.Wing.MASS_SCALER,
        Aircraft.HorizontalTail.MASS_SCALER,
        Aircraft.VerticalTail.MASS_SCALER,
        Aircraft.Fuselage.MASS_SCALER,
        Aircraft.LandingGear.TOTAL_MASS_SCALER,
        Aircraft.Engine.POD_MASS_SCALER,
        Aircraft.Design.STRUCTURAL_MASS_INCREMENT,
        Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
        Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
        Aircraft.Fuel.DENSITY,
        # Mission.Design.FUEL_MASS_REQUIRED,
        # Aircraft.Fuel.TOTAL_CAPACITY,
    ])

# make connections
x.connect("size", "design_load", [Aircraft.Wing.AVERAGE_CHORD])

wing_inputs = [Aircraft.Wing.SPAN]
if HAS_FOLD:
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Wing.AREA+"}")
    wing_inputs.append(r"\textcolor{gray}{"+Aircraft.Wing.FOLDING_AREA+"}")
if simplified:
    x.connect("size", "wing_mass", ["aircraft:*"])
else:
    x.connect("size", "wing_mass", wing_inputs)

if simplified:
    x.connect("size", "fixed_mass", ["aircraft:*"])
else:
    x.connect("size", "fixed_mass", [
        Aircraft.Fuselage.LENGTH,
        Aircraft.Wing.SPAN,
        Aircraft.Wing.AREA,
        Aircraft.VerticalTail.AREA,
        Aircraft.HorizontalTail.AREA,
        Aircraft.VerticalTail.SPAN,
        Aircraft.HorizontalTail.SPAN,
        Aircraft.HorizontalTail.MOMENT_ARM,
        Aircraft.HorizontalTail.ROOT_CHORD,
        Aircraft.VerticalTail.MOMENT_ARM,
        Aircraft.VerticalTail.ROOT_CHORD,
        Aircraft.Nacelle.SURFACE_AREA,
        Aircraft.Nacelle.AVG_DIAMETER,
    ])
x.connect("size", "equip", [
    Aircraft.Fuselage.LENGTH,
    Aircraft.Wing.SPAN,
    Aircraft.HorizontalTail.AREA,
    Aircraft.VerticalTail.AREA,
    Aircraft.Fuselage.AVG_DIAMETER,
    Aircraft.Wing.AREA,
])
if simplified:
    x.connect("size", "fuel_mass", ["aircraft:*"])
else:
    x.connect("size", "fuel_mass", [
        Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
        Aircraft.Fuselage.WETTED_AREA,
        Aircraft.Fuselage.AVG_DIAMETER,
        Aircraft.TailBoom.LENGTH,
    ])

x.connect("design_load", "fixed_mass", [
    "max_mach",
    "min_dive_vel",
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
])
x.connect("design_load", "wing_mass", [
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR
])
x.connect("design_load", "fuel_mass", [
    "min_dive_vel",
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
])

if simplified:
    x.connect("fixed_mass", "wing_mass", [
        "aircraft:*",
        "c_strut_braced",
        "c_gear_loc",
        "half_sweep",
    ])
else:
    x.connect("fixed_mass", "wing_mass", [
        Aircraft.Wing.MATERIAL_FACTOR,
        "c_strut_braced",
        "c_gear_loc",
        Aircraft.Engine.POSITION_FACTOR,
        "half_sweep",
        Aircraft.Wing.HIGH_LIFT_MASS,
    ])
if simplified:
    x.connect("fixed_mass", "fuel_mass", [
        "aircraft:*",
        "payload_mass_des",
        "payload_mass_max",
        "wing_mounted_mass",
        "eng_comb_mass",
    ])
else:
    x.connect("fixed_mass", "fuel_mass", [
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
        "payload_mass_des",
        "payload_mass_max",
        Aircraft.HorizontalTail.MASS,
        Aircraft.VerticalTail.MASS,
        Aircraft.Controls.TOTAL_MASS,
        Aircraft.LandingGear.TOTAL_MASS,
        Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,
        "wing_mounted_mass",
        "eng_comb_mass",
    ])
x.connect("fixed_mass", "equip", [
    Aircraft.Controls.TOTAL_MASS,
    Aircraft.LandingGear.TOTAL_MASS,
])
x.connect("equip", "fuel_mass", [
    Aircraft.Design.FIXED_EQUIPMENT_MASS,
    Aircraft.Design.FIXED_USEFUL_LOAD,
])
x.connect("wing_mass", "fuel_mass", [Aircraft.Wing.MASS])

x.write("mass_and_sizing_both_xdsm")
x.write_sys_specs("mass_and_sizing_both_specs")
