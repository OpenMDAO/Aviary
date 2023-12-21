"""
This XDSM is for the case with no strut, no fold, volume coefficients input, no
augmented system
"""

from pyxdsm.XDSM import FUNC, GROUP, IGROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

simplified = True

# Create subsystem components
x.add_system("design_load", GROUP, ["DesignLoadGroup"])
x.add_system("fixed_mass", GROUP, ["FixedMassGroup"])
x.add_system("equip", FUNC, ["FixedEquipAndUsefulMass"])
x.add_system("wing_mass", GROUP, ["WingMassGroup"])
x.add_system("fuel_mass", GROUP, ["FuelMassGroup"])

### make input connections ###

# design_load
if simplified:
    x.add_input("design_load", ["InputValues"])
else:
    x.add_input("design_load", [
        Aircraft.Wing.LOADING,
        Aircraft.Design.MAX_STRUCTURAL_SPEED,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.SWEEP,
        Mission.Design.MACH,
        Aircraft.Wing.AVERAGE_CHORD,
    ])

# fixed_mass
if simplified:
    x.add_input("fixed_mass", ["InputValues"])
else:
    x.add_input("fixed_mass", [
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
        "(prop_mass)",
        "("+Aircraft.Nacelle.CLEARANCE_RATIO+")",
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.SWEEP,
        Mission.Design.GROSS_MASS,
        Aircraft.Strut.ATTACHMENT_LOCATION,
        Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
        Aircraft.VerticalTail.ASPECT_RATIO,
        Aircraft.HorizontalTail.TAPER_RATIO,
        Aircraft.Engine.SCALED_SLS_THRUST,
        Aircraft.Engine.WING_LOCATIONS,
        Aircraft.VerticalTail.TAPER_RATIO,
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

# equip
if simplified:
    x.add_input("equip", ["InputValues"])
else:
    x.connect("equip", [
        Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
        Mission.Design.GROSS_MASS,
        Aircraft.Engine.SCALED_SLS_THRUST,
        Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
        Aircraft.Fuel.WING_FUEL_FRACTION,
        Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
        Aircraft.Fuselage.LENGTH,
        Aircraft.Wing.SPAN,
        Aircraft.HorizontalTail.AREA,
        Aircraft.VerticalTail.AREA,
        Aircraft.Fuselage.AVG_DIAMETER,
        Aircraft.Wing.AREA,
    ])

# wing_mass
if simplified:
    x.add_input("wing_mass", ["InputValues"])
else:
    x.add_input("wing_mass", [
        Aircraft.Wing.MASS_COEFFICIENT,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
        Mission.Design.GROSS_MASS,
        Aircraft.Wing.SPAN,
    ])

# fuel_mass
if simplified:
    x.add_input("fuel_mass", ["InputValues"])
else:
    x.add_input("fuel_mass", [
        Aircraft.Fuel.FUEL_MARGIN,
        Aircraft.Fuselage.MASS_COEFFICIENT,
        "fus_and_struct.pylon_len",
        "fus_and_struct.MAT",
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
        Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
        Mission.Design.GROSS_MASS,
        Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
        Aircraft.Fuselage.WETTED_AREA,
        Aircraft.Fuselage.AVG_DIAMETER,
        Aircraft.TailBoom.LENGTH,
    ])


x.connect("design_load", "fixed_mass", [
    "max_mach",
    "min_dive_vel",
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR
])
x.connect("design_load", "wing_mass", [Aircraft.Wing.ULTIMATE_LOAD_FACTOR])
x.connect("design_load", "fuel_mass", [
    "min_dive_vel",
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
])

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
    Aircraft.LandingGear.TOTAL_MASS
])
x.connect("equip", "fuel_mass", [
    Aircraft.Design.FIXED_EQUIPMENT_MASS,
    Aircraft.Design.FIXED_USEFUL_LOAD
])
x.connect("wing_mass", "fuel_mass", [Aircraft.Wing.MASS])

x.write("mass_basic_xdsm")
# x.write_sys_specs("mass_basic_specs")
