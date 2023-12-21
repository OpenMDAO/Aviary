from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

simplified = False
show_outputs = True

# Create subsystem components
x.add_system("equip", FUNC, ["EquipAndUsefulMass"])

### make input connections ###
if simplified is True:
    # FixedEquipMass
    x.add_input("equip", ["InputValues"])
else:
    x.add_input("equip", [
        Mission.Design.GROSS_MASS,
        Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
        Aircraft.Fuselage.LENGTH,
        Aircraft.Wing.SPAN,
        Aircraft.LandingGear.TOTAL_MASS,
        Aircraft.Controls.TOTAL_MASS,
        Aircraft.HorizontalTail.AREA,
        Aircraft.VerticalTail.AREA,
        Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
        Aircraft.Fuselage.AVG_DIAMETER,
        Aircraft.Engine.SCALED_SLS_THRUST,
        Aircraft.Fuel.WING_FUEL_FRACTION,
        Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
        Aircraft.Wing.AREA,
    ])

### add outputs ###
if show_outputs is True:
    x.add_output("equip", [
        Aircraft.Design.FIXED_USEFUL_LOAD,
        Aircraft.Design.FIXED_EQUIPMENT_MASS,
    ], side="right")

x.write("equipment_and_useful_load_mass_xdsm")
x.write_sys_specs("equipment_and_useful_load_mass_specs")
