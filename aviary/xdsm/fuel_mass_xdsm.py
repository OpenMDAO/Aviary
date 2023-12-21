from pyxdsm.XDSM import FUNC, GROUP, IFUNC, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

simplified = True
show_outputs = True

# Create subsystem components
x.add_system("sys_and_fus", FUNC, ["FuelSysAndFullFuselageMass"])
x.add_system("fus_and_struct", FUNC, ["FuselageAndStructMass"])
x.add_system("fuel", FUNC, ["FuelMass"])
x.add_system("fuel_and_oem", FUNC, ["FuelAndOEMOutputs"])
x.add_system("body", IFUNC, ["BodyTankCalculations"])

### make input connections ###
if simplified is True:
    x.add_input("sys_and_fus", ["InputValues"])
    x.add_input("fus_and_struct", ["InputValues"])
    x.add_input("fuel", ["InputValues"])
    x.add_input("fuel_and_oem", ["InputValues"])
else:
    x.add_input("sys_and_fus", [
        "wing_mounted_mass",
        Mission.Design.GROSS_MASS,  # GrossMassInitial
        Aircraft.Wing.MASS,  # TotalWingMass
        Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,  # CK21
        Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,  # CMassTrendFuelSys
        Aircraft.Fuel.DENSITY,  # RhoFuel
        Aircraft.Fuel.FUEL_MARGIN,  # FuelMargin
    ])

    x.add_input("fus_and_struct", [
        Aircraft.Wing.MASS,  # TotalWingMass
        Aircraft.Fuselage.MASS_COEFFICIENT,  # CFuselage
        Aircraft.Fuselage.WETTED_AREA,  # FusSA
        Aircraft.Fuselage.AVG_DIAMETER,   # CabinWidth
        Aircraft.TailBoom.LENGTH,  # CabinLenTailboom
        "pylon_len",
        "min_dive_vel",
        Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,  # PDiffFus
        Aircraft.Wing.ULTIMATE_LOAD_FACTOR,  # ULF
        "MAT",
        Aircraft.Wing.MASS_SCALER,  # CK8
        Aircraft.HorizontalTail.MASS_SCALER,  # CK9
        Aircraft.HorizontalTail.MASS,  # HtailMass
        Aircraft.VerticalTail.MASS_SCALER,  # CK10
        Aircraft.VerticalTail.MASS,  # VtailMass
        Aircraft.Fuselage.MASS_SCALER,  # CK11
        Aircraft.LandingGear.TOTAL_MASS_SCALER,  # CK12
        Aircraft.LandingGear.TOTAL_MASS,  # LandingGearMass
        Aircraft.Engine.POD_MASS_SCALER,  # CK14
        Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,  # SecMass
        Aircraft.Design.STRUCTURAL_MASS_INCREMENT,  # DeltaStructMass
    ])

    x.add_input("fuel", [
        "eng_comb_mass",
        "payload_mass_des",
        "payload_mass_max",
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,  # PayloadMass
        Mission.Design.GROSS_MASS,  # GrossMassInitial
        Aircraft.Controls.TOTAL_MASS,  # ControlMass
        Aircraft.Design.FIXED_EQUIPMENT_MASS,  # FixedEquipMass
        Aircraft.Design.FIXED_USEFUL_LOAD,  # UsefulMass
        Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,  # CK21
        Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,  # CMassTrendFuelSys
        Aircraft.Fuel.DENSITY,  # RhoFuel
        Aircraft.Fuel.FUEL_MARGIN,  # FuelMargin
    ])

    x.add_input("fuel_and_oem", [
        Aircraft.Fuel.DENSITY,  # RhoFuel
        Mission.Design.GROSS_MASS,  # GrossMassInitial
        # Aircraft.Propulsion.MASS,
        Aircraft.Controls.TOTAL_MASS,  # ControlMass
        # Aircraft.Design.STRUCTURE_MASS,
        Aircraft.Design.FIXED_EQUIPMENT_MASS,  # FixedEquipMass
        Aircraft.Design.FIXED_USEFUL_LOAD,  # UsefulMass
        Mission.Design.FUEL_MASS_REQUIRED,
        Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,  # GeometricFuelVol
        Aircraft.Fuel.FUEL_MARGIN,  # FuelMargin
        Aircraft.Fuel.TOTAL_CAPACITY,
    ])

    x.add_input("body", [
        Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,  # GeometricFuelVol
        Aircraft.Fuel.DENSITY,  # RhoFuel
        Mission.Design.GROSS_MASS  # GrossMassInitial
    ])

### make component connections ###
x.connect("sys_and_fus", "fus_and_struct", ["fus_mass_full"])  # FusMassFull
x.connect("sys_and_fus", "fuel", [Aircraft.Fuel.FUEL_SYSTEM_MASS])
x.connect("fus_and_struct", "fuel", [Aircraft.Design.STRUCTURE_MASS])
x.connect("fus_and_struct", "fuel_and_oem", [Aircraft.Design.STRUCTURE_MASS])
x.connect("fuel", "sys_and_fus", [Mission.Design.FUEL_MASS])  # FuelMassDes
x.connect("fuel", "fuel_and_oem", [
    Aircraft.Propulsion.MASS,
    Mission.Design.FUEL_MASS_REQUIRED,  # ReqFuelMass
])
x.connect("fuel", "body", [
    "fuel_mass_min",  # FuelMassMin
    Mission.Design.FUEL_MASS_REQUIRED,  # ReqFuelMass
    Mission.Design.FUEL_MASS,  # FuelMassDes
])
x.connect("fuel_and_oem", "body", [
    Aircraft.Fuel.WING_VOLUME_DESIGN,  # DesignFuelVol
    Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,  # MaxWingfuelVol
    "max_wingfuel_mass",  # MaxWingfuelMass
    Aircraft.Design.OPERATING_MASS,
])
x.connect("body", "sys_and_fus", ["wingfuel_mass_min"])  # WingfuelMassMin
x.connect("body", "fuel_and_oem", [Aircraft.Fuel.TOTAL_CAPACITY])  # MaxFuelAvail

### add outputs ###
if show_outputs is True:
    # fus_and_struct
    x.add_output("fus_and_struct", [Aircraft.Fuselage.MASS], side="right")

    # fuel_and_oem
    x.add_output("fuel_and_oem", [
        "OEM_wingfuel_mass",
        "OEM_fuel_vol",
        "payload_mass_max_fuel",  # PayloadMassMaxFuel
        "volume_wingfuel_mass",  # VolumeWingfuelMass
    ], side="right")

    # body
    x.add_output("body", [
        Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,  # ExtraFuelMass
        "extra_fuel_volume",  # ExtraFuelVolume
        "max_extra_fuel_mass",  # MaxExtraFuelMass
    ], side="right")

x.write("fuel_mass_xdsm")
x.write_sys_specs("fuel_mass_specs")
