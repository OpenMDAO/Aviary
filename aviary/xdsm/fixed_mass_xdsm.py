from pyxdsm.XDSM import FUNC, GROUP, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

simplified = False
show_outputs = True
has_hybrid_system = False
has_propellers = False

# Create subsystem components
x.add_system("params", FUNC, ["MassParameters"])
x.add_system("payload", FUNC, ["PayloadMass"])
x.add_system("tail", FUNC, ["TailMass"])
x.add_system("HL", FUNC, ["HighLiftMass"])
x.add_system("controls", FUNC, ["ControlMass"])
x.add_system("gear", FUNC, ["GearMass"])
if has_hybrid_system:
    x.add_system("aug", FUNC, [r"\textcolor{gray}{ElectricAugmentationMass}"])
x.add_system("engine", FUNC, ["EngineMass"])

### make input connections ###
if simplified is True:
    x.add_input("params", ["InputValues"])
    x.add_input("payload", ["InputValues"])
    x.add_input("tail", ["InputValues"])
    x.add_input("HL", ["InputValues"])
    x.add_input("controls", ["InputValues"])
    x.add_input("gear", ["InputValues"])
    if has_hybrid_system:
        x.add_input("aug", [r"\textcolor{gray}{InputValues}"])
    x.add_input("engine", ["InputValues"])
else:
    x.add_input("params", [
        Aircraft.Wing.SPAN,
        Aircraft.Wing.SWEEP,  # SweepC4
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.ASPECT_RATIO,
        "max_mach",
        Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS,  # StrutX
        Aircraft.LandingGear.MAIN_GEAR_LOCATION,
    ])
    x.add_input("payload", [
        # Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS,  # PaxMass. This is an option
        Aircraft.CrewPayload.CARGO_MASS,  # CargoMass
    ])
    x.add_input("tail", [
        Aircraft.Wing.SWEEP,
        Mission.Design.GROSS_MASS,  # GrossMassInitial
        "min_dive_vel",
        Aircraft.Wing.SPAN,
        Aircraft.VerticalTail.TAPER_RATIO,  # TaperRatioVtail
        Aircraft.VerticalTail.ASPECT_RATIO,  # ARVtail
        Aircraft.VerticalTail.SWEEP,  # QuarterSweepTail
        Aircraft.VerticalTail.SPAN,  # SpanVtail
        Aircraft.HorizontalTail.MASS_COEFFICIENT,  # CoefHtail
        Aircraft.Fuselage.LENGTH,
        Aircraft.HorizontalTail.SPAN,
        Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
        Aircraft.HorizontalTail.TAPER_RATIO,  # TaperRatioHtail
        Aircraft.VerticalTail.MASS_COEFFICIENT,  # CoefVtail
        Aircraft.HorizontalTail.AREA,  # HtailArea
        Aircraft.HorizontalTail.MOMENT_ARM,  # HtailMomArm
        Aircraft.HorizontalTail.THICKNESS_TO_CHORD,  # TcRatioRootHtail
        Aircraft.HorizontalTail.ROOT_CHORD,  # RootChordHtail
        Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,  # TailLoc
        Aircraft.VerticalTail.AREA,
        Aircraft.VerticalTail.MOMENT_ARM,  # VtailMomArm
        Aircraft.VerticalTail.THICKNESS_TO_CHORD,  # TcRatioRootVtail
        Aircraft.VerticalTail.ROOT_CHORD,  # RootChordVtail
        # "CMassTrendHighLift",
    ])
    x.add_input("HL", [
        "density",
        Aircraft.Wing.SWEEP,
        # "GrossMassInitial",
        # "minDiveVel",
        Aircraft.Wing.AREA,
        Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,  # CMassTrendHighLift
        # "wing_area",
        # "num_flaps",
        Aircraft.Wing.SLAT_CHORD_RATIO,  # slat_chord_ratio
        Aircraft.Wing.FLAP_CHORD_RATIO,  # flap_chord_ratio
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.FLAP_SPAN_RATIO,  # flap_span_ratio
        Aircraft.Wing.SLAT_SPAN_RATIO,  # slat_span_ratio
        Aircraft.Wing.LOADING,
        Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,  # tc_ratio_root
        Aircraft.Wing.SPAN,
        Aircraft.Fuselage.AVG_DIAMETER,  # cabin_width
        Aircraft.Wing.CENTER_CHORD,
        Mission.Landing.LIFT_COEFFICIENT_MAX,  # CL_max_flaps_landing
    ])
    x.add_input("controls", [
        Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
        Mission.Design.GROSS_MASS,  # GrossMassInitial
        "min_dive_vel",
        Aircraft.Wing.AREA,
        # "CMassTrendWingControl",
        Aircraft.Wing.ULTIMATE_LOAD_FACTOR,  # ULF
        Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,  # CMassTrendCockpitControl
        Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,  # StabAugMass
        Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER,  # CK15
        Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,  # CK18
        Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,  # CK19
        Aircraft.Controls.CONTROL_MASS_INCREMENT,  # DeltaControlMass
    ])
    x.add_input("gear", [
        Aircraft.Wing.MOUNTING_TYPE,
        Mission.Design.GROSS_MASS,  # GrossMassInitial
        Aircraft.LandingGear.MASS_COEFFICIENT,  # CGearMass
        Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT,  # CMainGear
        Aircraft.Nacelle.CLEARANCE_RATIO,  # ClearanceRatio
        Aircraft.Nacelle.AVG_DIAMETER,  # NacelleDiam
    ])

    if has_hybrid_system:
        x.add_input("aug", [
            r"\textcolor{gray}{motor_power}",  # MotorPowerKw
            r"\textcolor{gray}{motor_voltage}",  # MotorVoltage
            r"\textcolor{gray}{max_amp_per_wire}",  # MaxAmpPerWire
            r"\textcolor{gray}{safety_factor}",  # SafetyFactor
            r"\textcolor{gray}{"
            + Aircraft.Electrical.HYBRID_CABLE_LENGTH + "}",  # CableLen
            r"\textcolor{gray}{wire_area}",
            r"\textcolor{gray}{rho_wire}",
            r"\textcolor{gray}{battery_energy}",
            r"\textcolor{gray}{motor_eff}",
            r"\textcolor{gray}{inverter_eff}",
            r"\textcolor{gray}{transmission_eff}",
            r"\textcolor{gray}{battery_eff}",
            r"\textcolor{gray}{rho_battery}",
            r"\textcolor{gray}{motor_spec_mass}",
            r"\textcolor{gray}{inverter_spec_mass}",
            r"\textcolor{gray}{TMS_spec_mass}",
        ])
    # engine
    engine_inputs = [
        Aircraft.Engine.MASS_SPECIFIC,
        Aircraft.Engine.SCALED_SLS_THRUST,
        Aircraft.Nacelle.MASS_SPECIFIC,
        Aircraft.Nacelle.SURFACE_AREA,
        Aircraft.Engine.PYLON_FACTOR,
        Aircraft.Engine.ADDITIONAL_MASS_FRACTION,
        Aircraft.Engine.MASS_SCALER,
        Aircraft.Propulsion.MISC_MASS_SCALER,
        Aircraft.Engine.WING_LOCATIONS,
        Aircraft.LandingGear.MAIN_GEAR_LOCATION,
    ]
    if has_hybrid_system:
        engine_inputs.append(r"\textcolor{gray}{aug_mass}")
    if has_propellers:
        engine_inputs.append(r"\textcolor{gray}{prop_mass}")
    x.add_input("engine", engine_inputs)

### make component connections ###
x.connect("gear", "engine", ["main_gear_mass"])
if has_hybrid_system:
    x.connect("aug", "engine", [r"\textcolor{gray}{aug_mass}"])

### add outputs ###
if show_outputs is True:
    x.add_output("params", [
        Aircraft.Wing.MATERIAL_FACTOR,  # CMaterial
        "c_strut_braced",
        "c_gear_loc",
        Aircraft.Engine.POSITION_FACTOR,  # CEngPos
        "half_sweep",
    ], side="right")

    x.add_output("payload", [
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
        "payload_mass_des",
        "payload_mass_max",
    ], side="right")

    x.add_output("tail", [
        "loc_MAC_vtail",
        Aircraft.HorizontalTail.MASS,
        Aircraft.VerticalTail.MASS
    ], side="right")

    x.add_output("HL", [
        Aircraft.Wing.HIGH_LIFT_MASS,
        "flap_mass",
        "slat_mass",
    ], side="right")

    x.add_output("controls", [
        Aircraft.Controls.TOTAL_MASS,
        Aircraft.Wing.SURFACE_CONTROL_MASS,
    ], side="right")

    x.add_output("gear", [
        Aircraft.LandingGear.TOTAL_MASS,
        "main_gear_mass",
    ], side="right")

    if has_hybrid_system:
        x.add_output("aug", [
            r"\textcolor{gray}{aug_mass}",
        ], side="right")

    # engine
    engine_outputs = [
        "eng_comb_mass",
        "wing_mounted_mass",
        "pylon_mass",
        Aircraft.Propulsion.TOTAL_ENGINE_MASS,
        Aircraft.Nacelle.MASS,
        Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,
        Aircraft.Engine.ADDITIONAL_MASS,
    ]
    if has_propellers:
        engine_outputs.append(r"\textcolor{gray}{prop_mass_all}")
    x.add_output("engine", engine_outputs, side="right")

x.write("fixed_mass_xdsm")
x.write_sys_specs("fixed_mass_specs")
