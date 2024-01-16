from pyxdsm.XDSM import FUNC, GROUP, IFUNC, OPT, XDSM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

x = XDSM()

simplified = False

# Create subsystem components
if simplified is False:
    x.add_system("shared", GROUP, ["SharedInputs"])
else:
    x.add_system("inputs", GROUP, ["Inputs"])
x.add_system("opt", OPT, ["Optimizer"])
x.add_system(Dynamic.Mission.MASS, GROUP, [r"\textbf{MassSummation}"])
x.add_system("xform", FUNC, ["EventXform"])
x.add_system("dymos", GROUP, ["Dymos"])
x.add_system("taxi", GROUP, ["Taxi"])
x.add_system("groundroll", GROUP, [r"\textbf{GroundRoll}"])
x.add_system("rotation", GROUP, [r"\textbf{Rotation}"])
x.add_system("ascent", GROUP, [r"\textbf{InitialAscent}"])
x.add_system("accelerate", GROUP, [r"\textbf{Accelerate}"])
x.add_system("climb1", GROUP, [r"\textbf{ClimbTo10kFt}"])
x.add_system("climb2", GROUP, [r"\textbf{ClimbToCruise}"])
x.add_system("poly", IFUNC, ["PolynomialFit"])
x.add_system("cruise", GROUP, [r"\textbf{BreguetRange}"])
x.add_system("descent1", GROUP, [r"\textbf{DescentTo10kFt}"])
x.add_system("descent2", GROUP, [r"\textbf{DescentTo1kFt}"])
x.add_system("landing", GROUP, [r"\textbf{Landing}"])
x.add_system("fuelburn", FUNC, ["FuelBurn"])
x.add_system("mass_diff", FUNC, ["MassDifference"])
x.add_system(Dynamic.Mission.DISTANCE, FUNC, ["RangeConstraint"])

if simplified is False:
    # independent vars input to ParamPort, common to all phases
    ivc_params = [
        Aircraft.Wing.INCIDENCE,
        Aircraft.Wing.HEIGHT,
        Mission.Summary.FUEL_FLOW_SCALER,
        Mission.Takeoff.AIRPORT_ALTITUDE,
        Mission.Landing.AIRPORT_ALTITUDE,
        Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF,
        Aircraft.Wing.FLAP_DEFLECTION_LANDING,
        Aircraft.Wing.ASPECT_RATIO,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
        Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
        Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
        Aircraft.Wing.SWEEP,
        Aircraft.HorizontalTail.SWEEP,
        Aircraft.HorizontalTail.MOMENT_RATIO,
        Aircraft.Wing.MOUNTING_TYPE,
        Aircraft.Design.STATIC_MARGIN,
        Aircraft.Design.CG_DELTA,
        Aircraft.Wing.FORM_FACTOR,
        Aircraft.Fuselage.FORM_FACTOR,
        Aircraft.Nacelle.FORM_FACTOR,
        Aircraft.VerticalTail.FORM_FACTOR,
        Aircraft.HorizontalTail.FORM_FACTOR,
        Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR,
        Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR,
        Aircraft.Design.DRAG_COEFFICIENT_INCREMENT,
        Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT,
        Aircraft.Wing.CENTER_DISTANCE,
        Aircraft.Wing.MIN_PRESSURE_LOCATION,
        Aircraft.Wing.MAX_THICKNESS_LOCATION,
        Aircraft.Strut.AREA_RATIO,
        Aircraft.Wing.ZERO_LIFT_ANGLE,
        Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT,
        Aircraft.Wing.FLAP_CHORD_RATIO,
        Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
        Mission.Takeoff.LIFT_COEFFICIENT_MAX,
        Mission.Landing.LIFT_COEFFICIENT_MAX,
        Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
        Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
        Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
        Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
        Aircraft.Strut.CHORD,  # normally would be output by sizing
        Mission.Summary.GROSS_MASS,
    ]

    # ParamPort inputs from mass/sizing
    sizing_params = [
        Aircraft.Wing.AREA,
        Aircraft.Wing.SPAN,
        Aircraft.Wing.AVERAGE_CHORD,
        Aircraft.Fuselage.AVG_DIAMETER,
        Aircraft.HorizontalTail.AVERAGE_CHORD,
        Aircraft.HorizontalTail.AREA,
        Aircraft.HorizontalTail.SPAN,
        Aircraft.VerticalTail.AVERAGE_CHORD,
        Aircraft.VerticalTail.AREA,
        Aircraft.VerticalTail.SPAN,
        Aircraft.Fuselage.LENGTH,
        Aircraft.Nacelle.AVG_LENGTH,
        Aircraft.Fuselage.WETTED_AREA,
        Aircraft.Nacelle.SURFACE_AREA,
        Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
        # Aircraft.Strut.CHORD,
    ]

    # Connect shared inputs
    x.connect("shared", "taxi", ivc_params)
    x.connect("shared", "groundroll", ivc_params)
    x.connect("shared", "rotation", ivc_params)
    x.connect("shared", "ascent", ivc_params)
    x.connect("shared", "accelerate", ivc_params)
    x.connect("shared", "climb1", ivc_params)
    x.connect("shared", "climb2", ivc_params)
    x.connect("shared", "cruise", ivc_params)
    x.connect("shared", "descent1", ivc_params)
    x.connect("shared", "descent2", ivc_params)
    x.connect("shared", "landing", ivc_params)

else:
    x.connect("inputs", Dynamic.Mission.MASS, ["InputValue"])
    x.connect("inputs", "taxi", ["InputValue"])
    x.connect("inputs", "groundroll", ["InputValue"])
    x.connect("inputs", "rotation", ["InputValue"])
    x.connect("inputs", "ascent", ["InputValue"])
    x.connect("inputs", "accelerate", ["InputValue"])
    x.connect("inputs", "climb1", ["InputValue"])
    x.connect("inputs", "climb2", ["InputValue"])
    x.connect("inputs", "cruise", ["InputValue"])
    x.connect("inputs", "descent1", ["InputValue"])
    x.connect("inputs", "descent2", ["InputValue"])
    x.connect("inputs", "landing", ["InputValue"])

# Connect mass
x.connect(
    Dynamic.Mission.MASS,
    "mass_diff",
    [Aircraft.Design.OPERATING_MASS, Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS],
)

x.connect("opt", Dynamic.Mission.MASS, [Mission.Design.GROSS_MASS])
x.connect("opt", "taxi", [Mission.Design.GROSS_MASS])
x.connect("opt", "landing", [Mission.Design.GROSS_MASS])
x.connect("fuelburn", "mass_diff", [Mission.Design.FUEL_MASS_REQUIRED])

# Connect sizing calculated values
if simplified is False:
    x.connect(Dynamic.Mission.MASS, "taxi", sizing_params)
    x.connect(Dynamic.Mission.MASS, "groundroll", sizing_params)
    x.connect(Dynamic.Mission.MASS, "rotation", sizing_params)
    x.connect(Dynamic.Mission.MASS, "ascent", sizing_params)
    x.connect(Dynamic.Mission.MASS, "accelerate", sizing_params)
    x.connect(Dynamic.Mission.MASS, "climb1", sizing_params)
    x.connect(Dynamic.Mission.MASS, "climb2", sizing_params)
    x.connect(Dynamic.Mission.MASS, "cruise", sizing_params)
    x.connect(Dynamic.Mission.MASS, "descent1", sizing_params)
    x.connect(Dynamic.Mission.MASS, "descent2", sizing_params)
    x.connect(Dynamic.Mission.MASS, "landing", sizing_params)

# Connect miscellaneous
x.connect("xform", "dymos", ["t_init_gear", "t_init_flaps"])
x.connect("xform", "ascent", ["t_init_gear", "t_init_flaps"])
x.connect("xform", "poly", ["t_init_gear", "t_init_flaps"])

if simplified is False:
    # Create inputs
    x.add_input(
        Dynamic.Mission.MASS,
        [
            Aircraft.HorizontalTail.TAPER_RATIO,
            Aircraft.HorizontalTail.MOMENT_RATIO,
            Aircraft.HorizontalTail.ASPECT_RATIO,
            Aircraft.VerticalTail.MOMENT_RATIO,
            Aircraft.VerticalTail.TAPER_RATIO,
            Aircraft.Engine.REFERENCE_DIAMETER,
            Aircraft.Nacelle.CORE_DIAMETER_RATIO,
            Aircraft.Nacelle.FINENESS,
            Aircraft.Fuselage.DELTA_DIAMETER,
            Aircraft.Fuselage.NOSE_FINENESS,
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
            Aircraft.Fuselage.TAIL_FINENESS,
            Aircraft.Fuselage.WETTED_AREA_FACTOR,
            Aircraft.Wing.LOADING,
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
            Aircraft.Design.MAX_STRUCTURAL_SPEED,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.SWEEP,
            Mission.Design.MACH,
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
            Aircraft.Fuel.WING_FUEL_FRACTION,
            Aircraft.Wing.SWEEP,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS,
            Aircraft.CrewPayload.CARGO_MASS,
            Aircraft.Engine.MASS_SPECIFIC,
            Aircraft.Nacelle.MASS_SPECIFIC,
            Aircraft.Engine.PYLON_FACTOR,
            Aircraft.Engine.ADDITIONAL_MASS_FRACTION,
            Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Engine.SCALED_SLS_THRUST,
            Aircraft.Engine.MASS_SCALER,
            Aircraft.Propulsion.MISC_MASS_SCALER,
            Aircraft.Engine.WING_LOCATIONS,
            Aircraft.LandingGear.MAIN_GEAR_LOCATION,
            Aircraft.VerticalTail.ASPECT_RATIO,
            Aircraft.VerticalTail.SWEEP,
            Aircraft.HorizontalTail.MASS_COEFFICIENT,
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
            Aircraft.VerticalTail.MASS_COEFFICIENT,
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
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
            Aircraft.Wing.MOUNTING_TYPE,
            Aircraft.Fuel.DENSITY,
            Aircraft.Fuel.FUEL_MARGIN,
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
            Aircraft.Fuselage.MASS_COEFFICIENT,
            "fuel_mass.fus_and_struct.pylon_len",
            "fuel_mass.fus_and_struct.MAT",
            Aircraft.Wing.MASS_SCALER,
            Aircraft.HorizontalTail.MASS_SCALER,
            Aircraft.VerticalTail.MASS_SCALER,
            Aircraft.Fuselage.MASS_SCALER,
            Aircraft.LandingGear.TOTAL_MASS_SCALER,
            Aircraft.Engine.POD_MASS_SCALER,
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT,
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
            Mission.Design.GROSS_MASS,
            Aircraft.Wing.MASS_COEFFICIENT,
            Aircraft.Wing.TAPER_RATIO,
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
            Aircraft.VerticalTail.VOLUME_COEFFICIENT,
            Aircraft.Nacelle.CLEARANCE_RATIO,
            Aircraft.Wing.FLAP_CHORD_RATIO,
            Aircraft.Wing.FLAP_SPAN_RATIO,
            Aircraft.Wing.SLAT_CHORD_RATIO,
            Aircraft.Wing.SLAT_SPAN_RATIO,
            Mission.Landing.LIFT_COEFFICIENT_MAX,
            "density",
            Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
        ],
    )

    aero_in = [
        'aero_ramps.flap_factor:final_val',
        'aero_ramps.gear_factor:final_val',
        'aero_ramps.flap_factor:initial_val',
        'aero_ramps.gear_factor:initial_val',
    ]

    # common propulsion inputs that are set internally but propagate up to IVC
    prop_in = [
        Aircraft.Engine.SCALE_FACTOR,
    ]

    x.add_input("taxi", ['throttle',
                         'propulsion.vectorize_performance.t4_0',
                         'propulsion.aircraft:engine:scale_factor'])
    x.add_input("groundroll",
                ["dt_flaps",
                 "dt_gear",
                 "t_init_gear",
                 "t_init_flaps",
                 Dynamic.Mission.FLIGHT_PATH_ANGLE,
                 Dynamic.Mission.ALTITUDE] + aero_in +
                [
                    'throttle',
                    'vectorize_performance.t4_0',
                    'aircraft:engine:scale_factor',
                ],
                )
    x.add_input("rotation",
                ["dt_flaps", "dt_gear", "t_init_gear", "t_init_flaps"] +
                aero_in + prop_in +
                ['throttle', 'vectorize_performance.t4_0']
                )
    x.add_input("ascent", ["dt_flaps",
                           "dt_gear",
                           'vectorize_performance.t4_0',
                           'throttle'] + aero_in + prop_in)
    x.add_input("accelerate", ['throttle',
                               'vectorize_performance.t4_0'] + prop_in)
    x.add_input("climb1", ["speed_bal.rhs:EAS",
                           'vectorize_performance.t4_0',
                           'throttle'] + prop_in)
    x.add_input("climb2", ["speed_bal.rhs:EAS",
                           'vectorize_performance.t4_0',
                           'throttle'] + prop_in)
    x.add_input(
        "cruise",
        [
            'prop_initial.prop.vectorize_performance.t4_0',
            'prop_final.prop.vectorize_performance.t4_0',
            'prop_final.prop.aircraft:engine:scale_factor',
            'prop_initial.prop.aircraft:engine:scale_factor'
        ]
    )
    x.add_input("descent1", ["speed_bal.rhs:mach",
                             'aircraft:engine:scale_factor',
                             'throttle',
                             'vectorize_performance.t4_0'])
    x.add_input("descent2", ["EAS",
                             'aircraft:engine:scale_factor',
                             'vectorize_performance.t4_0',
                             'throttle'])
    x.add_input(
        "landing",
        [
            Mission.Landing.AIRPORT_ALTITUDE,
            Mission.Landing.OBSTACLE_HEIGHT,
            Mission.Landing.INITIAL_MACH,
            "alpha",
            Mission.Landing.MAXIMUM_SINK_RATE,
            Mission.Landing.GLIDE_TO_STALL_RATIO,
            Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
            Mission.Landing.TOUCHDOWN_SINK_RATE,
            Mission.Landing.BRAKING_DELAY,
            "dt_flaps",
            "dt_gear",
            "t_init_flaps_app",
            "t_init_gear_app",
            "t_init_flaps_td",
            "t_init_gear_td",
            "t_curr",
            'aero_ramps.flap_factor:final_val',
            'aero_ramps.gear_factor:final_val',
            'aero_ramps.flap_factor:initial_val',
            'aero_ramps.gear_factor:initial_val',
            Dynamic.Mission.THROTTLE,
            'vectorize_performance.t4_0',
            'aircraft:engine:scale_factor'
        ]

    )
    x.add_input("fuelburn", [Aircraft.Fuel.FUEL_MARGIN, Mission.Summary.GROSS_MASS])
    x.add_input(Dynamic.Mission.DISTANCE, [Mission.Design.RANGE])

# Create outputs
x.add_output("landing", [Mission.Landing.GROUND_DISTANCE], side="right")

# Create phase continuities
x.connect("dymos", "groundroll", [Dynamic.Mission.MASS, "TAS", "t_curr"])
x.connect("dymos", "rotation", [Dynamic.Mission.MASS, "TAS",
          "alpha", Dynamic.Mission.FLIGHT_PATH_ANGLE, "t_curr", Dynamic.Mission.ALTITUDE])
x.connect("dymos", "ascent", [Dynamic.Mission.MASS, Dynamic.Mission.ALTITUDE,
          "TAS", Dynamic.Mission.FLIGHT_PATH_ANGLE, "t_curr", "alpha"])
x.connect("dymos", "accelerate", [Dynamic.Mission.ALTITUDE, "TAS", Dynamic.Mission.MASS])
x.connect("dymos", "climb1", [Dynamic.Mission.ALTITUDE, Dynamic.Mission.MASS])
x.connect("dymos", "climb2", [Dynamic.Mission.ALTITUDE, Dynamic.Mission.MASS])
x.connect("dymos", "poly", ["time_cp", "h_cp"])
x.connect("dymos", "descent2", [Dynamic.Mission.ALTITUDE, Dynamic.Mission.MASS])
x.connect("dymos", "landing", [Dynamic.Mission.MASS])
x.connect("taxi", "groundroll", [Dynamic.Mission.MASS])
x.connect("dymos",
          "cruise",
          [Dynamic.Mission.ALTITUDE,
           Dynamic.Mission.MACH,
           "mass_initial",
           "cruise_time_initial",
           "cruise_range_initial"],
          )
x.connect("dymos", "descent1", [Dynamic.Mission.ALTITUDE, Dynamic.Mission.MASS])
x.connect("dymos", "fuelburn", [Mission.Landing.TOUCHDOWN_MASS])
x.connect("dymos", Dynamic.Mission.DISTANCE, [Mission.Summary.RANGE])
x.connect("cruise", "dymos", ["cruise_time_final", "cruise_range_final"])

# Add Design Variables
x.connect(
    "opt",
    "dymos",
    [
        "cruise_mass_final",
        "ascent:t_initial",
        Mission.Takeoff.ASCENT_DURATION,
        Mission.Design.GROSS_MASS,
    ],
)
x.connect(
    "opt",
    "xform",
    [
        "tau_gear",
        "tau_flaps",
        "ascent:t_initial",
        Mission.Takeoff.ASCENT_DURATION,
    ],
)
x.connect("opt", "groundroll", [Mission.Design.GROSS_MASS])
x.connect("opt", "rotation", [Mission.Design.GROSS_MASS])
x.connect("opt", "ascent", [Mission.Design.GROSS_MASS])
x.connect("opt", "accelerate", [Mission.Design.GROSS_MASS])
x.connect("opt", "climb1", [Mission.Design.GROSS_MASS])
x.connect("opt", "climb2", [Mission.Design.GROSS_MASS])
x.connect("opt", "cruise", [Mission.Design.GROSS_MASS, "mass_final"])
x.connect("opt", "descent1", [Mission.Design.GROSS_MASS])
x.connect("opt", "descent2", [Mission.Design.GROSS_MASS])
x.connect("opt", "mass_diff", [Mission.Design.GROSS_MASS])

# Add Constraints
x.connect("dymos", "opt", [r"\mathcal{R}"])
x.connect("mass_diff", "opt", [Mission.Constraints.MASS_RESIDUAL])
x.connect(Dynamic.Mission.DISTANCE, "opt", [Mission.Constraints.RANGE_RESIDUAL])
x.connect("poly", "opt", ["h_init_gear", "h_init_flaps"])

# Connect State Rates
x.connect(
    "groundroll", "dymos", [
        "TAS_rate", Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, Dynamic.Mission.DISTANCE_RATE])
x.connect("rotation",
          "dymos",
          ["TAS_rate",
           Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
           Dynamic.Mission.DISTANCE_RATE,
           "alpha_rate"])
x.connect("ascent",
          "dymos",
          ["TAS_rate",
           Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
           Dynamic.Mission.DISTANCE_RATE,
           Dynamic.Mission.ALTITUDE_RATE,
           Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE],
          )
x.connect(
    "accelerate", "dymos", [
        "TAS_rate", Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, Dynamic.Mission.DISTANCE_RATE])
x.connect("climb1", "dymos", [Dynamic.Mission.ALTITUDE_RATE,
          Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, Dynamic.Mission.DISTANCE_RATE])
x.connect("climb2", "dymos", [Dynamic.Mission.ALTITUDE_RATE,
          Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, Dynamic.Mission.DISTANCE_RATE])
x.connect("descent1", "dymos", [Dynamic.Mission.ALTITUDE_RATE,
          Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, Dynamic.Mission.DISTANCE_RATE])
x.connect("descent2", "dymos", [Dynamic.Mission.ALTITUDE_RATE,
          Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, Dynamic.Mission.DISTANCE_RATE])

x.write("statics_xdsm")
x.write_sys_specs("statics_specs")
