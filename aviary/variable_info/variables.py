"""
This is a variable hierarchy that is for a single mission. Each mission
gets a copy of this hierarchy.
"""


class Aircraft:
    """Aircraft data hierarchy."""

    class AirConditioning:
        MASS = 'aircraft:air_conditioning:mass'
        MASS_COEFFICIENT = 'aircraft:air_conditioning:mass_coefficient'
        MASS_SCALER = 'aircraft:air_conditioning:mass_scaler'

    class AntiIcing:
        MASS = 'aircraft:anti_icing:mass'
        MASS_SCALER = 'aircraft:anti_icing:mass_scaler'

    class APU:
        MASS = 'aircraft:apu:mass'
        MASS_SCALER = 'aircraft:apu:mass_scaler'

    class Avionics:
        MASS = 'aircraft:avionics:mass'
        MASS_SCALER = 'aircraft:avionics:mass_scaler'

    class Battery:
        ADDITIONAL_MASS = 'aircraft:battery:additional_mass'
        DISCHARGE_LIMIT = 'aircraft:battery:discharge_limit'
        EFFICIENCY = 'aircraft:battery:efficiency'
        ENERGY_CAPACITY = 'aircraft:battery:energy_capacity'
        MASS = 'aircraft:battery:mass'
        PACK_ENERGY_DENSITY = 'aircraft:battery:pack_energy_density'
        PACK_MASS = 'aircraft:battery:pack_mass'
        PACK_VOLUMETRIC_DENSITY = 'aircraft:battery:pack_volumetric_density'
        VOLUME = 'aircraft:battery:volume'

    class BWB:
        NUM_BAYS = 'aircraft:blended_wing_body_design:num_bays'
        PASSENGER_LEADING_EDGE_SWEEP = (
            'aircraft:blended_wing_body_design:passenger_leading_edge_sweep'
        )

    class Canard:
        AREA = 'aircraft:canard:area'
        ASPECT_RATIO = 'aircraft:canard:aspect_ratio'
        CHARACTERISTIC_LENGTH = 'aircraft:canard:characteristic_length'
        FINENESS = 'aircraft:canard:fineness'
        LAMINAR_FLOW_LOWER = 'aircraft:canard:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:canard:laminar_flow_upper'
        MASS = 'aircraft:canard:mass'
        MASS_SCALER = 'aircraft:canard:mass_scaler'
        TAPER_RATIO = 'aircraft:canard:taper_ratio'
        THICKNESS_TO_CHORD = 'aircraft:canard:thickness_to_chord'
        WETTED_AREA = 'aircraft:canard:wetted_area'
        WETTED_AREA_SCALER = 'aircraft:canard:wetted_area_scaler'

    class Controls:
        COCKPIT_CONTROL_MASS_SCALER = 'aircraft:controls:cockpit_control_mass_scaler'
        CONTROL_MASS_INCREMENT = 'aircraft:controls:control_mass_increment'
        STABILITY_AUGMENTATION_SYSTEM_MASS = 'aircraft:controls:stability_augmentation_system_mass'
        STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER = (
            'aircraft:controls:stability_augmentation_system_mass_scaler'
        )
        TOTAL_MASS = 'aircraft:controls:total_mass'

    class CrewPayload:
        BAGGAGE_MASS = 'aircraft:crew_and_payload:baggage_mass'

        BAGGAGE_MASS_PER_PASSENGER = 'aircraft:crew_and_payload:baggage_mass_per_passenger'

        CARGO_CONTAINER_MASS = 'aircraft:crew_and_payload:cargo_container_mass'

        CARGO_CONTAINER_MASS_SCALER = 'aircraft:crew_and_payload:cargo_container_mass_scaler'

        CARGO_MASS = 'aircraft:crew_and_payload:cargo_mass'
        CATERING_ITEMS_MASS_PER_PASSENGER = (
            'aircraft:crew_and_payload:catering_items_mass_per_passenger'
        )

        FLIGHT_CREW_MASS = 'aircraft:crew_and_payload:flight_crew_mass'

        FLIGHT_CREW_MASS_SCALER = 'aircraft:crew_and_payload:flight_crew_mass_scaler'

        MASS_PER_PASSENGER = 'aircraft:crew_and_payload:mass_per_passenger'

        MISC_CARGO = 'aircraft:crew_and_payload:misc_cargo'

        NON_FLIGHT_CREW_MASS = 'aircraft:crew_and_payload:non_flight_crew_mass'

        NON_FLIGHT_CREW_MASS_SCALER = 'aircraft:crew_and_payload:non_flight_crew_mass_scaler'

        NUM_BUSINESS_CLASS = 'aircraft:crew_and_payload:num_business_class'
        NUM_FIRST_CLASS = 'aircraft:crew_and_payload:num_first_class'

        NUM_FLIGHT_ATTENDANTS = 'aircraft:crew_and_payload:num_flight_attendants'

        NUM_FLIGHT_CREW = 'aircraft:crew_and_payload:num_flight_crew'
        NUM_GALLEY_CREW = 'aircraft:crew_and_payload:num_galley_crew'

        NUM_PASSENGERS = 'aircraft:crew_and_payload:num_passengers'
        NUM_TOURIST_CLASS = 'aircraft:crew_and_payload:num_tourist_class'

        PASSENGER_MASS = 'aircraft:crew_and_payload:passenger_mass'
        PASSENGER_MASS_WITH_BAGS = 'aircraft:crew_and_payload:passenger_mass_with_bags'

        PASSENGER_PAYLOAD_MASS = 'aircraft:crew_and_payload:passenger_payload_mass'

        PASSENGER_SERVICE_MASS = 'aircraft:crew_and_payload:passenger_service_mass'

        PASSENGER_SERVICE_MASS_PER_PASSENGER = (
            'aircraft:crew_and_payload:passenger_service_mass_per_passenger'
        )

        PASSENGER_SERVICE_MASS_SCALER = 'aircraft:crew_and_payload:passenger_service_mass_scaler'

        TOTAL_PAYLOAD_MASS = 'aircraft:crew_and_payload:total_payload_mass'
        ULD_MASS_PER_PASSENGER = 'aircraft:crew_and_payload:uld_mass_per_passenger'
        WATER_MASS_PER_OCCUPANT = 'aircraft:crew_and_payload:water_mass_per_occupant'
        WING_CARGO = 'aircraft:crew_and_payload:wing_cargo'

        class Design:
            CARGO_MASS = 'aircraft:crew_and_payload:design:cargo_mass'
            MAX_CARGO_MASS = 'aircraft:crew_and_payload:design:max_cargo_mass'
            NUM_BUSINESS_CLASS = 'aircraft:crew_and_payload:design:num_business_class'
            NUM_FIRST_CLASS = 'aircraft:crew_and_payload:design:num_first_class'
            NUM_TOURIST_CLASS = 'aircraft:crew_and_payload:design:num_tourist_class'
            NUM_PASSENGERS = 'aircraft:crew_and_payload:design:num_passengers'

    class Design:
        # These variables are values that do not fall into a particular aircraft
        # component.

        BASE_AREA = 'aircraft:design:base_area'
        CG_DELTA = 'aircraft:design:cg_delta'
        CHARACTERISTIC_LENGTHS = 'aircraft:design:characteristic_lengths'
        COCKPIT_CONTROL_MASS_COEFFICIENT = 'aircraft:design:cockpit_control_mass_coefficient'
        COMPUTE_HTAIL_VOLUME_COEFF = 'aircraft:design:compute_htail_volume_coeff'
        COMPUTE_VTAIL_VOLUME_COEFF = 'aircraft:design:compute_vtail_volume_coeff'
        DRAG_COEFFICIENT_INCREMENT = 'aircraft:design:drag_increment'
        DRAG_POLAR = 'aircraft:design:drag_polar'

        EMERGENCY_EQUIPMENT_MASS = 'aircraft:design:emergency_equipment_mass'
        EMPTY_MASS = 'aircraft:design:empty_mass'
        EMPTY_MASS_MARGIN = 'aircraft:design:empty_mass_margin'

        EMPTY_MASS_MARGIN_SCALER = 'aircraft:design:empty_mass_margin_scaler'

        EXTERNAL_SUBSYSTEMS_MASS = 'aircraft:design:external_subsystems_mass'
        FINENESS = 'aircraft:design:fineness'
        FIXED_EQUIPMENT_MASS = 'aircraft:design:fixed_equipment_mass'
        FIXED_USEFUL_LOAD = 'aircraft:design:fixed_useful_load'
        IJEFF = 'ijeff'
        LAMINAR_FLOW_LOWER = 'aircraft:design:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:design:laminar_flow_upper'

        LANDING_TO_TAKEOFF_MASS_RATIO = 'aircraft:design:landing_to_takeoff_mass_ratio'

        LIFT_CURVE_SLOPE = 'aircraft:design:lift_curve_slope'
        LIFT_DEPENDENT_DRAG_COEFF_FACTOR = 'aircraft:design:lift_dependent_drag_coeff_factor'

        LIFT_DEPENDENT_DRAG_POLAR = 'aircraft:design:lift_dependent_drag_polar'
        LIFT_INDEPENDENT_DRAG_POLAR = 'aircraft:design:lift_independent_drag_polar'

        LIFT_POLAR = 'aircraft:design:lift_polar'

        MAX_FUSELAGE_PITCH_ANGLE = 'aircraft:design:max_fuselage_pitch_angle'
        MAX_STRUCTURAL_SPEED = 'aircraft:design:max_structural_speed'
        OPERATING_MASS = 'aircraft:design:operating_mass'
        PART25_STRUCTURAL_CATEGORY = 'aircraft:design:part25_structural_category'
        RESERVE_FUEL_ADDITIONAL = 'aircraft:design:reserve_fuel_additional'
        RESERVE_FUEL_FRACTION = 'aircraft:design:reserve_fuel_fraction'
        SMOOTH_MASS_DISCONTINUITIES = 'aircraft:design:smooth_mass_discontinuities'
        STATIC_MARGIN = 'aircraft:design:static_margin'
        STRUCTURAL_MASS_INCREMENT = 'aircraft:design:structural_mass_increment'
        STRUCTURE_MASS = 'aircraft:design:structure_mass'

        SUBSONIC_DRAG_COEFF_FACTOR = 'aircraft:design:subsonic_drag_coeff_factor'

        SUPERCRITICAL_DIVERGENCE_SHIFT = 'aircraft:design:supercritical_drag_shift'

        SUPERSONIC_DRAG_COEFF_FACTOR = 'aircraft:design:supersonic_drag_coeff_factor'

        SYSTEMS_EQUIP_MASS = 'aircraft:design:systems_equip_mass'
        SYSTEMS_EQUIP_MASS_BASE = 'aircraft:design:systems_equip_mass_base'
        THRUST_TO_WEIGHT_RATIO = 'aircraft:design:thrust_to_weight_ratio'
        TOTAL_WETTED_AREA = 'aircraft:design:total_wetted_area'
        TOUCHDOWN_MASS = 'aircraft:design:touchdown_mass'
        TYPE = 'aircraft:design:type'
        ULF_CALCULATED_FROM_MANEUVER = 'aircraft:design:ulf_calculated_from_maneuver'
        USE_ALT_MASS = 'aircraft:design:use_alt_mass'
        WETTED_AREAS = 'aircraft:design:wetted_areas'
        ZERO_FUEL_MASS = 'aircraft:design:zero_fuel_mass'
        ZERO_LIFT_DRAG_COEFF_FACTOR = 'aircraft:design:zero_lift_drag_coeff_factor'

    class Electrical:
        HAS_HYBRID_SYSTEM = 'aircraft:electrical:has_hybrid_system'
        HYBRID_CABLE_LENGTH = 'aircraft:electrical:hybrid_cable_length'
        MASS = 'aircraft:electrical:mass'
        MASS_SCALER = 'aircraft:electrical:mass_scaler'
        SYSTEM_MASS_PER_PASSENGER = 'aircraft:electrical:system_mass_per_passenger'

    class Engine:
        ADDITIONAL_MASS = 'aircraft:engine:additional_mass'
        ADDITIONAL_MASS_FRACTION = 'aircraft:engine:additional_mass_fraction'
        CONSTANT_FUEL_CONSUMPTION = 'aircraft:engine:constant_fuel_consumption'
        CONTROLS_MASS = 'aircraft:engine:controls_mass'
        DATA_FILE = 'aircraft:engine:data_file'
        FIXED_RPM = 'aircraft:engine:fixed_rpm'
        FLIGHT_IDLE_MAX_FRACTION = 'aircraft:engine:flight_idle_max_fraction'
        FLIGHT_IDLE_MIN_FRACTION = 'aircraft:engine:flight_idle_min_fraction'
        FLIGHT_IDLE_THRUST_FRACTION = 'aircraft:engine:flight_idle_thrust_fraction'
        FUEL_FLOW_SCALER_CONSTANT_TERM = 'aircraft:engine:fuel_flow_scaler_constant_term'
        FUEL_FLOW_SCALER_LINEAR_TERM = 'aircraft:engine:fuel_flow_scaler_linear_term'
        GENERATE_FLIGHT_IDLE = 'aircraft:engine:generate_flight_idle'
        GEOPOTENTIAL_ALT = 'aircraft:engine:geopotential_alt'
        GLOBAL_HYBRID_THROTTLE = 'aircraft:engine:global_hybrid_throttle'
        GLOBAL_THROTTLE = 'aircraft:engine:global_throttle'
        HAS_PROPELLERS = 'aircraft:engine:has_propellers'
        IGNORE_NEGATIVE_THRUST = 'aircraft:engine:ignore_negative_thrust'
        INTERPOLATION_METHOD = 'aircraft:engine:interpolation_method'
        MASS = 'aircraft:engine:mass'
        MASS_SCALER = 'aircraft:engine:mass_scaler'
        MASS_SPECIFIC = 'aircraft:engine:mass_specific'
        NUM_ENGINES = 'aircraft:engine:num_engines'
        NUM_FUSELAGE_ENGINES = 'aircraft:engine:num_fuselage_engines'
        NUM_WING_ENGINES = 'aircraft:engine:num_wing_engines'
        POD_MASS = 'aircraft:engine:pod_mass'
        POD_MASS_SCALER = 'aircraft:engine:pod_mass_scaler'
        POSITION_FACTOR = 'aircraft:engine:position_factor'
        PYLON_FACTOR = 'aircraft:engine:pylon_factor'
        REFERENCE_DIAMETER = 'aircraft:engine:reference_diameter'
        REFERENCE_MASS = 'aircraft:engine:reference_mass'
        REFERENCE_SLS_THRUST = 'aircraft:engine:reference_sls_thrust'
        RPM_DESIGN = 'aircraft:engine:rpm_design'
        SCALE_FACTOR = 'aircraft:engine:scale_factor'
        SCALE_MASS = 'aircraft:engine:scale_mass'
        SCALE_PERFORMANCE = 'aircraft:engine:scale_performance'
        SCALED_SLS_THRUST = 'aircraft:engine:scaled_sls_thrust'
        STARTER_MASS = 'aircraft:engine:starter_mass'
        SUBSONIC_FUEL_FLOW_SCALER = 'aircraft:engine:subsonic_fuel_flow_scaler'
        SUPERSONIC_FUEL_FLOW_SCALER = 'aircraft:engine:supersonic_fuel_flow_scaler'
        THRUST_REVERSERS_MASS = 'aircraft:engine:thrust_reversers_mass'
        THRUST_REVERSERS_MASS_SCALER = 'aircraft:engine:thrust_reversers_mass_scaler'
        TYPE = 'aircraft:engine:type'
        WING_LOCATIONS = 'aircraft:engine:wing_locations'

        class Gearbox:
            EFFICIENCY = 'aircraft:engine:gearbox:efficiency'
            GEAR_RATIO = 'aircraft:engine:gearbox:gear_ratio'
            MASS = 'aircraft:engine:gearbox:mass'
            SHAFT_POWER_DESIGN = 'aircraft:engine:gearbox:shaft_power_design'
            SPECIFIC_TORQUE = 'aircraft:engine:gearbox:specific_torque'

        class Motor:
            MASS = 'aircraft:engine:motor:mass'
            TORQUE_MAX = 'aircraft:engine:motor:torque_max'

        class Propeller:
            ACTIVITY_FACTOR = 'aircraft:engine:propeller:activity_factor'
            COMPUTE_INSTALLATION_LOSS = 'aircraft:engine:propeller:compute_installation_loss'
            DATA_FILE = 'aircraft:engine:propeller:data_file'
            DIAMETER = 'aircraft:engine:propeller:diameter'
            INTEGRATED_LIFT_COEFFICIENT = 'aircraft:engine:propeller:integrated_lift_coefficient'
            NUM_BLADES = 'aircraft:engine:propeller:num_blades'
            TIP_MACH_MAX = 'aircraft:engine:propeller:tip_mach_max'
            TIP_SPEED_MAX = 'aircraft:engine:propeller:tip_speed_max'

    class Fins:
        AREA = 'aircraft:fins:area'
        MASS = 'aircraft:fins:mass'
        MASS_SCALER = 'aircraft:fins:mass_scaler'
        NUM_FINS = 'aircraft:fins:num_fins'
        TAPER_RATIO = 'aircraft:fins:taper_ratio'

    class Fuel:
        AUXILIARY_FUEL_CAPACITY = 'aircraft:fuel:auxiliary_fuel_capacity'
        BURN_PER_PASSENGER_MILE = 'aircraft:fuel:burn_per_passenger_mile'
        CAPACITY_FACTOR = 'aircraft:fuel:capacity_factor'
        DENSITY = 'aircraft:fuel:density'
        DENSITY_RATIO = 'aircraft:fuel:density_ratio'
        FUEL_MARGIN = 'aircraft:fuel:fuel_margin'
        FUEL_SYSTEM_MASS = 'aircraft:fuel:fuel_system_mass'
        FUEL_SYSTEM_MASS_COEFFICIENT = 'aircraft:fuel:fuel_system_mass_coefficient'
        FUEL_SYSTEM_MASS_SCALER = 'aircraft:fuel:fuel_system_mass_scaler'
        FUSELAGE_FUEL_CAPACITY = 'aircraft:fuel:fuselage_fuel_capacity'
        NUM_TANKS = 'aircraft:fuel:num_tanks'
        TOTAL_CAPACITY = 'aircraft:fuel:total_capacity'
        TOTAL_VOLUME = 'aircraft:fuel:total_volume'
        UNUSABLE_FUEL_MASS = 'aircraft:fuel:unusable_fuel_mass'
        UNUSABLE_FUEL_MASS_COEFFICIENT = 'aircraft:fuel:unusable_fuel_mass_coefficient'
        UNUSABLE_FUEL_MASS_SCALER = 'aircraft:fuel:unusable_fuel_mass_scaler'
        WING_FUEL_CAPACITY = 'aircraft:fuel:wing_fuel_capacity'
        WING_FUEL_FRACTION = 'aircraft:fuel:wing_fuel_fraction'
        WING_REF_CAPACITY = 'aircraft:fuel:wing_ref_capacity'
        WING_REF_CAPACITY_AREA = 'aircraft:fuel:wing_ref_capacity_area'
        WING_REF_CAPACITY_TERM_A = 'aircraft:fuel:wing_ref_capacity_term_A'
        WING_REF_CAPACITY_TERM_B = 'aircraft:fuel:wing_ref_capacity_term_B'
        # WING_VOLUME = 'aircraft:fuel:wing_volume'
        WING_VOLUME_DESIGN = 'aircraft:fuel:wing_volume_design'
        WING_VOLUME_GEOMETRIC_MAX = 'aircraft:fuel:wing_volume_geometric_max'
        WING_VOLUME_STRUCTURAL_MAX = 'aircraft:fuel:wing_volume_structural_max'

    class Furnishings:
        MASS = 'aircraft:furnishings:mass'
        MASS_BASE = 'aircraft:furnishings:mass_base'
        MASS_SCALER = 'aircraft:furnishings:mass_scaler'
        USE_EMPIRICAL_EQUATION = 'aircraft:furnishings:use_empirical_equation'

    class Fuselage:
        AFTBODY_MASS = 'aircraft:fuselage:aftbody_mass'
        AFTBODY_MASS_PER_UNIT_AREA = 'aircraft:fuselage:aftbody_mass_per_unit_area'
        AISLE_WIDTH = 'aircraft:fuselage:aisle_width'
        AVG_DIAMETER = 'aircraft:fuselage:avg_diameter'
        CABIN_AREA = 'aircraft:fuselage:cabin_area'
        CHARACTERISTIC_LENGTH = 'aircraft:fuselage:characteristic_length'
        CROSS_SECTION = 'aircraft:fuselage:cross_section'
        DELTA_DIAMETER = 'aircraft:fuselage:delta_diameter'
        DIAMETER_TO_WING_SPAN = 'aircraft:fuselage:diameter_to_wing_span'
        FINENESS = 'aircraft:fuselage:fineness'
        FLAT_PLATE_AREA_INCREMENT = 'aircraft:fuselage:flat_plate_area_increment'
        FOREBODY_MASS = 'aircraft:fuselage:forebody_mass'
        FORM_FACTOR = 'aircraft:fuselage:form_factor'
        HEIGHT_TO_WIDTH_RATIO = 'aircraft:fuselage:height_to_width_ratio'
        HYDRAULIC_DIAMETER = 'aircraft:fuselage:hydraulic_diameter'
        LAMINAR_FLOW_LOWER = 'aircraft:fuselage:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:fuselage:laminar_flow_upper'
        LENGTH = 'aircraft:fuselage:length'
        LENGTH_TO_DIAMETER = 'aircraft:fuselage:length_to_diameter'
        LIFT_COEFFICENT_RATIO_BODY_TO_WING = 'aircraft:fuselage:lift_coefficient_ratio_body_to_wing'
        LIFT_CURVE_SLOPE_MACH0 = 'aircraft:fuselage:lift_curve_slope_mach0'
        MASS = 'aircraft:fuselage:mass'
        MASS_COEFFICIENT = 'aircraft:fuselage:mass_coefficient'
        MASS_SCALER = 'aircraft:fuselage:mass_scaler'
        MAX_HEIGHT = 'aircraft:fuselage:max_height'
        MAX_WIDTH = 'aircraft:fuselage:max_width'
        MILITARY_CARGO_FLOOR = 'aircraft:fuselage:military_cargo_floor'
        NOSE_FINENESS = 'aircraft:fuselage:nose_fineness'
        NUM_AISLES = 'aircraft:fuselage:num_aisles'
        NUM_FUSELAGES = 'aircraft:fuselage:num_fuselages'
        NUM_SEATS_ABREAST = 'aircraft:fuselage:num_seats_abreast'

        PASSENGER_COMPARTMENT_LENGTH = 'aircraft:fuselage:passenger_compartment_length'

        PILOT_COMPARTMENT_LENGTH = 'aircraft:fuselage:pilot_compartment_length'
        PLANFORM_AREA = 'aircraft:fuselage:planform_area'
        PRESSURE_DIFFERENTIAL = 'aircraft:fuselage:pressure_differential'
        PRESSURIZED_WIDTH_ADDITIONAL = 'aircraft:fuselage:pressurized_width_additional'
        SEAT_PITCH = 'aircraft:fuselage:seat_pitch'
        SEAT_WIDTH = 'aircraft:fuselage:seat_width'
        TAIL_FINENESS = 'aircraft:fuselage:tail_fineness'
        WETTED_AREA = 'aircraft:fuselage:wetted_area'
        WETTED_AREA_RATIO_AFTBODY_TO_TOTAL = 'aircraft:fuselage:wetted_area_ratio_aftbody_to_total'
        WETTED_AREA_SCALER = 'aircraft:fuselage:wetted_area_scaler'

    class HorizontalTail:
        AREA = 'aircraft:horizontal_tail:area'
        ASPECT_RATIO = 'aircraft:horizontal_tail:aspect_ratio'
        AVERAGE_CHORD = 'aircraft:horizontal_tail:average_chord'

        CHARACTERISTIC_LENGTH = 'aircraft:horizontal_tail:characteristic_length'

        FINENESS = 'aircraft:horizontal_tail:fineness'
        FORM_FACTOR = 'aircraft:horizontal_tail:form_factor'
        LAMINAR_FLOW_LOWER = 'aircraft:horizontal_tail:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:horizontal_tail:laminar_flow_upper'
        MASS = 'aircraft:horizontal_tail:mass'
        MASS_COEFFICIENT = 'aircraft:horizontal_tail:mass_coefficient'
        MASS_SCALER = 'aircraft:horizontal_tail:mass_scaler'
        MOMENT_ARM = 'aircraft:horizontal_tail:moment_arm'
        MOMENT_RATIO = 'aircraft:horizontal_tail:moment_ratio'
        ROOT_CHORD = 'aircraft:horizontal_tail:root_chord'
        SPAN = 'aircraft:horizontal_tail:span'
        SWEEP = 'aircraft:horizontal_tail:sweep'
        TAPER_RATIO = 'aircraft:horizontal_tail:taper_ratio'
        THICKNESS_TO_CHORD = 'aircraft:horizontal_tail:thickness_to_chord'

        VERTICAL_TAIL_FRACTION = 'aircraft:horizontal_tail:vertical_tail_fraction'

        VOLUME_COEFFICIENT = 'aircraft:horizontal_tail:volume_coefficient'
        WETTED_AREA = 'aircraft:horizontal_tail:wetted_area'
        WETTED_AREA_SCALER = 'aircraft:horizontal_tail:wetted_area_scaler'

    class Hydraulics:
        FLIGHT_CONTROL_MASS_COEFFICIENT = 'aircraft:hydraulics:flight_control_mass_coefficient'
        GEAR_MASS_COEFFICIENT = 'aircraft:hydraulics:gear_mass_coefficient'
        MASS = 'aircraft:hydraulics:mass'
        MASS_SCALER = 'aircraft:hydraulics:mass_scaler'
        SYSTEM_PRESSURE = 'aircraft:hydraulics:system_pressure'

    class Instruments:
        MASS = 'aircraft:instruments:mass'
        MASS_COEFFICIENT = 'aircraft:instruments:mass_coefficient'
        MASS_SCALER = 'aircraft:instruments:mass_scaler'

    class LandingGear:
        DRAG_COEFFICIENT = 'aircraft:landing_gear:drag_coefficient'
        FIXED_GEAR = 'aircraft:landing_gear:fixed_gear'
        MAIN_GEAR_LOCATION = 'aircraft:landing_gear:main_gear_location'
        MAIN_GEAR_MASS = 'aircraft:landing_gear:main_gear_mass'
        MAIN_GEAR_MASS_COEFFICIENT = 'aircraft:landing_gear:main_gear_mass_coefficient'
        MAIN_GEAR_MASS_SCALER = 'aircraft:landing_gear:main_gear_mass_scaler'
        MAIN_GEAR_OLEO_LENGTH = 'aircraft:landing_gear:main_gear_oleo_length'

        MASS_COEFFICIENT = 'aircraft:landing_gear:mass_coefficient'

        NOSE_GEAR_MASS = 'aircraft:landing_gear:nose_gear_mass'
        NOSE_GEAR_MASS_SCALER = 'aircraft:landing_gear:nose_gear_mass_scaler'
        NOSE_GEAR_OLEO_LENGTH = 'aircraft:landing_gear:nose_gear_oleo_length'

        TAIL_HOOK_MASS_SCALER = 'aircraft:landing_gear:tail_hook_mass_scaler'
        TOTAL_MASS = 'aircraft:landing_gear:total_mass'
        TOTAL_MASS_SCALER = 'aircraft:landing_gear:total_mass_scaler'

    class Nacelle:
        AVG_DIAMETER = 'aircraft:nacelle:avg_diameter'
        AVG_LENGTH = 'aircraft:nacelle:avg_length'
        CHARACTERISTIC_LENGTH = 'aircraft:nacelle:characteristic_length'
        CLEARANCE_RATIO = 'aircraft:nacelle:clearance_ratio'
        CORE_DIAMETER_RATIO = 'aircraft:nacelle:core_diameter_ratio'
        FINENESS = 'aircraft:nacelle:fineness'
        FORM_FACTOR = 'aircraft:nacelle:form_factor'
        LAMINAR_FLOW_LOWER = 'aircraft:nacelle:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:nacelle:laminar_flow_upper'
        MASS = 'aircraft:nacelle:mass'
        MASS_SCALER = 'aircraft:nacelle:mass_scaler'
        MASS_SPECIFIC = 'aircraft:nacelle:mass_specific'
        PERCENT_DIAM_BURIED_IN_FUSELAGE = 'aircraft:nacelle:percent_diam_buried_in_fuselage'
        SURFACE_AREA = 'aircraft:nacelle:surface_area'
        TOTAL_WETTED_AREA = 'aircraft:nacelle:total_wetted_area'
        WETTED_AREA = 'aircraft:nacelle:wetted_area'
        WETTED_AREA_SCALER = 'aircraft:nacelle:wetted_area_scaler'

    class Paint:
        MASS = 'aircraft:paint:mass'
        MASS_PER_UNIT_AREA = 'aircraft:paint:mass_per_unit_area'

    class Propulsion:
        ENGINE_OIL_MASS_SCALER = 'aircraft:propulsion:engine_oil_mass_scaler'

        MASS = 'aircraft:propulsion:mass'
        MISC_MASS_SCALER = 'aircraft:propulsion:misc_mass_scaler'
        TOTAL_ENGINE_CONTROLS_MASS = 'aircraft:propulsion:total_engine_controls_mass'
        TOTAL_ENGINE_MASS = 'aircraft:propulsion:total_engine_mass'
        TOTAL_ENGINE_OIL_MASS = 'aircraft:propulsion:total_engine_oil_mass'
        TOTAL_ENGINE_POD_MASS = 'aircraft:propulsion:total_engine_pod_mass'
        TOTAL_MISC_MASS = 'aircraft:propulsion:total_misc_mass'
        TOTAL_NUM_ENGINES = 'aircraft:propulsion:total_num_engines'
        TOTAL_NUM_FUSELAGE_ENGINES = 'aircraft:propulsion:total_num_fuselage_engines'
        TOTAL_NUM_WING_ENGINES = 'aircraft:propulsion:total_num_wing_engines'
        TOTAL_REFERENCE_SLS_THRUST = 'aircraft:propulsion:total_reference_sls_thrust'
        TOTAL_SCALED_SLS_THRUST = 'aircraft:propulsion:total_scaled_sls_thrust'
        TOTAL_STARTER_MASS = 'aircraft:propulsion:total_starter_mass'

        TOTAL_THRUST_REVERSERS_MASS = 'aircraft:propulsion:total_thrust_reversers_mass'

    class Strut:
        AREA = 'aircraft:strut:area'
        AREA_RATIO = 'aircraft:strut:area_ratio'
        ATTACHMENT_LOCATION = 'aircraft:strut:attachment_location'
        ATTACHMENT_LOCATION_DIMENSIONLESS = 'aircraft:strut:attachment_location_dimensionless'
        CHORD = 'aircraft:strut:chord'
        DIMENSIONAL_LOCATION_SPECIFIED = 'aircraft:strut:dimensional_location_specified'
        FUSELAGE_INTERFERENCE_FACTOR = 'aircraft:strut:fuselage_interference_factor'
        LENGTH = 'aircraft:strut:length'
        MASS = 'aircraft:strut:mass'
        MASS_COEFFICIENT = 'aircraft:strut:mass_coefficient'
        THICKNESS_TO_CHORD = 'aircraft:strut:thickness_to_chord'

    class TailBoom:
        LENGTH = 'aircraft:tail_boom:length'

    class VerticalTail:
        AREA = 'aircraft:vertical_tail:area'
        ASPECT_RATIO = 'aircraft:vertical_tail:aspect_ratio'
        AVERAGE_CHORD = 'aircraft:vertical_tail:average_chord'
        CHARACTERISTIC_LENGTH = 'aircraft:vertical_tail:characteristic_length'
        FINENESS = 'aircraft:vertical_tail:fineness'
        FORM_FACTOR = 'aircraft:vertical_tail:form_factor'
        LAMINAR_FLOW_LOWER = 'aircraft:vertical_tail:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:vertical_tail:laminar_flow_upper'
        MASS = 'aircraft:vertical_tail:mass'
        MASS_COEFFICIENT = 'aircraft:vertical_tail:mass_coefficient'
        MASS_SCALER = 'aircraft:vertical_tail:mass_scaler'
        MOMENT_ARM = 'aircraft:vertical_tail:moment_arm'
        MOMENT_RATIO = 'aircraft:vertical_tail:moment_ratio'
        NUM_TAILS = 'aircraft:vertical_tail:num_tails'
        ROOT_CHORD = 'aircraft:vertical_tail:root_chord'
        SPAN = 'aircraft:vertical_tail:span'
        SWEEP = 'aircraft:vertical_tail:sweep'
        TAPER_RATIO = 'aircraft:vertical_tail:taper_ratio'
        THICKNESS_TO_CHORD = 'aircraft:vertical_tail:thickness_to_chord'
        VOLUME_COEFFICIENT = 'aircraft:vertical_tail:volume_coefficient'
        WETTED_AREA = 'aircraft:vertical_tail:wetted_area'
        WETTED_AREA_SCALER = 'aircraft:vertical_tail:wetted_area_scaler'

    class Wing:
        AEROELASTIC_TAILORING_FACTOR = 'aircraft:wing:aeroelastic_tailoring_factor'

        AIRFOIL_TECHNOLOGY = 'aircraft:wing:airfoil_technology'
        AREA = 'aircraft:wing:area'
        ASPECT_RATIO = 'aircraft:wing:aspect_ratio'
        ASPECT_RATIO_REF = 'aircraft:wing:aspect_ratio_reference'
        AVERAGE_CHORD = 'aircraft:wing:average_chord'
        BENDING_MATERIAL_FACTOR = 'aircraft:wing:bending_material_factor'
        BENDING_MATERIAL_MASS = 'aircraft:wing:bending_material_mass'
        # Not defined in metadata!
        # BENDING_MASS_NO_INERTIA = 'aircraft:wing:bending_mass_no_inertia'
        BENDING_MATERIAL_MASS_SCALER = 'aircraft:wing:bending_material_mass_scaler'
        BWB_AFTBODY_MASS = 'aircraft:wing:bwb_aft_body_mass'
        BWB_AFTBODY_MASS_SCALER = 'aircraft:wing:bwb_aft_body_mass_scaler'
        CENTER_CHORD = 'aircraft:wing:center_chord'
        CENTER_DISTANCE = 'aircraft:wing:center_distance'
        CHARACTERISTIC_LENGTH = 'aircraft:wing:characteristic_length'
        CHOOSE_FOLD_LOCATION = 'aircraft:wing:choose_fold_location'
        CHORD_PER_SEMISPAN_DIST = 'aircraft:wing:chord_per_semispan'
        COMPOSITE_FRACTION = 'aircraft:wing:composite_fraction'
        CONTROL_SURFACE_AREA = 'aircraft:wing:control_surface_area'
        CONTROL_SURFACE_AREA_RATIO = 'aircraft:wing:control_surface_area_ratio'
        DIHEDRAL = 'aircraft:wing:dihedral'
        ENG_POD_INERTIA_FACTOR = 'aircraft:wing:eng_pod_inertia_factor'
        EXPOSED_AREA = 'aircraft:wing:exposed_area'
        FINENESS = 'aircraft:wing:fineness'
        FLAP_CHORD_RATIO = 'aircraft:wing:flap_chord_ratio'
        FLAP_DEFLECTION_LANDING = 'aircraft:wing:flap_deflection_landing'
        FLAP_DEFLECTION_TAKEOFF = 'aircraft:wing:flap_deflection_takeoff'
        FLAP_DRAG_INCREMENT_OPTIMUM = 'aircraft:wing:flap_drag_increment_optimum'
        FLAP_LIFT_INCREMENT_OPTIMUM = 'aircraft:wing:flap_lift_increment_optimum'
        FLAP_SPAN_RATIO = 'aircraft:wing:flap_span_ratio'
        FLAP_TYPE = 'aircraft:wing:flap_type'
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = 'aircraft:wing:fold_dimensional_location_specified'
        FOLD_MASS = 'aircraft:wing:fold_mass'
        FOLD_MASS_COEFFICIENT = 'aircraft:wing:fold_mass_coefficient'
        FOLDED_SPAN = 'aircraft:wing:folded_span'
        FOLDED_SPAN_DIMENSIONLESS = 'aircraft:wing:folded_span_dimensionless'
        FOLDING_AREA = 'aircraft:wing:folding_area'
        FORM_FACTOR = 'aircraft:wing:form_factor'
        FUSELAGE_INTERFERENCE_FACTOR = 'aircraft:wing:fuselage_interference_factor'
        GLOVE_AND_BAT = 'aircraft:wing:glove_and_bat'
        HAS_FOLD = 'aircraft:wing:has_fold'
        HAS_STRUT = 'aircraft:wing:has_strut'
        HEIGHT = 'aircraft:wing:height'
        HIGH_LIFT_MASS = 'aircraft:wing:high_lift_mass'
        HIGH_LIFT_MASS_COEFFICIENT = 'aircraft:wing:high_lift_mass_coefficient'
        INCIDENCE = 'aircraft:wing:incidence'
        INPUT_STATION_DIST = 'aircraft:wing:input_station_dist'
        LAMINAR_FLOW_LOWER = 'aircraft:wing:laminar_flow_lower'
        LAMINAR_FLOW_UPPER = 'aircraft:wing:laminar_flow_upper'
        LEADING_EDGE_SWEEP = 'aircraft:wing:leading_edge_sweep'
        LOAD_DISTRIBUTION_CONTROL = 'aircraft:wing:load_distribution_control'
        LOAD_FRACTION = 'aircraft:wing:load_fraction'
        LOAD_PATH_SWEEP_DIST = 'aircraft:wing:load_path_sweep_dist'
        LOADING = 'aircraft:wing:loading'
        LOADING_ABOVE_20 = 'aircraft:wing:loading_above_20'
        MASS = 'aircraft:wing:mass'
        MASS_COEFFICIENT = 'aircraft:wing:mass_coefficient'
        MASS_SCALER = 'aircraft:wing:mass_scaler'
        MATERIAL_FACTOR = 'aircraft:wing:material_factor'
        MAX_CAMBER_AT_70_SEMISPAN = 'aircraft:wing:max_camber_at_70_semispan'
        MAX_LIFT_REF = 'aircraft:wing:max_lift_ref'
        MAX_SLAT_DEFLECTION_LANDING = 'aircraft:wing:max_slat_deflection_landing'
        MAX_SLAT_DEFLECTION_TAKEOFF = 'aircraft:wing:max_slat_deflection_takeoff'
        MAX_THICKNESS_LOCATION = 'aircraft:wing:max_thickness_location'
        MIN_PRESSURE_LOCATION = 'aircraft:wing:min_pressure_location'
        MISC_MASS = 'aircraft:wing:misc_mass'
        MISC_MASS_SCALER = 'aircraft:wing:misc_mass_scaler'
        NUM_FLAP_SEGMENTS = 'aircraft:wing:num_flap_segments'
        NUM_INTEGRATION_STATIONS = 'aircraft:wing:num_integration_stations'
        OPTIMUM_FLAP_DEFLECTION = 'aircraft:wing:optimum_flap_deflection'
        OPTIMUM_SLAT_DEFLECTION = 'aircraft:wing:optimum_slat_deflection'
        ROOT_CHORD = 'aircraft:wing:root_chord'
        SHEAR_CONTROL_MASS = 'aircraft:wing:shear_control_mass'

        SHEAR_CONTROL_MASS_SCALER = 'aircraft:wing:shear_control_mass_scaler'

        SLAT_CHORD_RATIO = 'aircraft:wing:slat_chord_ratio'
        SLAT_LIFT_INCREMENT_OPTIMUM = 'aircraft:wing:slat_lift_increment_optimum'
        SLAT_SPAN_RATIO = 'aircraft:wing:slat_span_ratio'
        SPAN = 'aircraft:wing:span'
        SPAN_EFFICIENCY_FACTOR = 'aircraft:wing:span_efficiency_factor'
        SPAN_EFFICIENCY_REDUCTION = 'aircraft:wing:span_efficiency_reduction'
        STRUT_BRACING_FACTOR = 'aircraft:wing:strut_bracing_factor'
        SURFACE_CONTROL_MASS = 'aircraft:wing:surface_ctrl_mass'
        SURFACE_CONTROL_MASS_COEFFICIENT = 'aircraft:wing:surface_ctrl_mass_coefficient'

        SURFACE_CONTROL_MASS_SCALER = 'aircraft:wing:surface_ctrl_mass_scaler'

        SWEEP = 'aircraft:wing:sweep'
        TAPER_RATIO = 'aircraft:wing:taper_ratio'
        THICKNESS_TO_CHORD = 'aircraft:wing:thickness_to_chord'
        THICKNESS_TO_CHORD_DIST = 'aircraft:wing:thickness_to_chord_dist'
        THICKNESS_TO_CHORD_REF = 'aircraft:wing:thickness_to_chord_reference'
        THICKNESS_TO_CHORD_ROOT = 'aircraft:wing:thickness_to_chord_root'
        THICKNESS_TO_CHORD_TIP = 'aircraft:wing:thickness_to_chord_tip'
        THICKNESS_TO_CHORD_UNWEIGHTED = 'aircraft:wing:thickness_to_chord_unweighted'
        ULTIMATE_LOAD_FACTOR = 'aircraft:wing:ultimate_load_factor'
        USE_DETAILED_MASS = 'aircraft:wing:use_detailed_mass'
        VAR_SWEEP_MASS_PENALTY = 'aircraft:wing:var_sweep_mass_penalty'
        VERTICAL_MOUNT_LOCATION = 'aircraft:wing:vertical_mount_location'
        WETTED_AREA = 'aircraft:wing:wetted_area'
        WETTED_AREA_SCALER = 'aircraft:wing:wetted_area_scaler'
        ZERO_LIFT_ANGLE = 'aircraft:wing:zero_lift_angle'


class Dynamic:
    """All time-dependent variables used during mission analysis."""

    class Atmosphere:
        """Atmospheric and freestream conditions."""

        DENSITY = 'density'
        DYNAMIC_PRESSURE = 'dynamic_pressure'
        KINEMATIC_VISCOSITY = 'kinematic_viscosity'
        MACH = 'mach'
        MACH_RATE = 'mach_rate'
        SPEED_OF_SOUND = 'speed_of_sound'
        STATIC_PRESSURE = 'static_pressure'
        TEMPERATURE = 'temperature'

    class Mission:
        """
        Kinematic description of vehicle states in a ground-fixed axis.
        These values are typically used by the Equations of Motion to determine
        vehicle states at other timesteps.
        """

        # TODO Vehicle summary forces, torques, etc. in X,Y,Z axes should also go here
        ALTITUDE = 'altitude'
        ALTITUDE_RATE = 'altitude_rate'
        ALTITUDE_RATE_MAX = 'altitude_rate_max'
        # TODO Angle of Attack
        DISTANCE = 'distance'
        DISTANCE_RATE = 'distance_rate'
        FLIGHT_PATH_ANGLE = 'flight_path_angle'
        FLIGHT_PATH_ANGLE_RATE = 'flight_path_angle_rate'
        SPECIFIC_ENERGY = 'specific_energy'
        SPECIFIC_ENERGY_RATE = 'specific_energy_rate'
        SPECIFIC_ENERGY_RATE_EXCESS = 'specific_energy_rate_excess'
        VELOCITY = 'velocity'
        VELOCITY_RATE = 'velocity_rate'

    class Vehicle:
        """Vehicle properties and states in a vehicle-fixed reference frame."""

        ANGLE_OF_ATTACK = 'angle_of_attack'
        BATTERY_STATE_OF_CHARGE = 'battery_state_of_charge'
        CUMULATIVE_ELECTRIC_ENERGY_USED = 'cumulative_electric_energy_used'
        DRAG = 'drag'
        LIFT = 'lift'
        MASS = 'mass'
        MASS_RATE = 'mass_rate'

        class Propulsion:
            # variables specific to the propulsion subsystem
            ELECTRIC_POWER_IN = 'electric_power_in'
            ELECTRIC_POWER_IN_TOTAL = 'electric_power_in_total'
            # EXIT_AREA = 'exit_area'
            FUEL_FLOW_RATE = 'fuel_flow_rate'
            FUEL_FLOW_RATE_NEGATIVE = 'fuel_flow_rate_negative'
            FUEL_FLOW_RATE_NEGATIVE_TOTAL = 'fuel_flow_rate_negative_total'
            FUEL_FLOW_RATE_TOTAL = 'fuel_flow_rate_total'
            HYBRID_THROTTLE = 'hybrid_throttle'
            NOX_RATE = 'nox_rate'
            NOX_RATE_TOTAL = 'nox_rate_total'
            PROPELLER_TIP_SPEED = 'propeller_tip_speed'
            RPM = 'rotations_per_minute'
            SHAFT_POWER = 'shaft_power'
            SHAFT_POWER_MAX = 'shaft_power_max'
            TEMPERATURE_T4 = 't4'
            THROTTLE = 'throttle'
            THRUST = 'thrust_net'
            THRUST_MAX = 'thrust_net_max'
            THRUST_MAX_TOTAL = 'thrust_net_max_total'
            THRUST_TOTAL = 'thrust_net_total'
            TORQUE = 'torque'
            TORQUE_MAX = 'torque_max'


class Mission:
    """Mission data hierarchy."""

    class Constraints:
        # these can be residuals (for equality constraints),
        # upper bounds, or lower bounds
        GEARBOX_SHAFT_POWER_RESIDUAL = 'mission:constraints:gearbox_shaft_power_residual'
        MASS_RESIDUAL = 'mission:constraints:mass_residual'
        MAX_MACH = 'mission:constraints:max_mach'
        RANGE_RESIDUAL = 'mission:constraints:range_residual'
        RANGE_RESIDUAL_RESERVE = 'mission:constraints:range_residual_reserve'

    class Design:
        # These values MAY change in design mission, but in off-design
        # they cannot change. In a design mission these are either user inputs
        # or calculated outputs, in off-design they are strictly inputs
        # and do not change.
        CRUISE_ALTITUDE = 'mission:design:cruise_altitude'
        CRUISE_RANGE = 'mission:design:cruise_range'
        FUEL_MASS = 'mission:design:fuel_mass'
        FUEL_MASS_REQUIRED = 'mission:design:fuel_mass_required'
        GROSS_MASS = 'mission:design:gross_mass'
        LIFT_COEFFICIENT = 'mission:design:lift_coefficient'
        LIFT_COEFFICIENT_MAX_FLAPS_UP = 'mission:design:lift_coefficient_max_flaps_up'
        MACH = 'mission:design:mach'
        RANGE = 'mission:design:range'
        RATE_OF_CLIMB_AT_TOP_OF_CLIMB = 'mission:design:rate_of_climb_at_top_of_climb'
        RESERVE_FUEL = 'mission:design:reserve_fuel'
        THRUST_TAKEOFF_PER_ENG = 'mission:design:thrust_takeoff_per_eng'

    class Landing:
        # These are values which have to do with landing
        AIRPORT_ALTITUDE = 'mission:landing:airport_altitude'
        BRAKING_DELAY = 'mission:landing:braking_delay'
        BRAKING_FRICTION_COEFFICIENT = 'mission:landing:braking_friction_coefficient'

        DRAG_COEFFICIENT_FLAP_INCREMENT = 'mission:landing:drag_coefficient_flap_increment'
        DRAG_COEFFICIENT_MIN = 'mission:landing:drag_coefficient_min'

        FIELD_LENGTH = 'mission:landing:field_length'
        FLARE_RATE = 'mission:landing:flare_rate'
        GLIDE_TO_STALL_RATIO = 'mission:landing:glide_to_stall_ratio'
        GROUND_DISTANCE = 'mission:landing:ground_distance'
        INITIAL_ALTITUDE = 'mission:landing:initial_altitude'
        INITIAL_MACH = 'mission:landing:initial_mach'
        INITIAL_VELOCITY = 'mission:landing:initial_velocity'

        LIFT_COEFFICIENT_FLAP_INCREMENT = 'mission:landing:lift_coefficient_flap_increment'

        LIFT_COEFFICIENT_MAX = 'mission:landing:lift_coefficient_max'
        MAXIMUM_FLARE_LOAD_FACTOR = 'mission:landing:maximum_flare_load_factor'
        MAXIMUM_SINK_RATE = 'mission:landing:maximum_sink_rate'
        OBSTACLE_HEIGHT = 'mission:landing:obstacle_height'
        ROLLING_FRICTION_COEFFICIENT = 'mission:landing:rolling_friction_coefficient'
        SPOILER_DRAG_COEFFICIENT = 'mission:landing:spoiler_drag_coefficient'
        SPOILER_LIFT_COEFFICIENT = 'mission:landing:spoiler_lift_coefficient'
        STALL_VELOCITY = 'mission:landing:stall_velocity'
        TOUCHDOWN_MASS = 'mission:landing:touchdown_mass'
        TOUCHDOWN_SINK_RATE = 'mission:landing:touchdown_sink_rate'

    class Objectives:
        # these are values that can be fed to the optimizer as objectives,
        # they may often be composite and/or regularized
        FUEL = 'mission:objectives:fuel'
        RANGE = 'mission:objectives:range'

    class Summary:
        # These values are inputs and outputs to/from mission analysis
        # for the given mission (whether it is design or off-design).
        # In on-design these may be constrained to design values, but
        # in off-design they independently represent the final analysis
        # based on the user-selection.
        CRUISE_MACH = 'mission:summary:cruise_mach'
        CRUISE_MASS_FINAL = 'mission:summary:cruise_mass_final'
        FUEL_BURNED = 'mission:summary:fuel_burned'
        FUEL_FLOW_SCALER = 'mission:summary:fuel_flow_scaler'
        GROSS_MASS = 'mission:summary:gross_mass'
        RANGE = 'mission:summary:range'
        RESERVE_FUEL_BURNED = 'mission:summary:reserve_fuel_burned'
        TOTAL_FUEL_MASS = 'mission:summary:total_fuel_mass'

    class Takeoff:
        # These are values which have to do with takeoff
        AIRPORT_ALTITUDE = 'mission:takeoff:airport_altitude'
        ANGLE_OF_ATTACK_RUNWAY = 'mission:takeoff:angle_of_attack_runway'
        ASCENT_DURATION = 'mission:takeoff:ascent_duration'
        ASCENT_T_INITIAL = 'mission:takeoff:ascent_t_initial'
        BRAKING_FRICTION_COEFFICIENT = 'mission:takeoff:braking_friction_coefficient'
        DECISION_SPEED_INCREMENT = 'mission:takeoff:decision_speed_increment'

        DRAG_COEFFICIENT_FLAP_INCREMENT = 'mission:takeoff:drag_coefficient_flap_increment'

        DRAG_COEFFICIENT_MIN = 'mission:takeoff:drag_coefficient_min'
        FIELD_LENGTH = 'mission:takeoff:field_length'
        FINAL_ALTITUDE = 'mission:takeoff:final_altitude'
        FINAL_MACH = 'mission:takeoff:final_mach'
        FINAL_MASS = 'mission:takeoff:final_mass'
        FINAL_VELOCITY = 'mission:takeoff:final_velocity'
        FUEL_SIMPLE = 'mission:takeoff:fuel_simple'
        GROUND_DISTANCE = 'mission:takeoff:ground_distance'

        LIFT_COEFFICIENT_FLAP_INCREMENT = 'mission:takeoff:lift_coefficient_flap_increment'

        LIFT_COEFFICIENT_MAX = 'mission:takeoff:lift_coefficient_max'
        LIFT_OVER_DRAG = 'mission:takeoff:lift_over_drag'
        OBSTACLE_HEIGHT = 'mission:takeoff:obstacle_height'
        ROLLING_FRICTION_COEFFICIENT = 'mission:takeoff:rolling_friction_coefficient'
        ROTATION_SPEED_INCREMENT = 'mission:takeoff:rotation_speed_increment'
        ROTATION_VELOCITY = 'mission:takeoff:rotation_velocity'
        SPOILER_DRAG_COEFFICIENT = 'mission:takeoff:spoiler_drag_coefficient'
        SPOILER_LIFT_COEFFICIENT = 'mission:takeoff:spoiler_lift_coefficient'
        THRUST_INCIDENCE = 'mission:takeoff:thrust_incidence'

    class Taxi:
        DURATION = 'mission:taxi:duration'
        MACH = 'mission:taxi:mach'


class Settings:
    """Setting data hierarchy."""

    AERODYNAMICS_METHOD = 'settings:aerodynamics_method'
    EQUATIONS_OF_MOTION = 'settings:equations_of_motion'
    MASS_METHOD = 'settings:mass_method'
    PROBLEM_TYPE = 'settings:problem_type'
    VERBOSITY = 'settings:verbosity'
