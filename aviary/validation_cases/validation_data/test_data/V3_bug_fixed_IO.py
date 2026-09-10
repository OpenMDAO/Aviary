"""
Initial cut at a centralized file for the IO
of the large single aisle 1 V3 bug fixed case for the GASP-based
mass calculations and geometry calculations.
"""

import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Mission, Settings
from aviary.variable_info.enums import (
    AircraftTypes,
    AtmosphereModel,
    FlapType,
    GASPEngineType,
    Verbosity,
)


def local_get_option_defaults() -> AviaryValues:
    """Here are the variables missing in the rest of V3_bug_fixed_options"""
    option_defaults = AviaryValues()

    option_defaults.set_val(Aircraft.BWB.DETAILED_WING_PROVIDED, True, units='unitless')
    option_defaults.set_val(Aircraft.BWB.MAX_BAY_WIDTH, 0.0, units='ft')
    option_defaults.set_val(Aircraft.BWB.MAX_NUM_BAYS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 0, units='unitless')
    option_defaults.set_val(
        Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 5, units='unitless'
    )
    option_defaults.set_val(
        Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 4, units='unitless'
    )
    option_defaults.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.NUM_CABIN_CREW, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, 0, units='unitless')
    option_defaults.set_val(Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER, 0.0, units='lbm')
    option_defaults.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, False, units='unitless')
    option_defaults.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, False, units='unitless')
    option_defaults.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, 3, units='unitless')
    option_defaults.set_val(Aircraft.Design.PERCENT_EXCRESCENCE_DRAG, 0.0, units='unitless')
    option_defaults.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, False, units='unitless')
    option_defaults.set_val(Aircraft.Design.TYPE, AircraftTypes.TRANSPORT, units='unitless')
    option_defaults.set_val(Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER, False, units='unitless')
    option_defaults.set_val(Aircraft.Design.USE_ALT_MASS, False, units='unitless')
    option_defaults.set_val(Aircraft.Engine.CONSTANT_FUEL_MASS_CONSUMPTION, [0.0], units='lbm/h')
    option_defaults.set_val(
        Aircraft.Engine.DATA_FILE,
        ['/home0/xjiang/workspace/Aviary_XJ/om-Aviary/aviary/models/engines/turbofan_23k_1.csv'],
        units='unitless',
    )
    option_defaults.set_val(Aircraft.Engine.FIXED_RPM, [1.0], units='rpm')
    option_defaults.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, [1.0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, [0.08], units='unitless')
    option_defaults.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, [0.0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, [0.0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, [0.0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, [False], units='unitless')
    option_defaults.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, [False], units='unitless')
    option_defaults.set_val(Aircraft.Engine.GLOBAL_HYBRID_THROTTLE, [False], units='unitless')
    option_defaults.set_val(Aircraft.Engine.GLOBAL_THROTTLE, [False], units='unitless')
    option_defaults.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, [False], units='unitless')
    option_defaults.set_val(Aircraft.Engine.INTERPOLATION_METHOD, ['slinear'], units='unitless')
    option_defaults.set_val(Aircraft.Engine.INTERPOLATION_SORT, ['mach'], units='unitless')
    option_defaults.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]), units='unitless')
    option_defaults.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, [0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.NUM_WING_ENGINES, [2], units='unitless')
    option_defaults.set_val(
        Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS, [True], units='unitless'
    )
    option_defaults.set_val(Aircraft.Engine.Propeller.NUM_BLADES, [0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.RPM_DESIGN, [0], units='rpm')
    option_defaults.set_val(Aircraft.Engine.SCALE_MASS, [True], units='unitless')
    option_defaults.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, [1.0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, [1.0], units='unitless')
    option_defaults.set_val(Aircraft.Engine.TYPE, [GASPEngineType.TURBOJET], units='unitless')
    option_defaults.set_val(Aircraft.Fins.NUM_FINS, 0, units='unitless')
    option_defaults.set_val(Aircraft.Fuel.NUM_TANKS, 7, units='unitless')
    option_defaults.set_val(Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, True, units='unitless')
    option_defaults.set_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR, False, units='unitless')
    option_defaults.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1, units='unitless')
    option_defaults.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, True, units='unitless')
    option_defaults.set_val(Aircraft.HorizontalTail.NUM_TAILS, 1, units='unitless')
    option_defaults.set_val(Aircraft.LandingGear.DRAG_COEFFICIENT, 1.0, units='unitless')
    option_defaults.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
    option_defaults.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 0, units='unitless')
    option_defaults.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 2, units='unitless')
    option_defaults.set_val(Aircraft.Propulsion.TOTAL_REFERENCE_SLS_THRUST, 0.0, units='lbf')
    option_defaults.set_val(Aircraft.VerticalTail.NUM_TAILS, 1, units='unitless')
    option_defaults.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 1.0, units='unitless')
    option_defaults.set_val(Aircraft.Wing.DETAILED_WING, False, units='unitless')
    option_defaults.set_val(Aircraft.Wing.FLAP_TYPE, FlapType.DOUBLE_SLOTTED, units='unitless')
    option_defaults.set_val(Aircraft.Wing.HAS_FOLD, False, units='unitless')
    option_defaults.set_val(Aircraft.Wing.HAS_STRUT, False, units='unitless')
    option_defaults.set_val(
        Aircraft.Wing.INPUT_STATION_DISTRIBUTION, np.array([0.0]), units='unitless'
    )
    option_defaults.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0, units='unitless')
    option_defaults.set_val(Aircraft.Wing.LOADING_ABOVE_20, True, units='unitless')
    option_defaults.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, 2, units='unitless')
    option_defaults.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 50, units='unitless')
    option_defaults.set_val(Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, False, units='unitless')
    option_defaults.set_val(Aircraft.Design.MAX_MACH, 0.0, units='unitless')
    option_defaults.set_val(Mission.GRAVITY, 9.80665, units='m/s**2')
    option_defaults.set_val(Mission.RESERVE_FUEL_MARGIN, 0.0, units='unitless')
    option_defaults.set_val(Mission.RESERVE_FUEL_MASS_ADDITIONAL, 0.0, units='lbm')
    option_defaults.set_val(Mission.SEA_LEVEL_DENSITY, 1.225, units='kg/m**3')
    option_defaults.set_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 0.0, units='deg')
    option_defaults.set_val(Mission.Takeoff.OBSTACLE_HEIGHT, 35.0, units='ft')
    option_defaults.set_val(Mission.Takeoff.THRUST_INCIDENCE, 0.0, units='deg')
    option_defaults.set_val(Mission.Taxi.DURATION, 0.167, units='h')
    option_defaults.set_val(Settings.ATMOSPHERE_MODEL, AtmosphereModel.STANDARD, units='unitless')
    option_defaults.set_val(Settings.PAYLOAD_RANGE, False, units='unitless')
    option_defaults.set_val(Settings.VERBOSITY, Verbosity.BRIEF, units='unitless')

    return option_defaults


V3_bug_fixed_options = local_get_option_defaults()
V3_bug_fixed_options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
# we keep CrewPayload.NUM_PASSENGERS here because preprocess_crewpayload is often not run in these
# tests which prevents these values being assigned from Design.NUM_PASSENGERS as would normally happen
V3_bug_fixed_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Design.CRUISE_ALTITUDE, val=37500, units='ft')
V3_bug_fixed_options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
)
V3_bug_fixed_options.set_val(
    Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')


V3_bug_fixed_options.set_val(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Wing.SWEEP, val=25, units='deg')
V3_bug_fixed_options.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.Design.WING_LOADING, val=126, units='lbf/ft**2')
V3_bug_fixed_options.set_val(
    Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION, val=0, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
V3_bug_fixed_options.set_val(Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, np.array([28690]), units='lbf')
# NOTE override required for mass summation test
V3_bug_fixed_options.set_val(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')


V3_bug_fixed_non_metadata = AviaryValues()


V3_bug_fixed_options.set_val(Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi')
V3_bug_fixed_options.set_val(
    Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.NUM_AISLES, 1, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 6, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.WETTED_AREA, 4000, units='ft**2')
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.INLET_AREA_COEFFICIENT, 0.00030975, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Nacelle.FINENESS, 2, units='unitless')

V3_bug_fixed_options.set_val(Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h')
V3_bug_fixed_options.set_val(Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units='1/rad')
V3_bug_fixed_options.set_val(
    Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, val=200, units='lbm'
)
V3_bug_fixed_options.set_val(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
V3_bug_fixed_options.set_val(Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless')
V3_bug_fixed_options.set_val(
    # TODO: should be lbf/ft**2 I think
    Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
    val=2.66,
    units='unitless',
)
V3_bug_fixed_options.set_val(
    Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
)
V3_bug_fixed_options.set_val(
    Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
)
V3_bug_fixed_options.set_val(
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
)
V3_bug_fixed_options.set_val(Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, val=0.85, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
V3_bug_fixed_options.set_val(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
V3_bug_fixed_options.set_val(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, val=0.14, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.APU.MASS, val=928.0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
V3_bug_fixed_options.set_val(
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
)
V3_bug_fixed_options.set_val(Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm')
V3_bug_fixed_options.set_val(
    Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
)
V3_bug_fixed_options.set_val(
    Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
)

V3_bug_fixed_options.set_val(Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless')

V3_bug_fixed_options.set_val(Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm')
V3_bug_fixed_options.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
V3_bug_fixed_options.set_val(
    Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
)  # Based on large single aisle 1 for updated flaps mass model
V3_bug_fixed_options.set_val(
    Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
)  # Based on large single aisle 1 for updated flaps mass model
V3_bug_fixed_options.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless')
V3_bug_fixed_options.set_val(
    Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
)
V3_bug_fixed_options.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3, units='unitless')

V3_bug_fixed_non_metadata.set_val('pylon_len', val=0, units='ft')
V3_bug_fixed_non_metadata.set_val('MAT', val=0, units='lbm')
