"""
FLOPS derived input/output for use with Aviary unit tests and benchmarks
- FLOPS title: "REF MDL N3CC (26616) AR11 1220t 1340p turbofan_22k M785 20210721"
- FLOPS input file: "N3CC FLOPS In- generic low speed polars.txt"
- FLOPS output file: "N3CC FLOPS Out -low speed polar input echo deleted.txt"
- FLOPS engine deck: "turbofan_22k.csv".
"""

import numpy as np

from aviary.mission.flops_based.phases.detailed_landing_phases import (
    LandingApproachToMicP3,
    LandingFlareToTouchdown,
    LandingMicP3ToObstacle,
    LandingNoseDownToStop,
    LandingObstacleToFlare,
    LandingTouchdownToNoseDown,
    LandingTrajectory,
)
from aviary.mission.flops_based.phases.detailed_takeoff_phases import (
    TakeoffBrakeReleaseToDecisionSpeed,
    TakeoffBrakeToAbort,
    TakeoffDecisionSpeedBrakeDelay,
    TakeoffDecisionSpeedToRotate,
    TakeoffEngineCutback,
    TakeoffEngineCutbackToMicP1,
    TakeoffLiftoffToObstacle,
    TakeoffMicP1ToClimb,
    TakeoffMicP2ToEngineCutback,
    TakeoffObstacleToMicP2,
    TakeoffRotateToLiftoff,
    TakeoffTrajectory,
)
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.test_utils.default_subsystems import (
    get_default_mission_subsystems,
    get_default_premission_subsystems,
)
from aviary.variable_info.enums import EquationsOfMotion, LegacyCode, ProblemType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

N3CC = {}
inputs = N3CC['inputs'] = AviaryValues()
outputs = N3CC['outputs'] = AviaryValues()

# Overall Aircraft
# ---------------------------
inputs.set_val(Aircraft.Design.BASE_AREA, 0.0, 'ft**2')
inputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN_SCALER, 0.01498)
inputs.set_val(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 0.84)
inputs.set_val(Mission.Design.GROSS_MASS, 129734.0, 'lbm')
inputs.set_val(Aircraft.Design.USE_ALT_MASS, False)
inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 0.93)
inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 0.95)
inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)
inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 0.96)

# Air Conditioning
# ---------------------------
inputs.set_val(Aircraft.AirConditioning.MASS_SCALER, 0.98094)

# Anti-Icing
# ---------------------------
inputs.set_val(Aircraft.AntiIcing.MASS_SCALER, 0.53202)

# APU
# ---------------------------
inputs.set_val(Aircraft.APU.MASS_SCALER, 1.02321)

# Avionics
# ---------------------------
inputs.set_val(Aircraft.Avionics.MASS_SCALER, 1.123226)

# Canard
# ---------------------------

# Crew and Payload
# ---------------------------
inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 20)
inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 16)
inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 154, units='unitless')
inputs.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 118)
inputs.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 20)
inputs.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 16)
inputs.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 154, units='unitless')
inputs.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, 118)

inputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 35.0, 'lbm')
inputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, 0.0)
inputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.MISC_CARGO, 0.0, 'lbm')
inputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 165.0, 'lbm')
inputs.set_val(Aircraft.CrewPayload.WING_CARGO, 0.0, 'lbm')

# Electrical
# ---------------------------
inputs.set_val(Aircraft.Electrical.MASS_SCALER, 1.1976)

# Fins
# ---------------------------
inputs.set_val(Aircraft.Fins.NUM_FINS, 0)

# Fuel
# ---------------------------
inputs.set_val(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0, 'lbm')
inputs.set_val(Aircraft.Fuel.CAPACITY_FACTOR, 25.903)
inputs.set_val(Aircraft.Fuel.DENSITY_RATIO, 1.0)
inputs.set_val(Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0, 'lbm')
inputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 0.93202)
inputs.set_val(Aircraft.Fuel.NUM_TANKS, 7)
inputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, 1.0)

# Furnishings
# ---------------------------
inputs.set_val(Aircraft.Furnishings.MASS_SCALER, 0.81859)

# Fuselage
# ---------------------------
inputs.set_val(Aircraft.Fuselage.LAMINAR_FLOW_LOWER, 0.0)
inputs.set_val(Aircraft.Fuselage.LAMINAR_FLOW_UPPER, 0.0)
fuselage_length = 125.0
fuselage_length_units = 'ft'
inputs.set_val(Aircraft.Fuselage.LENGTH, fuselage_length, fuselage_length_units)
inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 13.0, 'ft')
fuselage_max_width = 12.3
fuselage_max_width_units = 'ft'
inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, fuselage_max_width, fuselage_max_width_units)
inputs.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)
inputs.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 97.5, 'ft')
inputs.set_val(
    Aircraft.Fuselage.PLANFORM_AREA,
    fuselage_length * fuselage_max_width,
    f'{fuselage_length_units}**2',
)
inputs.set_val(Aircraft.Fuselage.MASS_SCALER, 0.69981)
inputs.set_val(Aircraft.Fuselage.WETTED_AREA, 4235.082096, 'ft**2')  # Override
inputs.set_val(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0)
inputs.set_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR, False)

# Horizontal Tail
# ---------------------------
inputs.set_val(Aircraft.HorizontalTail.AREA, 349.522730527158, 'ft**2')
inputs.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, 5.22699386503068)
inputs.set_val(Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER, 0.0)
inputs.set_val(Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER, 29.0)
inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 0.2734375)
inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.115)
inputs.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0.0)
inputs.set_val(Aircraft.HorizontalTail.MASS_SCALER, 1.42225)
inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA, 576.571192, 'ft**2')  # Override
inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA_SCALER, 1.0)

# Hydraulics
# ---------------------------
inputs.set_val(Aircraft.Hydraulics.SYSTEM_PRESSURE, 5000.0, 'psi')
inputs.set_val(Aircraft.Hydraulics.MASS_SCALER, 0.95543)

# Instruments
# ---------------------------
inputs.set_val(Aircraft.Instruments.MASS_SCALER, 1.66955)

# Landing Gear
# ---------------------------
inputs.set_val(Aircraft.LandingGear.DRAG_COEFFICIENT, 0.024)
inputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, 0.8846)
inputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, 0.8846)

# Nacelle
# ---------------------------
inputs.set_val(Aircraft.Nacelle.AVG_DIAMETER, 7.2, 'ft')
inputs.set_val(Aircraft.Nacelle.AVG_LENGTH, 35.0, 'ft')
inputs.set_val(Aircraft.Nacelle.LAMINAR_FLOW_LOWER, 0.0)
inputs.set_val(Aircraft.Nacelle.LAMINAR_FLOW_UPPER, 0.0)
inputs.set_val(Aircraft.Nacelle.MASS_SCALER, 0.0)
nacelle_wetted_area = 244.468282
nacelle_wetted_area_units = 'ft**2'
inputs.set_val(
    Aircraft.Nacelle.WETTED_AREA, nacelle_wetted_area, nacelle_wetted_area_units
)  # Override
inputs.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, 1.0)
inputs.set_val(
    Aircraft.Nacelle.TOTAL_WETTED_AREA,
    nacelle_wetted_area * 2,
    nacelle_wetted_area_units,
)

# Paint
# ---------------------------
inputs.set_val(Aircraft.Paint.MASS_PER_UNIT_AREA, 0.0, 'lbm/ft**2')

# Propulsion and Engine
# ---------------------------
inputs.set_val(Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, 0.0, 'lbm')
# Must set this to zero if misc_mass is zero.
inputs.set_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 0.0, 'lbm')

filename = get_path('models/engines/turbofan_22k.csv')

inputs.set_val(Aircraft.Engine.DATA_FILE, filename)
inputs.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 0.99997747798473)
inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)
inputs.set_val(Aircraft.Engine.NUM_ENGINES, 2)
inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 0)
inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)
inputs.set_val(Aircraft.Engine.WING_LOCATIONS, 0.289682918)
inputs.set_val(Aircraft.Engine.SCALE_MASS, True)
inputs.set_val(Aircraft.Engine.MASS_SCALER, 1.15)
inputs.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
inputs.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
inputs.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
inputs.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lb/h')
inputs.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0)
inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
inputs.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
inputs.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
inputs.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')


# Vertical Tail
# ---------------------------
inputs.set_val(Aircraft.VerticalTail.AREA, 227.184358191707, 'ft**2')
inputs.set_val(Aircraft.VerticalTail.ASPECT_RATIO, 1.77777777777778)
inputs.set_val(Aircraft.VerticalTail.LAMINAR_FLOW_LOWER, 29.0)
inputs.set_val(Aircraft.VerticalTail.LAMINAR_FLOW_UPPER, 29.0)
inputs.set_val(Aircraft.VerticalTail.NUM_TAILS, 1)
inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 0.25)
inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.1)
inputs.set_val(Aircraft.VerticalTail.MASS_SCALER, 1.42225)
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA, 445.645658, 'ft**2')  # Override
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA_SCALER, 1.0)

# Wing
# ---------------------------
inputs.set_val(Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.33333)
inputs.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 1.6)
inputs.set_val(Aircraft.Wing.AREA, 1220.0, 'ft**2')
inputs.set_val(Aircraft.Wing.ASPECT_RATIO, 11.5587605382765)
inputs.set_val(Aircraft.Wing.ASPECT_RATIO_REF, 11.5587605382765)
inputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0)
inputs.set_val(
    Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
    np.array([0.273522534166506, 0.204274849507037, 0.0888152947868224, 0.0725353313595661]),
)
inputs.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 0.33333)
inputs.set_val(Aircraft.Wing.DIHEDRAL, 6.0, 'deg')
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333)
inputs.set_val(Aircraft.Wing.USE_DETAILED_MASS, True)
inputs.set_val(Aircraft.Wing.GLOVE_AND_BAT, 0.0, 'ft**2')
inputs.set_val(Aircraft.Wing.HEIGHT, 8.6, 'ft')
inputs.set_val(Aircraft.Wing.INPUT_STATION_DIST, np.array([0.0, 0.34453777998, 0.919, 1.0]))
inputs.set_val(Aircraft.Wing.LAMINAR_FLOW_LOWER, 0.0)
inputs.set_val(Aircraft.Wing.LAMINAR_FLOW_UPPER, 58.0)
inputs.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0)
inputs.set_val(Aircraft.Wing.LOAD_FRACTION, 1.0)
inputs.set_val(
    Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
    np.array([0.0, 23.6286942529271, 23.6286942529271]),
    'deg',
)
inputs.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.015)
inputs.set_val(Aircraft.Wing.MISC_MASS_SCALER, 1.7)
inputs.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 100)
inputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 0.749)
inputs.set_val(Aircraft.Wing.SPAN, 118.7505278165, 'ft')
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, False)
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 0.95)
inputs.set_val(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0)
inputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1.77696)
inputs.set_val(Aircraft.Wing.SWEEP, 23.6286942529271, 'deg')
inputs.set_val(Aircraft.Wing.TAPER_RATIO, 0.265189599754917)
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.12233)
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_DIST, np.array([0.14233, 0.12233, 0.1108, 0.1058]))
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_REF, 0.116565)
inputs.set_val(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75)
inputs.set_val(Aircraft.Wing.MASS_SCALER, 0.7412)
inputs.set_val(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0.0)
inputs.set_val(Aircraft.Wing.WETTED_AREA, 2210.280228, 'ft**2')  # Override
inputs.set_val(Aircraft.Wing.WETTED_AREA_SCALER, 1.0)

# Mission
# ---------------------------
inputs.set_val(Mission.Summary.CRUISE_MACH, 0.785)
inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)
inputs.set_val(Mission.Design.RANGE, 3500, 'NM')
inputs.set_val(Mission.Constraints.MAX_MACH, 0.785)
inputs.set_val(Mission.Landing.DRAG_COEFFICIENT_MIN, 0.045, 'unitless')
inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, 2.0, 'unitless')
inputs.set_val(Mission.Takeoff.AIRPORT_ALTITUDE, 0.0, 'ft')
inputs.set_val(Mission.Takeoff.DRAG_COEFFICIENT_MIN, 0.05, 'unitless')
inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2.0, 'unitless')
inputs.set_val(Mission.Takeoff.OBSTACLE_HEIGHT, 35.0, 'ft')
inputs.set_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 0.0, 'deg')
inputs.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.0175)
inputs.set_val(Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT, 0.35)
inputs.set_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT, 0.085000)
inputs.set_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT, -0.810000)
inputs.set_val(Mission.Landing.AIRPORT_ALTITUDE, 0.0, 'ft')
inputs.set_val(Mission.Landing.ROLLING_FRICTION_COEFFICIENT, 0.0175)
inputs.set_val(Mission.Landing.BRAKING_FRICTION_COEFFICIENT, 0.35)
inputs.set_val(Mission.Landing.FLARE_RATE, 2.2, 'deg/s')
inputs.set_val(Mission.Landing.OBSTACLE_HEIGHT, 50.0, 'ft')
inputs.set_val(Mission.Landing.SPOILER_DRAG_COEFFICIENT, 0.085000)
inputs.set_val(Mission.Landing.SPOILER_LIFT_COEFFICIENT, -0.810000)
inputs.set_val(Mission.Takeoff.THRUST_INCIDENCE, 0.0, 'deg')
inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, 577.0, 'lbm')

# Settings
# ---------------------------
inputs.set_val(Settings.EQUATIONS_OF_MOTION, EquationsOfMotion.HEIGHT_ENERGY)
inputs.set_val(Settings.AERODYNAMICS_METHOD, LegacyCode.FLOPS)
inputs.set_val(Settings.MASS_METHOD, LegacyCode.FLOPS)
inputs.set_val(Settings.VERBOSITY, 0)
inputs.set_val(Settings.PROBLEM_TYPE, ProblemType.SIZING)

# ---------------------------
#          OUTPUTS
# ---------------------------

outputs.set_val(Aircraft.Design.EMPTY_MASS, 67542.0, 'lbm')
outputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN, 996, 'lbm')
outputs.set_val(Aircraft.Design.TOUCHDOWN_MASS, 108976.4, 'lbm')
outputs.set_val(Aircraft.Design.OPERATING_MASS, 72642.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.MASS, 13105.0, 'lbm')
outputs.set_val(Aircraft.Design.STRUCTURE_MASS, 29336.0, 'lbm')
outputs.set_val(Aircraft.Design.SYSTEMS_EQUIP_MASS, 24105.0, 'lbm')
outputs.set_val(Aircraft.Design.TOTAL_WETTED_AREA, 7956.515738, 'ft**2')
outputs.set_val(Aircraft.Design.ZERO_FUEL_MASS, 103442.0, 'lbm')
outputs.set_val(Mission.Design.FUEL_MASS, 26292.0, 'lbm')

outputs.set_val(Aircraft.AirConditioning.MASS, 1541.0, 'lbm')

outputs.set_val(Aircraft.AntiIcing.MASS, 108.0, 'lbm')

outputs.set_val(Aircraft.APU.MASS, 1014.0, 'lbm')

outputs.set_val(Aircraft.Avionics.MASS, 2032.0, 'lbm')

outputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS, 5390.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.CARGO_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS, 675.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, 820.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 3033, 'lbm')
outputs.set_val(Aircraft.CrewPayload.PASSENGER_MASS, 25410.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 30800.0, 'lbm')

outputs.set_val(Aircraft.Electrical.MASS, 2375.0, 'lbm')

outputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS, 518.0, 'lbm')
outputs.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 33136.4, 'lbm')
outputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 462.0, 'lbm')
outputs.set_val(Aircraft.Fuel.WING_FUEL_CAPACITY, 33136.4, 'lbm')

outputs.set_val(Aircraft.Fins.MASS, 0.0, 'lbm')

outputs.set_val(Aircraft.Furnishings.MASS, 12556.0, 'lbm')

outputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, 12.65, 'ft')
outputs.set_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 125.0, 'ft')
# hand computed
outputs.set_val(Aircraft.Fuselage.CROSS_SECTION, 125.68137760226817, 'ft**2')
outputs.set_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 12.65 / (11.5587605382765 * 1220) ** 0.5)
outputs.set_val(Aircraft.Fuselage.FINENESS, 9.8814)
outputs.set_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER, 125.0 / 12.65)
outputs.set_val(Aircraft.Fuselage.MASS, 11750.0, 'lbm')

outputs.set_val(Aircraft.Canard.MASS, 0.0, 'lbm')

outputs.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 8.15, 'ft')
outputs.set_val(Aircraft.HorizontalTail.FINENESS, 0.1150)
outputs.set_val(Aircraft.HorizontalTail.MASS, 2147.0, 'lbm')

outputs.set_val(Aircraft.Hydraulics.MASS, 832.0, 'lbm')

outputs.set_val(Aircraft.Instruments.MASS, 907.0, 'lbm')

outputs.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 106.94, 'inch')
# Not printed in FLOPS, but total mass matches
outputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS, 4709.00236171, 'lbm')
outputs.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 74.86, 'inch')
# Not printed in FLOPS, but total mass matches
outputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS, 644.06298784, 'lbm')

outputs.set_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.array([35.0]), 'ft')
outputs.set_val(Aircraft.Nacelle.FINENESS, np.array([4.8611]))
outputs.set_val(Aircraft.Nacelle.MASS, 0.0, 'lbm')

outputs.set_val(Aircraft.Paint.MASS, 0.0, 'lbm')

inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5, 'lbf')
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
outputs.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 22200.5 * 2, 'lbf')

outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, 110.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2)

outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 2)

outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 0)

outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, 0.0, 'lbm')
# Not printed in FLOPS, but total mass matches
outputs.set_val(Aircraft.Engine.POD_MASS, 6619.27529209, 'lbm')
outputs.set_val(Aircraft.Engine.MASS, 12587.0 / 2.0, 'lbm')
outputs.set_val(Aircraft.Engine.ADDITIONAL_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_MISC_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS, 0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, 0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_MASS, 6293.8 * 2, 'lbm')

outputs.set_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 11.25, 'ft')
outputs.set_val(Aircraft.VerticalTail.FINENESS, 0.1000)
outputs.set_val(Aircraft.VerticalTail.MASS, 1175.0, 'lbm')

outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR, 11.9602)
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS, 5410.5, 'lbm')
outputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 10.27, 'ft')
outputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 0.333 * 1220, 'ft**2')
outputs.set_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 0.960516)
outputs.set_val(Aircraft.Wing.FINENESS, 0.1223)
outputs.set_val(Aircraft.Wing.MISC_MASS, 2281.9, 'lbm')
outputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS, 4329.9, 'lbm')
outputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS, 2741.0, 'lbm')
outputs.set_val(Aircraft.Wing.MASS, 8911.0, 'lbm')

outputs.set_val(Mission.Design.MACH, 0.779)
outputs.set_val(Mission.Design.LIFT_COEFFICIENT, 0.583)

# Create engine model
engines = [build_engine_deck(options=inputs)]
# Calls to preprocess_options() in this location should be avoided because they
# # will trigger when get_flops_inputs() is imported
# preprocess_options(inputs, engine_models=engine)

# build subsystems
default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)
default_mission_subsystems = get_default_mission_subsystems('FLOPS', engines)

# region - detailed takeoff
takeoff_trajectory_builder = TakeoffTrajectory('detailed_takeoff')

# region - takeoff aero
# block auto-formatting of tables
# fmt: off
takeoff_subsystem_options = {
    'core_aerodynamics': {
        'method': 'low_speed',
        'ground_altitude': 0.0,  # units='m'
        'angles_of_attack': [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],  # units='deg'
        'lift_coefficients': [
            0.5178, 0.6, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,
            1.35, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95,
        ],
        'drag_coefficients': [
            0.0674, 0.065, 0.065, 0.07, 0.072, 0.076, 0.084, 0.09,
            0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20,
        ],
        'lift_coefficient_factor': 1.0,
        'drag_coefficient_factor': 1.0,
    }
}
# fmt: on

takeoff_subsystem_options_spoilers = {
    'core_aerodynamics': {
        **takeoff_subsystem_options['core_aerodynamics'],
        'use_spoilers': True,
        'spoiler_drag_coefficient': inputs.get_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT),
        'spoiler_lift_coefficient': inputs.get_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT),
    }
}

# endregion - takeoff aero

# region - takeoff brake release
takeoff_brake_release_user_options = AviaryValues()

takeoff_brake_release_user_options.set_val('max_duration', val=60.0, units='s')
takeoff_brake_release_user_options.set_val('time_duration_ref', val=60.0, units='s')
takeoff_brake_release_user_options.set_val('distance_max', val=7500.0, units='ft')
takeoff_brake_release_user_options.set_val('max_velocity', val=167.85, units='kn')

takeoff_brake_release_initial_guesses = AviaryValues()

takeoff_brake_release_initial_guesses.set_val('time', [0.0, 30.0], 's')
takeoff_brake_release_initial_guesses.set_val('distance', [0.0, 4100.0], 'ft')
takeoff_brake_release_initial_guesses.set_val('velocity', [0.01, 150.0], 'kn')

gross_mass_units = 'lbm'
gross_mass = inputs.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
takeoff_brake_release_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_brake_release_initial_guesses.set_val('throttle', 1.0)
takeoff_brake_release_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

takeoff_brake_release_builder = TakeoffBrakeReleaseToDecisionSpeed(
    'takeoff_brake_release',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_brake_release_user_options,
    initial_guesses=takeoff_brake_release_initial_guesses,
)

takeoff_trajectory_builder.set_brake_release_to_decision_speed(takeoff_brake_release_builder)
# endregion - takeoff brake release

# region - takeoff decision speed
takeoff_decision_speed_user_options = AviaryValues()

takeoff_decision_speed_user_options.set_val('max_duration', val=60.0, units='s')
takeoff_decision_speed_user_options.set_val('time_duration_ref', val=60.0, units='s')
takeoff_decision_speed_user_options.set_val('time_initial_ref', val=35.0, units='s')
takeoff_decision_speed_user_options.set_val('distance_max', val=7500.0, units='ft')
takeoff_decision_speed_user_options.set_val('max_velocity', val=167.85, units='kn')

takeoff_decision_speed_initial_guesses = AviaryValues()

takeoff_decision_speed_initial_guesses.set_val('time', [30.0, 2.0], 's')
takeoff_decision_speed_initial_guesses.set_val('distance', [4100.0, 4500.0], 'ft')
takeoff_decision_speed_initial_guesses.set_val('velocity', [150.0, 160.0], 'kn')
takeoff_decision_speed_initial_guesses.set_val('mass', gross_mass, gross_mass_units)
takeoff_decision_speed_initial_guesses.set_val('throttle', 1.0)
takeoff_decision_speed_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

takeoff_decision_speed_builder = TakeoffDecisionSpeedToRotate(
    'takeoff_decision_speed',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_decision_speed_user_options,
    initial_guesses=takeoff_decision_speed_initial_guesses,
)

takeoff_trajectory_builder.set_decision_speed_to_rotate(takeoff_decision_speed_builder)
# endregion - takeoff decision speed

# region - takeoff rotate
takeoff_rotate_user_options = AviaryValues()

takeoff_rotate_user_options.set_val('max_duration', val=60.0, units='s')
takeoff_rotate_user_options.set_val('time_duration_ref', val=60.0, units='s')
takeoff_rotate_user_options.set_val('time_initial_ref', val=38.0, units='s')
takeoff_rotate_user_options.set_val('distance_max', val=7500.0, units='ft')
takeoff_rotate_user_options.set_val('max_velocity', val=167.85, units='kn')
takeoff_rotate_user_options.set_val('max_angle_of_attack', val=10.0, units='deg')

takeoff_rotate_initial_guesses = AviaryValues()

takeoff_rotate_initial_guesses.set_val('time', [32.0, 1.0], 's')
takeoff_rotate_initial_guesses.set_val('distance', [4500.0, 4800.0], 'ft')
takeoff_rotate_initial_guesses.set_val('velocity', [160.0, 160.0], 'kn')
takeoff_rotate_initial_guesses.set_val('throttle', 1.0)
takeoff_rotate_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [0.0, 8.0], 'deg')
takeoff_rotate_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_rotate_builder = TakeoffRotateToLiftoff(
    'takeoff_rotate',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_rotate_user_options,
    initial_guesses=takeoff_rotate_initial_guesses,
)

takeoff_trajectory_builder.set_rotate_to_liftoff(takeoff_rotate_builder)
# endregion - takeoff rotate

# region - takeoff liftoff
takeoff_liftoff_user_options = AviaryValues()

takeoff_liftoff_user_options.set_val('max_duration', val=12.0, units='s')
takeoff_liftoff_user_options.set_val('time_duration_ref', val=12.0, units='s')
takeoff_liftoff_user_options.set_val('time_initial_ref', val=39.0, units='s')
takeoff_liftoff_user_options.set_val('distance_max', val=7500.0, units='ft')
takeoff_liftoff_user_options.set_val('max_velocity', val=167.85, units='kn')
takeoff_liftoff_user_options.set_val('altitude_ref', val=35.0, units='ft')
takeoff_liftoff_user_options.set_val('flight_path_angle_ref', val=10.0, units='deg')
takeoff_liftoff_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')
takeoff_liftoff_user_options.set_val('upper_angle_of_attack', val=8.117, units='deg')
takeoff_liftoff_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')

takeoff_liftoff_initial_guesses = AviaryValues()

takeoff_liftoff_initial_guesses.set_val('time', [33.0, 4.0], 's')
takeoff_liftoff_initial_guesses.set_val('distance', [4800, 5700.0], 'ft')
takeoff_liftoff_initial_guesses.set_val('velocity', [160, 167.0], 'kn')
takeoff_liftoff_initial_guesses.set_val('throttle', 1.0)
takeoff_liftoff_initial_guesses.set_val('altitude', [0, 35.0], 'ft')
takeoff_liftoff_initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [0, 6.0], 'deg')
takeoff_liftoff_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 8.117, 'deg')
takeoff_liftoff_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_liftoff_builder = TakeoffLiftoffToObstacle(
    'takeoff_liftoff',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_liftoff_user_options,
    initial_guesses=takeoff_liftoff_initial_guesses,
)

takeoff_trajectory_builder.set_liftoff_to_obstacle(takeoff_liftoff_builder)
# endregion - takeoff liftoff

# region - takeoff mic p2
takeoff_mic_p2_user_options = AviaryValues()

takeoff_mic_p2_user_options.set_val('max_duration', val=25.0, units='s')
takeoff_mic_p2_user_options.set_val('time_duration_ref', val=25.0, units='s')
takeoff_mic_p2_user_options.set_val('time_initial_ref', val=50.0, units='s')
takeoff_mic_p2_user_options.set_val('distance_max', val=12000.0, units='ft')
takeoff_mic_p2_user_options.set_val('max_velocity', val=167.85, units='kn')
takeoff_mic_p2_user_options.set_val('altitude_ref', val=1500.0, units='ft')
takeoff_mic_p2_user_options.set_val('mic_altitude', val=985.0, units='ft')

takeoff_mic_p2_user_options.set_val('flight_path_angle_ref', val=12.0, units='deg')

takeoff_mic_p2_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

takeoff_mic_p2_user_options.set_val('upper_angle_of_attack', val=12, units='deg')
takeoff_mic_p2_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')
takeoff_mic_p2_user_options.set_val('mic_altitude', val=985.0, units='ft')

takeoff_mic_p2_initial_guesses = AviaryValues()

takeoff_mic_p2_initial_guesses.set_val('time', [36.0, 18], 's')
takeoff_mic_p2_initial_guesses.set_val('distance', [5700, 10000.0], 'ft')
takeoff_mic_p2_initial_guesses.set_val('velocity', [167, 167.0], 'kn')
takeoff_mic_p2_initial_guesses.set_val('throttle', 1.0)
takeoff_mic_p2_initial_guesses.set_val('altitude', [35, 985.0], 'ft')
takeoff_mic_p2_initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [7.0, 10.0], 'deg')
takeoff_mic_p2_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 8.117, 'deg')
takeoff_mic_p2_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_mic_p2_builder = TakeoffObstacleToMicP2(
    'takeoff_mic_p2',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_mic_p2_user_options,
    initial_guesses=takeoff_mic_p2_initial_guesses,
)

takeoff_trajectory_builder.set_obstacle_to_mic_p2(takeoff_mic_p2_builder)
# endregion - takeoff mic p2

# region - mic p2 to engine cutback
takeoff_mic_p2_to_engine_cutback_user_options = AviaryValues()

takeoff_mic_p2_to_engine_cutback_user_options.set_val('max_duration', val=30.0, units='s')
takeoff_mic_p2_to_engine_cutback_user_options.set_val('time_duration_ref', val=40.0, units='s')
takeoff_mic_p2_to_engine_cutback_user_options.set_val('time_initial_ref', val=65.0, units='s')

takeoff_mic_p2_to_engine_cutback_user_options.set_val('distance_max', val=20000.0, units='ft')

takeoff_mic_p2_to_engine_cutback_user_options.set_val('max_velocity', val=167.85, units='kn')

takeoff_mic_p2_to_engine_cutback_user_options.set_val('altitude_ref', val=3000.0, units='ft')

takeoff_mic_p2_to_engine_cutback_user_options.set_val(
    'flight_path_angle_ref', val=12.0, units='deg'
)

takeoff_mic_p2_to_engine_cutback_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

takeoff_mic_p2_to_engine_cutback_user_options.set_val('upper_angle_of_attack', val=12, units='deg')

takeoff_mic_p2_to_engine_cutback_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')

takeoff_mic_p2_to_engine_cutback_user_options.set_val('final_range', val=19000.0, units='ft')

takeoff_mic_p2_to_engine_cutback_initial_guesses = AviaryValues()

takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val('time', [53.0, 27], 's')
takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val('distance', [10000, 19000.0], 'ft')
takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val('velocity', [167, 167.0], 'kn')
takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val('throttle', 1.0)
takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val('altitude', [985, 2500.0], 'ft')

takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE, [11.0, 10.0], 'deg'
)

takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val(
    Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.0, 'deg'
)

takeoff_mic_p2_to_engine_cutback_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_mic_p2_to_engine_cutback_builder = TakeoffMicP2ToEngineCutback(
    'takeoff_mic_p2_to_engine_cutback',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_mic_p2_to_engine_cutback_user_options,
    initial_guesses=takeoff_mic_p2_to_engine_cutback_initial_guesses,
)

takeoff_trajectory_builder.set_mic_p2_to_engine_cutback(takeoff_mic_p2_to_engine_cutback_builder)
# endregion - mic p2 to engine cutback

# region - engine cutback phase
takeoff_engine_cutback_user_options = AviaryValues()

# NOTE: the N3CC data we have does not do the engine cutback or the post cutback
# acoustic phases. In general, this can be found by optimizing to find the
# setting that gives 2.29 deg flight path angle AND level flight with 1-engine out.
# TODO: Set up and perform this opt.
# This value is just an estimate.
cutback_throttle = 0.65

# TODO: Make this a variable in the hierarchy.
cutback_rate = 0.1  # Throttle setting per second
cutback_duration = (1.0 - cutback_throttle) / cutback_rate

takeoff_engine_cutback_user_options.set_val('time_initial_ref', val=95.0, units='s')
takeoff_engine_cutback_user_options.set_val('distance_max', val=21000.0, units='ft')
takeoff_engine_cutback_user_options.set_val('max_velocity', val=167.85, units='kn')
takeoff_engine_cutback_user_options.set_val('altitude_ref', val=4000.0, units='ft')

takeoff_engine_cutback_user_options.set_val('flight_path_angle_ref', val=12.0, units='deg')

takeoff_engine_cutback_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

takeoff_engine_cutback_user_options.set_val('upper_angle_of_attack', val=12, units='deg')
takeoff_engine_cutback_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')

takeoff_engine_cutback_initial_guesses = AviaryValues()

takeoff_engine_cutback_initial_guesses.set_val('time', [84.0, cutback_duration], 's')
takeoff_engine_cutback_initial_guesses.set_val('distance', [19000, 20000.0], 'ft')
takeoff_engine_cutback_initial_guesses.set_val('velocity', [167, 167.0], 'kn')
takeoff_engine_cutback_initial_guesses.set_val('throttle', [1.0, cutback_throttle])
takeoff_engine_cutback_initial_guesses.set_val('altitude', [2500.0, 2600.0], 'ft')

takeoff_engine_cutback_initial_guesses.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE, [10.0, 10.0], 'deg'
)

takeoff_engine_cutback_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.0, 'deg')
takeoff_engine_cutback_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_engine_cutback_builder = TakeoffEngineCutback(
    'takeoff_engine_cutback',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_engine_cutback_user_options,
    initial_guesses=takeoff_engine_cutback_initial_guesses,
)

takeoff_trajectory_builder.set_engine_cutback(takeoff_engine_cutback_builder)
# endregion - engine cutback phase

# region - engine cutback to mic p1
takeoff_engine_cutback_to_mic_p1_user_options = AviaryValues()

takeoff_engine_cutback_to_mic_p1_user_options.set_val('max_duration', val=11.0, units='s')
takeoff_engine_cutback_to_mic_p1_user_options.set_val('time_duration_ref', val=11.0, units='s')
takeoff_engine_cutback_to_mic_p1_user_options.set_val('time_initial_ref', val=97.0, units='s')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('distance_max', val=22000.0, units='ft')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('max_velocity', val=167.85, units='kn')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('altitude_ref', val=3500.0, units='ft')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('mic_range', val=21325.0, units='ft')

takeoff_engine_cutback_to_mic_p1_user_options.set_val(
    'flight_path_angle_ref', val=12.0, units='deg'
)

takeoff_engine_cutback_to_mic_p1_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('upper_angle_of_attack', val=15, units='deg')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')

takeoff_engine_cutback_to_mic_p1_user_options.set_val('mic_range', val=21325.0, units='ft')

takeoff_engine_cutback_to_mic_p1_initial_guesses = AviaryValues()

takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val('time', [87.0, 10], 's')
takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val('distance', [20000, 21325.0], 'ft')
takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val('velocity', [167, 167.0], 'kn')
takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val('throttle', cutback_throttle)

takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val('altitude', [2600, 2700.0], 'ft')

takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE, 2.29, 'deg'
)
takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val(
    Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.0, 'deg'
)

takeoff_engine_cutback_to_mic_p1_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_engine_cutback_to_mic_p1_builder = TakeoffEngineCutbackToMicP1(
    'takeoff_engine_cutback_to_mic_p1',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_engine_cutback_to_mic_p1_user_options,
    initial_guesses=takeoff_engine_cutback_to_mic_p1_initial_guesses,
)

takeoff_trajectory_builder.set_engine_cutback_to_mic_p1(takeoff_engine_cutback_to_mic_p1_builder)
# endregion - engine cutback to mic p1

# region - mic p1 to climb
takeoff_mic_p1_to_climb_user_options = AviaryValues()

takeoff_mic_p1_to_climb_user_options.set_val('max_duration', val=40.0, units='s')
takeoff_mic_p1_to_climb_user_options.set_val('time_duration_ref', val=40.0, units='s')
takeoff_mic_p1_to_climb_user_options.set_val('time_initial_ref', val=100.0, units='s')
takeoff_mic_p1_to_climb_user_options.set_val('distance_max', val=30000.0, units='ft')
takeoff_mic_p1_to_climb_user_options.set_val('max_velocity', val=167.85, units='kn')
takeoff_mic_p1_to_climb_user_options.set_val('altitude_ref', val=4000.0, units='ft')

takeoff_mic_p1_to_climb_user_options.set_val('flight_path_angle_ref', val=12.0, units='deg')

takeoff_mic_p1_to_climb_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

takeoff_mic_p1_to_climb_user_options.set_val('upper_angle_of_attack', val=15, units='deg')

takeoff_mic_p1_to_climb_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')
takeoff_mic_p1_to_climb_user_options.set_val('mic_range', val=30000.0, units='ft')

takeoff_mic_p1_to_climb_initial_guesses = AviaryValues()

takeoff_mic_p1_to_climb_initial_guesses.set_val('time', [95.0, 32], 's')
takeoff_mic_p1_to_climb_initial_guesses.set_val('distance', [21325, 30000.0], 'ft')
takeoff_mic_p1_to_climb_initial_guesses.set_val('velocity', [167, 167.0], 'kn')
takeoff_mic_p1_to_climb_initial_guesses.set_val('throttle', cutback_throttle)
takeoff_mic_p1_to_climb_initial_guesses.set_val('altitude', [2700, 3200.0], 'ft')
takeoff_mic_p1_to_climb_initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 2.29, 'deg')
takeoff_mic_p1_to_climb_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.0, 'deg')
takeoff_mic_p1_to_climb_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

takeoff_mic_p1_to_climb_builder = TakeoffMicP1ToClimb(
    'takeoff_mic_p1_to_climb',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=takeoff_mic_p1_to_climb_user_options,
    initial_guesses=takeoff_mic_p1_to_climb_initial_guesses,
)

takeoff_trajectory_builder.set_mic_p1_to_climb(takeoff_mic_p1_to_climb_builder)
# endregion - mic p1 to climb

# NOTE copied/derived from N3CC FLOPS output
#    - file: N3CC FLOPS Out -low speed polar input echo deleted.txt
#        - table starting near line 2297
#            * * * ALL ENGINES OPERATING TAKEOFF * * *
detailed_takeoff = AviaryValues()

detailed_takeoff.set_val('time', [0.77, 32.01, 33.00, 35.40], 's')
detailed_takeoff.set_val(Dynamic.Mission.DISTANCE, [3.08, 4626.88, 4893.40, 5557.61], 'ft')
detailed_takeoff.set_val(Dynamic.Mission.ALTITUDE, [0.00, 0.00, 0.64, 27.98], 'ft')
velocity = np.array([4.74, 157.58, 160.99, 166.68])
detailed_takeoff.set_val(Dynamic.Mission.VELOCITY, velocity, 'kn')
detailed_takeoff.set_val(Dynamic.Atmosphere.MACH, [0.007, 0.2342, 0.2393, 0.2477])

detailed_takeoff.set_val(
    Dynamic.Vehicle.Propulsion.THRUST_TOTAL, [44038.8, 34103.4, 33929.0, 33638.2], 'lbf'
)

detailed_takeoff.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [0.000, 3.600, 8.117, 8.117], 'deg')
detailed_takeoff.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [0.000, 0.000, 0.612, 4.096], 'deg')

# missing from the default FLOPS output generated by hand
# RANGE_RATE = VELOCITY * cos(flight_path_angle)
range_rate = np.array([4.74, 157.58, 160.98, 166.25])
detailed_takeoff.set_val(Dynamic.Mission.DISTANCE_RATE, range_rate, 'kn')
# ALTITUDE_RATE = VELOCITY * sin(flight_path_angle)
altitude_rate = np.array([0.00, 0.00, 1.72, 11.91])
detailed_takeoff.set_val(Dynamic.Mission.ALTITUDE_RATE, altitude_rate, 'kn')

# NOTE FLOPS output is horizontal acceleration only
#    - divide the FLOPS values by the cos(flight_path_angle)
# detailed_takeoff.set_val(Dynamic.Mission.VELOCITY_RATE, [10.36, 6.20, 5.23, 2.69], 'ft/s**2')
velocity_rate = [10.36, 6.20, 5.23, 2.70]
detailed_takeoff.set_val(Dynamic.Mission.VELOCITY_RATE, velocity_rate, 'ft/s**2')

# NOTE FLOPS output is based on "constant" takeoff mass - assume gross weight
#    - currently neglecting taxi
detailed_takeoff.set_val(Dynamic.Vehicle.MASS, [129734.0, 129734.0, 129734.0, 129734.0], 'lbm')

lift_coeff = np.array([0.5580, 0.9803, 1.4831, 1.3952])
drag_coeff = np.array([0.0801, 0.0859, 0.1074, 0.1190])

S = inputs.get_val(Aircraft.Wing.AREA, 'm**2')
v = detailed_takeoff.get_val(Dynamic.Mission.VELOCITY, 'm/s')
# NOTE sea level; includes effect of FLOPS &TOLIN DTCT 10 DEG C
rho = 1.18391  # kg/m**3

RHV2 = 0.5 * rho * v * v * S

lift = RHV2 * lift_coeff  # N
detailed_takeoff.set_val(Dynamic.Vehicle.LIFT, lift, 'N')

drag = RHV2 * drag_coeff  # N
detailed_takeoff.set_val(Dynamic.Vehicle.DRAG, drag, 'N')


def _split_aviary_values(aviary_values, slicing):
    tmp = AviaryValues()

    for key, (val, units) in aviary_values:
        tmpval = val[slicing]
        tmp.set_val(key, tmpval, units)

    return tmp


detailed_takeoff_ground = _split_aviary_values(detailed_takeoff, slice(2))
detailed_takeoff_climbing = _split_aviary_values(detailed_takeoff, slice(2, 4))
# endregion - detailed takeoff

# region balanced field length
balanced_trajectory_builder = TakeoffTrajectory('balanced_takeoff')

balanced_brake_release_user_options = AviaryValues()

balanced_brake_release_user_options.set_val('max_duration', val=60.0, units='s')
balanced_brake_release_user_options.set_val('time_duration_ref', val=60.0, units='s')
balanced_brake_release_user_options.set_val('distance_max', val=7500.0, units='ft')
balanced_brake_release_user_options.set_val('max_velocity', val=167.85, units='kn')

balanced_brake_release_initial_guesses = AviaryValues()

balanced_brake_release_initial_guesses.set_val('time', [0.0, 30.0], 's')
balanced_brake_release_initial_guesses.set_val('distance', [0.0, 4100.0], 'ft')
balanced_brake_release_initial_guesses.set_val('velocity', [0.01, 150.0], 'kn')
balanced_brake_release_initial_guesses.set_val('mass', gross_mass, gross_mass_units)
balanced_brake_release_initial_guesses.set_val('throttle', 1.0)
balanced_brake_release_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

balanced_brake_release_builder = TakeoffBrakeReleaseToDecisionSpeed(
    'balanced_brake_release',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=balanced_brake_release_user_options,
    initial_guesses=balanced_brake_release_initial_guesses,
)

balanced_trajectory_builder.set_brake_release_to_decision_speed(balanced_brake_release_builder)

balanced_decision_speed_user_options = AviaryValues()

balanced_decision_speed_user_options.set_val('max_duration', val=60.0, units='s')
balanced_decision_speed_user_options.set_val('time_duration_ref', val=5.0, units='s')
balanced_decision_speed_user_options.set_val('time_initial_ref', val=35.0, units='s')
balanced_decision_speed_user_options.set_val('distance_max', val=7500.0, units='ft')
balanced_decision_speed_user_options.set_val('max_velocity', val=167.85, units='kn')

balanced_decision_speed_initial_guesses = AviaryValues()

num_engines = float(inputs.get_val(Aircraft.Engine.NUM_ENGINES))
engine_out_throttle = (num_engines - 1) / num_engines

balanced_decision_speed_initial_guesses.set_val('time', [30.0, 2.0], 's')
balanced_decision_speed_initial_guesses.set_val('distance', [4100.0, 4500.0], 'ft')
balanced_decision_speed_initial_guesses.set_val('velocity', [150.0, 160.0], 'kn')
balanced_decision_speed_initial_guesses.set_val('mass', gross_mass, gross_mass_units)
balanced_decision_speed_initial_guesses.set_val('throttle', engine_out_throttle)
balanced_decision_speed_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

balanced_decision_speed_builder = TakeoffDecisionSpeedToRotate(
    'balanced_decision_speed',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=balanced_decision_speed_user_options,
    initial_guesses=balanced_decision_speed_initial_guesses,
)

balanced_trajectory_builder.set_decision_speed_to_rotate(balanced_decision_speed_builder)

balanced_rotate_user_options = AviaryValues()

balanced_rotate_user_options.set_val('max_duration', val=20.0, units='s')
balanced_rotate_user_options.set_val('time_duration_ref', val=5.0, units='s')
balanced_rotate_user_options.set_val('time_initial_ref', val=35.0, units='s')
balanced_rotate_user_options.set_val('distance_max', val=7500.0, units='ft')
balanced_rotate_user_options.set_val('max_velocity', val=167.85, units='kn')
balanced_rotate_user_options.set_val('max_angle_of_attack', val=8.117, units='deg')

balanced_rotate_initial_guesses = AviaryValues()

balanced_rotate_initial_guesses.set_val('time', [32.0, 1.0], 's')
balanced_rotate_initial_guesses.set_val('distance', [4500.0, 4800.0], 'ft')
balanced_rotate_initial_guesses.set_val('velocity', [160.0, 160.0], 'kn')
balanced_rotate_initial_guesses.set_val('throttle', engine_out_throttle)
balanced_rotate_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [0.0, 8.0], 'deg')
balanced_rotate_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

balanced_rotate_builder = TakeoffRotateToLiftoff(
    'balanced_rotate',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=balanced_rotate_user_options,
    initial_guesses=balanced_rotate_initial_guesses,
)

balanced_trajectory_builder.set_rotate_to_liftoff(balanced_rotate_builder)

balanced_liftoff_user_options = AviaryValues()

balanced_liftoff_user_options.set_val('max_duration', val=20.0, units='s')
balanced_liftoff_user_options.set_val('time_duration_ref', val=20.0, units='s')
balanced_liftoff_user_options.set_val('time_initial_ref', val=40.0, units='s')
balanced_liftoff_user_options.set_val('distance_max', val=7500.0, units='ft')
balanced_liftoff_user_options.set_val('max_velocity', val=167.85, units='kn')
balanced_liftoff_user_options.set_val('altitude_ref', val=35.0, units='ft')
balanced_liftoff_user_options.set_val('flight_path_angle_ref', val=5.0, units='deg')
balanced_liftoff_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')
balanced_liftoff_user_options.set_val('upper_angle_of_attack', val=8.117, units='deg')
balanced_liftoff_user_options.set_val('angle_of_attack_ref', val=10.0, units='deg')

balanced_liftoff_initial_guesses = AviaryValues()

balanced_liftoff_initial_guesses.set_val('time', [33.0, 4.0], 's')
balanced_liftoff_initial_guesses.set_val('distance', [4800.0, 7000.0], 'ft')
balanced_liftoff_initial_guesses.set_val('velocity', [160.0, 167.0], 'kn')
balanced_liftoff_initial_guesses.set_val('throttle', engine_out_throttle)
balanced_liftoff_initial_guesses.set_val('altitude', [0.0, 35.0], 'ft')
balanced_liftoff_initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [0.0, 5.0], 'deg')
balanced_liftoff_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 8.117, 'deg')
balanced_liftoff_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

balanced_liftoff_builder = TakeoffLiftoffToObstacle(
    'balanced_liftoff',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=balanced_liftoff_user_options,
    initial_guesses=balanced_liftoff_initial_guesses,
)

balanced_trajectory_builder.set_liftoff_to_obstacle(balanced_liftoff_builder)

balanced_delayed_brake_user_options = AviaryValues()

balanced_delayed_brake_user_options.set_val('time_duration_ref', val=4.0, units='s')
balanced_delayed_brake_user_options.set_val('time_initial_ref', val=35.0, units='s')
balanced_delayed_brake_user_options.set_val('distance_max', val=7500.0, units='ft')
balanced_delayed_brake_user_options.set_val('max_velocity', val=167.85, units='kn')

balanced_delayed_brake_initial_guesses = AviaryValues()

balanced_delayed_brake_initial_guesses.set_val('time', [30.0, 3.0], 's')
balanced_delayed_brake_initial_guesses.set_val('distance', [4100.0, 4600.0], 'ft')
balanced_delayed_brake_initial_guesses.set_val('velocity', [150.0, 150.0], 'kn')
balanced_delayed_brake_initial_guesses.set_val('mass', gross_mass, gross_mass_units)
balanced_delayed_brake_initial_guesses.set_val('throttle', engine_out_throttle)
balanced_delayed_brake_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

# NOTE: no special handling required; re-use existing phase builder type
balanced_delayed_brake_builder = TakeoffDecisionSpeedBrakeDelay(
    'balanced_delayed_brake',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options,
    user_options=balanced_delayed_brake_user_options,
    initial_guesses=balanced_delayed_brake_initial_guesses,
)

balanced_trajectory_builder.set_decision_speed_to_brake(balanced_delayed_brake_builder)

balanced_abort_user_options = AviaryValues()

balanced_abort_user_options.set_val('max_duration', val=60.0, units='s')
balanced_abort_user_options.set_val('time_initial_ref', val=35.0, units='s')
balanced_abort_user_options.set_val('time_duration_ref', val=60.0, units='s')
balanced_abort_user_options.set_val('distance_max', val=7500.0, units='ft')
balanced_abort_user_options.set_val('max_velocity', val=167.85, units='kn')

balanced_abort_initial_guesses = AviaryValues()

balanced_abort_initial_guesses.set_val('time', [32.0, 22.0], 's')
balanced_abort_initial_guesses.set_val('distance', [4600.0, 7000.0], 'ft')
balanced_abort_initial_guesses.set_val('velocity', [150.0, 0.01], 'kn')
balanced_abort_initial_guesses.set_val('mass', gross_mass, gross_mass_units)
balanced_abort_initial_guesses.set_val('throttle', 0.0)
balanced_abort_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

balanced_abort_builder = TakeoffBrakeToAbort(
    'balanced_abort',
    core_subsystems=default_mission_subsystems,
    subsystem_options=takeoff_subsystem_options_spoilers,
    user_options=balanced_abort_user_options,
    initial_guesses=balanced_abort_initial_guesses,
)

distance_max = balanced_liftoff_user_options.get_val('distance_max', 'ft')

balanced_trajectory_builder.set_brake_to_abort(
    balanced_abort_builder, balanced_field_ref=distance_max
)
# endregion balanced field length

# region - detailed landing
landing_trajectory_builder = LandingTrajectory('detailed_landing')

# block auto-formatting of tables
# fmt: off

# region - landing aero
landing_subsystem_options = {
    'core_aerodynamics': {
        'method': 'low_speed',
        'ground_altitude': 0.0,  # units='m'
        'angles_of_attack': [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],  # units='deg'
        'lift_coefficients': [
            0.7, 0.9, 1.05, 1.15, 1.25, 1.4, 1.5, 1.60,
            1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.40,
        ],
        'drag_coefficients': [
            0.1, 0.1, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
            0.18, 0.20, 0.22, 0.24, 0.26, 0.3, 0.32, 0.34,
        ],
        'lift_coefficient_factor': 1.0,
        'drag_coefficient_factor': 1.0,
    }
}

# fmt: on

landing_subsystem_options_spoilers = {
    'core_aerodynamics': {
        **landing_subsystem_options['core_aerodynamics'],
        'use_spoilers': True,
        'spoiler_lift_coefficient': -0.81,
        'spoiler_drag_coefficient': 0.085,
    }
}

#                            VELOCITY     TIME   DISTANCE    ALPHA   ALTITUDE
#                              KNOTS      SEC.     FEET       DEG.     FEET
# 50. FOOT OBSTACLE            138.65     0.00       0.00     5.22    50.00
# START OF FLARE               138.65     3.13     731.46     5.09    11.67
# TOUCHDOWN                    138.42     4.22     986.05     7.24     0.00
# SPOILER ACTUATION            136.12     5.68    1324.21     4.02     0.00
# WHEEL BRAKES APPLIED         136.12     5.68    1324.21     4.02     0.
# END OF LANDING                 0.00    24.49    3409.47     0.00     0.00

detailed_landing = AviaryValues()

# block auto-formatting of tables
# fmt: off
values = np.array(
    [
        -4.08, -4.08, -3.92, -3.76, -3.59, -3.43, -3.27, -3.1, -2.94, -2.78, -2.61, -2.45, -2.29,
        -2.12, -1.96, -1.8, -1.63, -1.47, -1.31, -1.14, -0.98, -0.82, -0.65, -0.49, -0.33, -0.16, 0,
        0.16, 0.33, 0.49, 0.65, 0.82, 0.98, 1.14, 1.31, 1.47, 1.63, 1.8, 1.96, 2.12, 2.29, 2.45,
        2.61, 2.78, 2.94, 3.1, 3.13, 3.92, 4.97, 5.68, 5.93, 6.97, 7.97, 8.97, 9.97, 10.97, 11.97,
        12.97, 13.97, 14.97, 15.97, 16.97, 17.97, 18.97, 19.97, 20.97, 21.97, 22.97, 23.97, 24.49
    ]
)

base = values[0]
values = values - base
detailed_landing.set_val('time', values, 's')

values = np.array(
    [
        -954.08, -954.06, -915.89, -877.73, -839.57, -801.41, -763.25, -725.08, -686.92, -648.76,
        -610.6, -572.43, -534.27, -496.11, -457.95, -419.78, -381.62, -343.46, -305.3, -267.14,
        -228.97, -190.81, -152.65, -114.49, -76.32, -38.16, 0, 38.16, 76.32, 114.49, 152.65, 190.81,
        228.97, 267.14, 305.3, 343.46, 381.62, 419.78, 457.95, 496.11, 534.27, 572.43, 610.6,
        648.76, 686.92, 725.08, 731.46, 917.22, 1160.47, 1324.21, 1381.29, 1610.61, 1817.53,
        2010.56, 2190, 2356.17, 2509.36, 2649.84, 2777.85, 2893.6, 2997.28, 3089.05, 3169.07,
        3237.45, 3294.31, 3339.73, 3373.78, 3396.51, 3407.96, 3409.47
    ]
)
# fmt: on

base = values[0]
values = values - base
detailed_landing.set_val(Dynamic.Mission.DISTANCE, values, 'ft')

# block auto-formatting of tables
# fmt: off
detailed_landing.set_val(
    Dynamic.Mission.ALTITUDE,
    [
        100, 100, 98, 96, 94, 92, 90, 88, 86, 84, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58,
        56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12,
        11.67, 2.49, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    'ft',
)

detailed_landing.set_val(
    Dynamic.Mission.VELOCITY,
    np.array(
        [
            138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65,
            138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65,  138.65, 138.65, 138.65, 138.65,
            138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65,
            138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65, 138.65,
            138.65, 138.65, 138.65, 138.60, 137.18, 136.12, 134.43, 126.69, 118.46, 110.31, 102.35,
            94.58, 86.97, 79.52, 72.19, 64.99, 57.88, 50.88, 43.95, 37.09, 30.29, 23.54, 16.82,
            10.12, 3.45, 0
        ]
    ),
    'kn',
)

detailed_landing.set_val(
    Dynamic.Atmosphere.MACH,
    [
        0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061,
        0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061,
        0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061,
        0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061, 0.2061,
        0.2061, 0.2061, 0.2061, 0.2060, 0.2039, 0.2023, 0.1998, 0.1883, 0.1761, 0.1639, 0.1521,
        0.1406, 0.1293, 0.1182, 0.1073, 0.0966, 0.086, 0.0756, 0.0653, 0.0551, 0.045, 0.035, 0.025,
        0.015, 0.0051, 0,
    ],
)

detailed_landing.set_val(
    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
    [
        7614.0, 7614.0, 7607.7, 7601.0, 7593.9, 7586.4, 7578.5, 7570.2, 7561.3, 7551.8, 7541.8,
        7531.1, 7519.7, 7507.6, 7494.6, 7480.6, 7465.7, 7449.7, 7432.5, 7414.0, 7394.0, 7372.3,
        7348.9, 7323.5, 7295.9, 7265.8, 7233.0, 7197.1, 7157.7, 7114.3, 7066.6, 7013.8, 6955.3,
        6890.2, 6817.7, 6736.7, 6645.8, 6543.5, 6428.2, 6297.6, 6149.5, 5980.9, 5788.7, 5569.3,
        5318.5, 5032.0, 4980.3, 4102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    'lbf',
)

detailed_landing.set_val(
    Dynamic.Vehicle.ANGLE_OF_ATTACK,
    [
        5.231, 5.231, 5.231, 5.23, 5.23, 5.23, 5.23, 5.23, 5.229, 5.229, 5.229, 5.229, 5.228, 5.228,
        5.227, 5.227, 5.227, 5.226, 5.226, 5.225, 5.224, 5.224, 5.223, 5.222, 5.221, 5.22, 5.219,
        5.218, 5.217, 5.215, 5.214, 5.212, 5.21, 5.207, 5.204, 5.201, 5.197, 5.193, 5.187, 5.181,
        5.173, 5.163, 5.151, 5.136, 5.117, 5.091, 5.086, 6.834, 5.585, 4.023, 3.473, 1.185, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    'deg')

# glide slope == flight path angle?
detailed_landing.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE,
    np.array(
        [
            -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
            -3, -3, -3, -2.47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    ),
    'deg',
)
# fmt: on

# missing from the default FLOPS output generated by script
# RANGE_RATE = VELOCITY * cos(flight_path_angle)
velocity: np.ndarray = detailed_landing.get_val(Dynamic.Mission.VELOCITY, 'kn')
flight_path_angle = detailed_landing.get_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 'rad')
range_rate = velocity * np.cos(-flight_path_angle)
detailed_landing.set_val(Dynamic.Mission.DISTANCE_RATE, range_rate, 'kn')
# ALTITUDE_RATE = VELOCITY * sin(flight_path_angle)
altitude_rate = velocity * np.sin(flight_path_angle)
detailed_landing.set_val(Dynamic.Mission.ALTITUDE_RATE, altitude_rate, 'kn')

# NOTE FLOPS output is horizontal acceleration only, and virtually no acceleration while
# airborne
#    - ignored for now

# NOTE FLOPS output is based on "constant" landing mass - assume reserves weight
#    - currently neglecting taxi
detailed_landing_mass = 106292.0  # units='lbm'

detailed_landing.set_val(
    Dynamic.Vehicle.MASS, np.full(velocity.shape, detailed_landing_mass), 'lbm'
)

# block auto-formatting of tables
# fmt: off
# lift/drag is calculated very close to landing altitude (sea level, in this case)...
lift_coeff = np.array(
    [
        1.4091, 1.4091, 1.4091, 1.4091, 1.4092, 1.4092, 1.4092, 1.4092, 1.4092, 1.4092, 1.4092,
        1.4092, 1.4092, 1.4093, 1.4093, 1.4093, 1.4093, 1.4093, 1.4094, 1.4094, 1.4094, 1.4094,
        1.4095, 1.4095, 1.4095, 1.4096, 1.4096, 1.4096, 1.4097, 1.4097, 1.4098, 1.4099, 1.4099,
        1.41, 1.4101, 1.4102, 1.4103, 1.4105, 1.4106, 1.4108, 1.4109, 1.4112, 1.4114, 1.4117, 1.412,
        1.4124, 1.4124, 1.6667, 1.595, 1.397, 0.5237, 0.2338, 0.046, 0.046, 0.046, 0.046, 0.046,
        0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046, 0.046
    ]
)

drag_coeff = np.array(
    [
        0.1731, 0.1731, 0.173, 0.173, 0.1729, 0.1728, 0.1727, 0.1726, 0.1724, 0.1723, 0.1722,
        0.1721, 0.1719, 0.1718, 0.1716, 0.1714, 0.1712, 0.171, 0.1708, 0.1705, 0.1703, 0.17, 0.1697,
        0.1694, 0.169, 0.1686, 0.1682, 0.1677, 0.1672, 0.1666, 0.166, 0.1653, 0.1646, 0.1637,
        0.1628, 0.1618, 0.1606, 0.1592, 0.1577, 0.1561, 0.1541, 0.1519, 0.1495, 0.1466, 0.1434,
        0.1396, 0.139, 0.13, 0.1207, 0.1099, 0.1922, 0.1827, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785,
        0.1785, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785, 0.1785,
        0.1785, 0.1785
    ]
)
# fmt: on

S = inputs.get_val(Aircraft.Wing.AREA, 'm**2')
v = detailed_landing.get_val(Dynamic.Mission.VELOCITY, 'm/s')
# NOTE sea level; includes effect of FLOPS &TOLIN DTCT 10 DEG C
rho = 1.18391  # kg/m**3

RHV2 = 0.5 * rho * v * v * S

lift = RHV2 * lift_coeff  # N
detailed_landing.set_val(Dynamic.Vehicle.LIFT, lift, 'N')

drag = RHV2 * drag_coeff  # N
detailed_landing.set_val(Dynamic.Vehicle.DRAG, drag, 'N')

# Flops variable APRANG
apr_angle = -3.0  # deg
apr_angle_ref = abs(apr_angle)

# From FLOPS output data.
# throttle position should be (roughly) thrust / max thrust
throttle = 7233.0 / 44000

# region - landing approach to mic P3
landing_approach_to_mic_p3_user_options = AviaryValues()

landing_approach_to_mic_p3_user_options.set_val('max_duration', val=50.0, units='s')
landing_approach_to_mic_p3_user_options.set_val('time_duration_ref', val=50.0, units='s')
landing_approach_to_mic_p3_user_options.set_val('time_initial_ref', val=50.0, units='s')
landing_approach_to_mic_p3_user_options.set_val('distance_max', val=10000.0, units='ft')
landing_approach_to_mic_p3_user_options.set_val('max_velocity', val=140.0, units='kn')
landing_approach_to_mic_p3_user_options.set_val('altitude_ref', val=800.0, units='ft')

landing_approach_to_mic_p3_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

landing_approach_to_mic_p3_user_options.set_val('upper_angle_of_attack', val=12.0, units='deg')

landing_approach_to_mic_p3_user_options.set_val('angle_of_attack_ref', val=12.0, units='deg')

landing_approach_to_mic_p3_user_options.set_val('initial_height', val=600.0, units='ft')

landing_approach_to_mic_p3_initial_guesses = AviaryValues()

landing_approach_to_mic_p3_initial_guesses.set_val('time', [-42.0, 15.0], 's')
landing_approach_to_mic_p3_initial_guesses.set_val('distance', [-4000.0, -2000.0], 'ft')
landing_approach_to_mic_p3_initial_guesses.set_val('velocity', 140.0, 'kn')
landing_approach_to_mic_p3_initial_guesses.set_val('mass', detailed_landing_mass, 'lbm')
landing_approach_to_mic_p3_initial_guesses.set_val('throttle', throttle)
landing_approach_to_mic_p3_initial_guesses.set_val('altitude', [600.0, 394.0], 'ft')

landing_approach_to_mic_p3_initial_guesses.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE, [apr_angle, apr_angle], 'deg'
)

landing_approach_to_mic_p3_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.25, 'deg')

landing_approach_to_mic_p3_builder = LandingApproachToMicP3(
    'landing_approach_to_mic_p3',
    core_subsystems=default_mission_subsystems,
    subsystem_options=landing_subsystem_options,
    user_options=landing_approach_to_mic_p3_user_options,
    initial_guesses=landing_approach_to_mic_p3_initial_guesses,
)

landing_trajectory_builder.set_approach_to_mic_p3(landing_approach_to_mic_p3_builder)

ibeg = 0
iend = 26

detailed_landing_approach_to_mic_p3 = _split_aviary_values(detailed_landing, slice(ibeg, iend))
# endregion - landing approach to mic P3

# region - landing mic P3-to-obstacle
landing_mic_p3_to_obstacle_user_options = AviaryValues()

landing_mic_p3_to_obstacle_user_options.set_val('max_duration', val=50.0, units='s')
landing_mic_p3_to_obstacle_user_options.set_val('time_duration_ref', val=50.0, units='s')
landing_mic_p3_to_obstacle_user_options.set_val('time_initial_ref', val=50.0, units='s')
landing_mic_p3_to_obstacle_user_options.set_val('distance_max', val=6000.0, units='ft')
landing_mic_p3_to_obstacle_user_options.set_val('max_velocity', val=140.0, units='kn')
landing_mic_p3_to_obstacle_user_options.set_val('altitude_ref', val=400.0, units='ft')

landing_mic_p3_to_obstacle_user_options.set_val('lower_angle_of_attack', val=0.0, units='deg')

landing_mic_p3_to_obstacle_user_options.set_val('upper_angle_of_attack', val=12.0, units='deg')

landing_mic_p3_to_obstacle_user_options.set_val('angle_of_attack_ref', val=12.0, units='deg')

landing_mic_p3_to_obstacle_user_options.set_val('initial_height', val=394.0, units='ft')

landing_mic_p3_to_obstacle_initial_guesses = AviaryValues()

landing_mic_p3_to_obstacle_initial_guesses.set_val('time', [-27.0, 27.0], 's')
landing_mic_p3_to_obstacle_initial_guesses.set_val('distance', [-2000.0, 0.0], 'ft')
landing_mic_p3_to_obstacle_initial_guesses.set_val('velocity', 140.0, 'kn')
landing_mic_p3_to_obstacle_initial_guesses.set_val('mass', detailed_landing_mass, 'lbm')
landing_mic_p3_to_obstacle_initial_guesses.set_val('throttle', throttle)
landing_mic_p3_to_obstacle_initial_guesses.set_val('altitude', [394.0, 50.0], 'ft')

landing_mic_p3_to_obstacle_initial_guesses.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE, [apr_angle, apr_angle], 'deg'
)

landing_mic_p3_to_obstacle_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.25, 'deg')

landing_mic_p3_to_obstacle_builder = LandingMicP3ToObstacle(
    'landing_mic_p3_to_obstacle',
    core_subsystems=default_mission_subsystems,
    subsystem_options=landing_subsystem_options,
    user_options=landing_mic_p3_to_obstacle_user_options,
    initial_guesses=landing_mic_p3_to_obstacle_initial_guesses,
)

landing_trajectory_builder.set_mic_p3_to_obstacle(landing_mic_p3_to_obstacle_builder)

ibeg = 0
iend = 26

detailed_landing_mic_p3_to_obstacle = _split_aviary_values(detailed_landing, slice(ibeg, iend))
# endregion - mic P3 -to-obstacle

# region - landing obstacle-to-flare
landing_obstacle_user_options = AviaryValues()

landing_obstacle_user_options.set_val('max_duration', val=5.0, units='s')
landing_obstacle_user_options.set_val('distance_max', val=800.0, units='ft')
landing_obstacle_user_options.set_val('max_velocity', val=140.0, units='kn')
landing_obstacle_user_options.set_val('altitude_ref', val=50.0, units='ft')

landing_obstacle_initial_guesses = AviaryValues()

landing_obstacle_initial_guesses.set_val('time', [0.0, 4.0], 's')
landing_obstacle_initial_guesses.set_val('distance', [0.0, 800.0], 'ft')
landing_obstacle_initial_guesses.set_val('velocity', 140.0, 'kn')
landing_obstacle_initial_guesses.set_val('mass', detailed_landing_mass, 'lbm')
landing_obstacle_initial_guesses.set_val('throttle', throttle)
landing_obstacle_initial_guesses.set_val('altitude', [50.0, 15.0], 'ft')

landing_obstacle_initial_guesses.set_val(
    Dynamic.Mission.FLIGHT_PATH_ANGLE, [apr_angle, apr_angle], 'deg'
)

landing_obstacle_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 5.2, 'deg')

landing_obstacle_builder = LandingObstacleToFlare(
    'landing_obstacle',
    core_subsystems=default_mission_subsystems,
    subsystem_options=landing_subsystem_options,
    user_options=landing_obstacle_user_options,
    initial_guesses=landing_obstacle_initial_guesses,
)

landing_trajectory_builder.set_obstacle_to_flare(landing_obstacle_builder)

ibeg = iend
iend = ibeg + 20
detailed_landing_obstacle = _split_aviary_values(detailed_landing, slice(ibeg, iend))
# endregion - landing obstacle-to-flare

# region - landing flare-to-touchdown
landing_flare_user_options = AviaryValues()

landing_flare_user_options.set_val('max_duration', val=7.0, units='s')
landing_flare_user_options.set_val('time_duration_ref', val=7.0, units='s')
landing_flare_user_options.set_val('time_initial_ref', val=4.0, units='s')
landing_flare_user_options.set_val('distance_max', val=1000.0, units='ft')
landing_flare_user_options.set_val('max_velocity', val=140.0, units='kn')
landing_flare_user_options.set_val('altitude_ref', val=15.0, units='ft')

landing_flare_user_options.set_val('lower_angle_of_attack', val=5.2, units='deg')
landing_flare_user_options.set_val('upper_angle_of_attack', val=12.0, units='deg')
landing_flare_user_options.set_val('angle_of_attack_ref', val=12.0, units='deg')

landing_flare_initial_guesses = AviaryValues()

landing_flare_initial_guesses.set_val('time', [4.0, 6.0], 's')
landing_flare_initial_guesses.set_val('distance', [800.0, 1000.0], 'ft')
landing_flare_initial_guesses.set_val('velocity', 140.0, 'kn')
landing_flare_initial_guesses.set_val('mass', detailed_landing_mass, 'lbm')
landing_flare_initial_guesses.set_val('throttle', [throttle, throttle * 4 / 7])
landing_flare_initial_guesses.set_val('altitude', [15.0, 0.0], 'ft')
landing_flare_initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [apr_angle, 0.0], 'deg')
landing_flare_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [5.2, 7.5], 'deg')

landing_flare_builder = LandingFlareToTouchdown(
    'landing_flare',
    core_subsystems=default_mission_subsystems,
    subsystem_options=landing_subsystem_options,
    user_options=landing_flare_user_options,
    initial_guesses=landing_flare_initial_guesses,
)

landing_trajectory_builder.set_flare_to_touchdown(landing_flare_builder)

ibeg = iend
iend = ibeg + 2
detailed_landing_flare = _split_aviary_values(detailed_landing, slice(ibeg, iend))
# endregion - landing flare-to-touchdown

# region touchdown-to-nose-down
landing_touchdown_user_options = AviaryValues()

landing_touchdown_user_options.set_val('max_duration', val=10.0, units='s')
landing_touchdown_user_options.set_val('time_duration_ref', val=10.0, units='s')
landing_touchdown_user_options.set_val('time_initial_ref', val=6.0, units='s')
landing_touchdown_user_options.set_val('distance_max', val=3000.0, units='ft')
landing_touchdown_user_options.set_val('max_velocity', val=140.0, units='kn')
landing_touchdown_user_options.set_val('max_angle_of_attack', val=8.0, units='deg')

landing_touchdown_initial_guesses = AviaryValues()

landing_touchdown_initial_guesses.set_val('time', [6.0, 9.0], 's')
landing_touchdown_initial_guesses.set_val('distance', [1000.0, 1400.0], 'ft')
landing_touchdown_initial_guesses.set_val('velocity', [140.0, 135.0], 'kn')
landing_touchdown_initial_guesses.set_val('mass', detailed_landing_mass, 'lbm')
landing_touchdown_initial_guesses.set_val('throttle', 0.0)
landing_touchdown_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [7.5, 0.0], 'deg')

landing_touchdown_builder = LandingTouchdownToNoseDown(
    'landing_touchdown',
    core_subsystems=default_mission_subsystems,
    subsystem_options=landing_subsystem_options,
    user_options=landing_touchdown_user_options,
    initial_guesses=landing_touchdown_initial_guesses,
)

landing_trajectory_builder.set_touchdown_to_nose_down(landing_touchdown_builder)

ibeg = iend
iend = ibeg + 5
detailed_landing_touchdown = _split_aviary_values(detailed_landing, slice(ibeg, iend))
# endregion touchdown-to-nose-down

# region nose-down-to-fullstop
landing_fullstop_user_options = AviaryValues()

landing_fullstop_user_options.set_val('max_duration', val=30.0, units='s')
landing_fullstop_user_options.set_val('time_duration_ref', val=30.0, units='s')
landing_fullstop_user_options.set_val('time_initial_ref', val=14.0, units='s')
landing_fullstop_user_options.set_val('distance_max', val=4400.0, units='ft')
landing_fullstop_user_options.set_val('max_velocity', val=140.0, units='kn')

landing_fullstop_initial_guesses = AviaryValues()

landing_fullstop_initial_guesses.set_val('time', [9.0, 29.0], 's')
landing_fullstop_initial_guesses.set_val('distance', [1400.0, 3500.0], 'ft')
landing_fullstop_initial_guesses.set_val('velocity', [135.0, 0.01], 'kn')
landing_fullstop_initial_guesses.set_val('mass', detailed_landing_mass, 'lbm')
landing_fullstop_initial_guesses.set_val('throttle', 0.0)
landing_fullstop_initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, 'deg')

landing_fullstop_builder = LandingNoseDownToStop(
    'landing_fullstop',
    core_subsystems=default_mission_subsystems,
    subsystem_options=landing_subsystem_options_spoilers,
    user_options=landing_fullstop_user_options,
    initial_guesses=landing_fullstop_initial_guesses,
)

landing_trajectory_builder.set_nose_down_to_stop(landing_fullstop_builder)

ibeg = iend
iend = ibeg + 18
detailed_landing_fullstop = _split_aviary_values(detailed_landing, slice(ibeg, iend))
# endregion nose-down-to-fullstop

# endregion - detailed landing
