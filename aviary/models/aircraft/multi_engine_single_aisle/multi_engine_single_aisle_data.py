import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.enums import EquationsOfMotion, LegacyCode
from aviary.variable_info.variables import Aircraft, Mission, Settings

MultiEngineSingleAisle = {}
inputs = MultiEngineSingleAisle['inputs'] = AviaryValues()
outputs = MultiEngineSingleAisle['outputs'] = AviaryValues()

# Overall Aircraft
# ---------------------------
inputs.set_val(Aircraft.Design.BASE_AREA, 0.0, 'ft**2')
inputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN_SCALER, 0.00514)
inputs.set_val(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 0.84)
inputs.set_val(Mission.Design.GROSS_MASS, 174200.0, 'lbm')
inputs.set_val(Aircraft.Design.USE_ALT_MASS, False)
inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)
inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)
inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)
inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 0.945)

# Air Conditioning
# ---------------------------
inputs.set_val(Aircraft.AirConditioning.MASS_SCALER, 1.0)

# Anti-Icing
# ---------------------------
inputs.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)

# APU
# ---------------------------
inputs.set_val(Aircraft.APU.MASS_SCALER, 1.0)

# Avionics
# ---------------------------
inputs.set_val(Aircraft.Avionics.MASS_SCALER, 1.0)

# Canard
# ---------------------------
inputs.set_val(Aircraft.Canard.AREA, 0.0, 'ft**2')
inputs.set_val(Aircraft.Canard.ASPECT_RATIO, 0.0)
inputs.set_val(Aircraft.Canard.THICKNESS_TO_CHORD, 0.0)

# Crew and Payload
# ---------------------------
inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 0)
inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 12)
inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 162, units='unitless')
inputs.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 150)
inputs.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 0)
inputs.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 12)
inputs.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 162, units='unitless')
inputs.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, 150)

inputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 35.0, 'lbm')
inputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, 5)
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, 2)
inputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, 1)
inputs.set_val(Aircraft.CrewPayload.MISC_CARGO, 4077.0, 'lbm')
inputs.set_val(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 165.0, 'lbm')
inputs.set_val(Aircraft.CrewPayload.WING_CARGO, 0.0, 'lbm')

# Electrical
# ---------------------------
inputs.set_val(Aircraft.Electrical.MASS_SCALER, 1.0)

# Fins
# ---------------------------
inputs.set_val(Aircraft.Fins.AREA, 0.0, 'ft**2')
inputs.set_val(Aircraft.Fins.NUM_FINS, 0)
inputs.set_val(Aircraft.Fins.TAPER_RATIO, 10.0)
inputs.set_val(Aircraft.Fins.MASS, 1.0, 'lbm')
inputs.set_val(Aircraft.Fins.MASS_SCALER, 1.0)

# Fuel
# ---------------------------
inputs.set_val(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0, 'lbm')
inputs.set_val(Aircraft.Fuel.DENSITY, 6.7, 'lbm/galUS')
inputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Fuel.NUM_TANKS, 7)
inputs.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 46063.0, 'lbm')
inputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, 1.0)

# Furnishings
# ---------------------------
inputs.set_val(Aircraft.Furnishings.MASS_SCALER, 1.0)

# Fuselage
# ---------------------------
inputs.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)
inputs.set_val(Aircraft.Fuselage.LENGTH, 124.75, 'ft')
inputs.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 98.5, 'ft')
inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 13.0208, 'ft')
inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.33, 'ft')
inputs.set_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR, False)
inputs.set_val(Aircraft.Fuselage.PLANFORM_AREA, 124.75 * 12.33, 'ft**2')
inputs.set_val(Aircraft.Fuselage.MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Fuselage.WETTED_AREA, 4142.317, 'ft**2')  # Override
inputs.set_val(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0)

# Horizontal Tail
# ---------------------------
inputs.set_val(Aircraft.HorizontalTail.AREA, 407.335370699457, 'ft**2')
inputs.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, 5.444)
inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 0.3008)
inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1195)
inputs.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0.0)
inputs.set_val(Aircraft.HorizontalTail.MASS_SCALER, 1.0)
inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA, 707.706, 'ft**2')  # Override
inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA_SCALER, 1.0)

# Hydraulics
# ---------------------------
inputs.set_val(Aircraft.Hydraulics.SYSTEM_PRESSURE, 3000.0, 'psi')
inputs.set_val(Aircraft.Hydraulics.MASS_SCALER, 1.0)

# Instruments
# ---------------------------
inputs.set_val(Aircraft.Instruments.MASS_SCALER, 1.0)

# Landing Gear
# ---------------------------
inputs.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 84.0, 'inch')
inputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 58.8, 'inch')
inputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, 1.0)

# Nacelle
# ---------------------------
inputs.set_val(Aircraft.Nacelle.AVG_DIAMETER, 7.0, 'ft')
inputs.set_val(Aircraft.Nacelle.AVG_LENGTH, 11.65, 'ft')
inputs.set_val(Aircraft.Nacelle.MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, 1.0)

# Paint
# ---------------------------
inputs.set_val(Aircraft.Paint.MASS_PER_UNIT_AREA, 0.07, 'lbm/ft**2')

# Propulsion and Engine
# ---------------------------
inputs.set_val(Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0)

# Engine 1: turbofan_28k
filename = get_path('models/engines/turbofan_28k.csv')

engine_1_inputs = AviaryValues()
engine_1_inputs.set_val(Aircraft.Engine.DATA_FILE, filename)
engine_1_inputs.set_val(Aircraft.Engine.MASS, 7400, 'lbm')
engine_1_inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 7400, 'lbm')
# engine_1_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 28928.1/2, 'lbf')
engine_1_inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 0.3837186)
# engine_1_inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 28928.1, 'lbf')
engine_1_inputs.set_val(Aircraft.Engine.NUM_ENGINES, 2)
engine_1_inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 0)
engine_1_inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)
engine_1_inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)
engine_1_inputs.set_val(Aircraft.Engine.WING_LOCATIONS, 15.8300 / (117.83 / 2))
engine_1_inputs.set_val(Aircraft.Engine.SCALE_MASS, True)
engine_1_inputs.set_val(Aircraft.Engine.MASS_SCALER, 1.15)
engine_1_inputs.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
engine_1_inputs.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
engine_1_inputs.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
engine_1_inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
engine_1_inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 0.0)
engine_1_inputs.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h')
engine_1_inputs.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0)
engine_1_inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
engine_1_inputs.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
engine_1_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
engine_1_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
engine_1_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
engine_1_inputs.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
engine_1_inputs.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')
engine_1_inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)

# Engine 2:
filename = get_path('models/engines/turbofan_22k.csv')

engine_2_inputs = AviaryValues()
engine_2_inputs.set_val(Aircraft.Engine.DATA_FILE, filename)
engine_2_inputs.set_val(Aircraft.Engine.MASS, 6293.8, 'lbm')
engine_2_inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 6293.8, 'lbm')
# engine_2_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 22200.5/2, 'lbf')
engine_2_inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 0.65151911)
# engine_2_inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 22200.5, 'lbf')
engine_2_inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)
engine_2_inputs.set_val(Aircraft.Engine.NUM_ENGINES, 2)
engine_2_inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 0)
engine_2_inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)
engine_2_inputs.set_val(Aircraft.Engine.WING_LOCATIONS, 0.289682918)
engine_2_inputs.set_val(Aircraft.Engine.SCALE_MASS, True)
engine_2_inputs.set_val(Aircraft.Engine.MASS_SCALER, 1.15)
engine_2_inputs.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
engine_2_inputs.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
engine_2_inputs.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
engine_2_inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
engine_2_inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
engine_2_inputs.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lb/h')
engine_2_inputs.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0)
engine_2_inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
engine_2_inputs.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
engine_2_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
engine_2_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
engine_2_inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
engine_2_inputs.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
engine_2_inputs.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')
engine_2_inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)

# Vertical Tail
# ---------------------------
inputs.set_val(Aircraft.VerticalTail.NUM_TAILS, 1)
inputs.set_val(Aircraft.VerticalTail.AREA, 284.499779284585, 'ft**2')
inputs.set_val(Aircraft.VerticalTail.ASPECT_RATIO, 2.2262)
inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 0.21082082638898)
inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.137459440381375)
inputs.set_val(Aircraft.VerticalTail.MASS_SCALER, 1.0)
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA, 589.35, 'ft**2')  # Override
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA_SCALER, 1.0)

# Wing
# ---------------------------
inputs.set_val(Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0)
inputs.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 1.87)
inputs.set_val(Aircraft.Wing.AREA, 1341.0, 'ft**2')
inputs.set_val(Aircraft.Wing.ASPECT_RATIO, 9.42519)
inputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Wing.CHORD_PER_SEMISPAN_DIST, np.array([0.4441, 0.2313, 0.0729]))
inputs.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 0.0)
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333)
inputs.set_val(Aircraft.Wing.DETAILED_WING, True)
inputs.set_val(Aircraft.Wing.GLOVE_AND_BAT, 0.0, 'ft**2')
inputs.set_val(Aircraft.Wing.INPUT_STATION_DIST, np.array([0.0, 0.3238, 1.0]))
inputs.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0)
inputs.set_val(Aircraft.Wing.LOAD_FRACTION, 1.0)
inputs.set_val(Aircraft.Wing.LOAD_PATH_SWEEP_DIST, np.array([0.0, 22.0]), 'deg')
inputs.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.015)
inputs.set_val(Aircraft.Wing.MISC_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 100)
inputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Wing.SPAN, 112.57, 'ft')
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 1.35)
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, False)
inputs.set_val(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0)
inputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1.0)
inputs.set_val(Aircraft.Wing.SWEEP, 25.03, 'deg')
inputs.set_val(Aircraft.Wing.TAPER_RATIO, 0.237343146184852)
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.131732727515702)
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_DIST, np.array([0.145, 0.115, 0.104]))
inputs.set_val(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75)
inputs.set_val(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0)
inputs.set_val(Aircraft.Wing.MASS, 15548, 'lbm')
inputs.set_val(Aircraft.Wing.WETTED_AREA, 2423.02, 'ft**2')  # Override 2423.02
inputs.set_val(Aircraft.Wing.WETTED_AREA_SCALER, 1.0)


# Mission
# ---------------------------
inputs.set_val(Mission.Summary.CRUISE_MACH, 0.785)  # was 0.82
inputs.set_val(Mission.Design.RANGE, 2960.0, 'NM')
inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)
inputs.set_val(Mission.Constraints.MAX_MACH, 0.82)
# TODO investigate the origin of these values (taken from benchmark tests)
# TODO: where should this get connected from?
inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, 577, 'lbm')

# region TODO: should this come from aero?
inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, 3)
inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2)
inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG, 17.35)
# endregion TODO: should this come from aero?

# TODO: should this be a user input or should it be hard coded somewhere assuming it will
# never change?
inputs.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.0175)
# lbf TODO: where should this get connected from?
inputs.set_val(Mission.Design.THRUST_TAKEOFF_PER_ENG, 24555.5, 'lbf')

# Settings
# ---------------------------
inputs.set_val(Settings.EQUATIONS_OF_MOTION, EquationsOfMotion.HEIGHT_ENERGY)
inputs.set_val(Settings.AERODYNAMICS_METHOD, LegacyCode.FLOPS)
inputs.set_val(Settings.MASS_METHOD, LegacyCode.FLOPS)
inputs.set_val(Settings.VERBOSITY, 0)

# ---------------------------
#          OUTPUTS
# ---------------------------

outputs.set_val(Aircraft.Design.EMPTY_MASS, 92023.0, 'lbm')
outputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN, 0.0, 'lbm')
outputs.set_val(Mission.Summary.OPERATING_MASS, 97992.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.MASS, 16118.0, 'lbm')
outputs.set_val(Aircraft.Design.STRUCTURE_MASS, 50736.0, 'lbm')
outputs.set_val(Aircraft.Design.SYSTEMS_EQUIP_MASS, 25169.0, 'lbm')
outputs.set_val(Aircraft.Design.TOTAL_WETTED_AREA, 8275.86, 'ft**2')
outputs.set_val(Mission.Summary.ZERO_FUEL_MASS, 135848.0, 'lbm')
outputs.set_val(Mission.Summary.FUEL_MASS, 45352.0, 'lbm')

outputs.set_val(Aircraft.AirConditioning.MASS, 1602.0, 'lbm')

outputs.set_val(Aircraft.AntiIcing.MASS, 208.85, 'lbm')

outputs.set_val(Aircraft.APU.MASS, 1142.0, 'lbm')

outputs.set_val(Aircraft.Avionics.MASS, 1652.6, 'lbm')

outputs.set_val(Aircraft.Canard.CHARACTERISTIC_LENGTH, 0.0, 'ft')
outputs.set_val(Aircraft.Canard.FINENESS, 0.0)
outputs.set_val(Aircraft.Canard.WETTED_AREA, 0.0, 'ft**2')
outputs.set_val(Aircraft.Canard.MASS, 0.0, 'lbm')

outputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS, 7436.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.CARGO_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS, 1400.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS, 450.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, 465.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 3022.74805809, 'lbm')
outputs.set_val(Aircraft.CrewPayload.PASSENGER_MASS, 30420.0, 'lbm')
outputs.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 37856.0, 'lbm')

outputs.set_val(Aircraft.Electrical.MASS, 2464.0, 'lbm')

outputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS, 669.58, 'lbm')
outputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 501.3, 'lbm')

outputs.set_val(Aircraft.Fins.MASS, 0.0, 'lbm')

outputs.set_val(Aircraft.Furnishings.MASS, 15517.0, 'lbm')

avg_diameter = 12.75
avg_diameter_units = 'ft'
outputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, avg_diameter, avg_diameter_units)
outputs.set_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 128.0, 'ft')
outputs.set_val(
    Aircraft.Fuselage.CROSS_SECTION,
    np.pi * (avg_diameter / 2.0) ** 2.0,
    f'{avg_diameter_units}**2',
)
outputs.set_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 0.108207)
outputs.set_val(Aircraft.Fuselage.FINENESS, 10.0392)
outputs.set_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER, 10.039216)
outputs.set_val(Aircraft.Fuselage.MASS, 18357.0, 'lbm')

outputs.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 7.69, 'ft')
outputs.set_val(Aircraft.HorizontalTail.FINENESS, 0.1250)
outputs.set_val(Aircraft.HorizontalTail.MASS, 1831.0, 'lbm')

outputs.set_val(Aircraft.Hydraulics.MASS, 1086.7, 'lbm')

outputs.set_val(Aircraft.Instruments.MASS, 601, 'lbm')

outputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS, 7910.32, 'lbm')
outputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS, 870.59, 'lbm')

outputs.set_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.array([12.30]), 'ft')
outputs.set_val(Aircraft.Nacelle.FINENESS, np.array([1.5491]))
outputs.set_val(Aircraft.Nacelle.MASS, 1971.4, 'lbm')

nacelle_wetted_area = np.array([273.45])
nacelle_wetted_area_units = 'ft**2'
outputs.set_val(Aircraft.Nacelle.WETTED_AREA, nacelle_wetted_area, nacelle_wetted_area_units)

outputs.set_val(
    Aircraft.Nacelle.TOTAL_WETTED_AREA,
    2 * nacelle_wetted_area,
    nacelle_wetted_area_units,
)

outputs.set_val(Aircraft.Paint.MASS, 306.2, 'lbm')

outputs.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 51128.6, 'lbf')

outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 4)

engine_ctrls_mass = 88.44
engine_ctrls_mass_units = 'lbm'
outputs.set_val(Aircraft.Engine.CONTROLS_MASS, engine_ctrls_mass, engine_ctrls_mass_units)
outputs.set_val(
    Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
    engine_ctrls_mass,
    engine_ctrls_mass_units,
)

outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, 130.23, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 4)

outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 0)

outputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, np.array([28928.1 / 2, 22200.5 / 2]), 'lbf')
outputs.set_val(Aircraft.Engine.MASS, 14800 / 2, 'lbm')
outputs.set_val(Aircraft.Engine.POD_MASS, 9000, 'lbm')
outputs.set_val(Aircraft.Engine.ADDITIONAL_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_MISC_MASS, 648.83, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 560.39, 'lbm')

outputs.set_val(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, 0, 'lbm')
outputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS, 0, 'lbm')

# outputs.set_val(
#     Aircraft.Propulsion.TOTAL_ENGINE_MASS,
#     engine_mass * num_engines, engine_mass_units)

outputs.set_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 12.74, 'ft')
outputs.set_val(Aircraft.VerticalTail.FINENESS, 0.1195)
outputs.set_val(Aircraft.VerticalTail.MASS, 1221.8, 'lbm')

outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR, 11.5918)
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS, 8184.8, 'lbm')
outputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 10.49, 'ft')
# Not in FLOPS output; calculated from inputs.
outputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 137, 'ft**2')
outputs.set_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 0.967333)
outputs.set_val(Aircraft.Wing.FINENESS, 0.1300)
outputs.set_val(Aircraft.Wing.MISC_MASS, 1668.3, 'lbm')
outputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS, 4998.8, 'lbm')
outputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS, 894, 'lbm')

outputs.set_val(Aircraft.Wing.MASS, 18268, 'lbm')

outputs.set_val(Mission.Design.MACH, 0.800)
outputs.set_val(Mission.Design.LIFT_COEFFICIENT, 0.568)
