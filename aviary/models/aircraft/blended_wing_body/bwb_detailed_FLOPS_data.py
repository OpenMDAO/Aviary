import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.enums import AircraftTypes, EquationsOfMotion, LegacyCode
from aviary.variable_info.variables import Aircraft, Mission, Settings

BWBdetailedFLOPS = {}
inputs = BWBdetailedFLOPS['inputs'] = AviaryValues()
outputs = BWBdetailedFLOPS['outputs'] = AviaryValues()

# Overall Aircraft
# ---------------------------
inputs.set_val(
    Aircraft.Design.BASE_AREA, 0.0, 'ft**2'
)  # SBASE not in bwb.in, set to 0.0 as all others
inputs.set_val(
    Aircraft.Design.EMPTY_MASS_MARGIN_SCALER, 0.0
)  # EWMARG not in bwb.in, set to default
inputs.set_val(Mission.Design.GROSS_MASS, 874099.0, 'lbm')  # DGW in bwb.in
inputs.set_val(Aircraft.Design.USE_ALT_MASS, False)
inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)  # FCDI in bwb.in
inputs.set_val(
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0
)  # FCDSUB not in bwb.in, set to default
inputs.set_val(
    Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0
)  # FCDSUP not in bwb.in, set to default
inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)  # FCDO in bwb.in
inputs.set_val(Aircraft.Design.TYPE, AircraftTypes.BLENDED_WING_BODY)
inputs.set_val(Mission.Design.LIFT_COEFFICIENT, -1.0)  # FCLDES

# Air Conditioning
# ---------------------------
inputs.set_val(Aircraft.AirConditioning.MASS_SCALER, 1.0)  # WAC not in bwb.in

# Anti-Icing
# ---------------------------
inputs.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)  # WAI not in bwb.in

# APU
# ---------------------------
inputs.set_val(Aircraft.APU.MASS_SCALER, 1.0)  # WAPU not in bwb.in, set to Aviary default

# Avionics
# ---------------------------
inputs.set_val(Aircraft.Avionics.MASS_SCALER, 1.0)  # WAVONC in bwb.in

# Canard
# ---------------------------
inputs.set_val(Aircraft.Canard.AREA, 0.0, 'ft**2')  # SCAN not in bwb.in, set to default
inputs.set_val(Aircraft.Canard.ASPECT_RATIO, 0.0)  # ARCAN not in bwb.in, set to default
inputs.set_val(Aircraft.Canard.THICKNESS_TO_CHORD, 0.0)  # TCCAN not in bwb.in, default to TCHT

# Crew and Payload
# ---------------------------
inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 100)  # NPB in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 28)  # NPF in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 468, units='unitless')  # NPB+NPF+NPT
inputs.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 340)  # NPT in bwb.in
inputs.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 100)  # NPB in bwb.in
inputs.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 28)  # NPF in bwb.in
inputs.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 468, units='unitless')  # sum of three classes
inputs.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, 340)  # NPT in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 4)  # NBABR in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 4)  # NFABR in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)  # NTABR in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS, 39, 'inch')  # BPITCH in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 61, 'inch')  # FPITCH in bwb.in
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 32, 'inch')  # TPITCH in bwb.in

inputs.set_val(
    Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, 1.0
)  # WCON not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, 22)  # NSTU in bwb.in
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, 2)  # NFLCR in bwb.in
inputs.set_val(
    Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER, 1.0
)  # WFLCRB not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, 2)  # NGALC is computed
inputs.set_val(
    Aircraft.CrewPayload.MISC_CARGO, 0.0, 'lbm'
)  # CARGOF not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER, 1.0
)  # WSTUAB not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, 1.0
)  # WSRV not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.CrewPayload.MASS_PER_PASSENGER, 165.0, 'lbm'
)  # WPPASS not in bwb.in, set to default
inputs.set_val(
    Aircraft.CrewPayload.WING_CARGO, 0.0, 'lbm'
)  # CARGOW not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 44.0, 'lbm'
)  # BPP not in bwb.in, set to default

# Electrical
# ---------------------------
inputs.set_val(Aircraft.Electrical.MASS_SCALER, 1.0)  # WELEC not in bwb.in, set to Aviary default

# Fins
# ---------------------------
inputs.set_val(Aircraft.Fins.AREA, 184.89, 'ft**2')  # SFIN in bwb.in
inputs.set_val(Aircraft.Fins.NUM_FINS, 2)  # NFIN in bwb.in
inputs.set_val(Aircraft.Fins.TAPER_RATIO, 0.464)  # TRFIN in bwb.in
# inputs.set_val(Aircraft.Fins.MASS, 0.0, 'lbm')  # WFIN not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Fins.MASS_SCALER, 1.0)  # FRFIN not in bwb.in, set to Aviary default

# Fuel
# ---------------------------
inputs.set_val(
    Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0, 'lbm'
)  # FULAUX not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Fuel.DENSITY, 6.7, 'lbm/galUS'
)  # FULDEN not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0
)  # WFSYS not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0, 'lbm'
)  # FULFMX not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Fuel.NUM_TANKS, 7)  # NTANK not in bwb.in, set to default
inputs.set_val(
    Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, 1.0
)  # WUF not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT, False
)  # IFUFU not in bwb.in, set to default

# Furnishings
# ---------------------------
inputs.set_val(Aircraft.Furnishings.MASS_SCALER, 1.0)  # WFURN in bwb.in

# Fuselage
# ---------------------------
inputs.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)  # NFUSE in bwb.in
# inputs.set_val(Aircraft.Fuselage.LENGTH, 137.5, 'ft')  # XL in bwb.in
inputs.set_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR, False)  # CARGF in bwb.in
inputs.set_val(Aircraft.Fuselage.MASS_SCALER, 1.0)  # FRFU in bwb.in
# inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 64.58, 'ft')  # WF in bwb.in
# inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 15.125, 'ft')  # DF in bwb.in, but should not be here
# inputs.set_val(Aircraft.Fuselage.CABIN_AREA, 0, 'ft**2')  # ACABIN in bwb.in, but should not be here
# inputs.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 0, 'ft')  # XLP in bwb.in, but should not be here
inputs.set_val(
    Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 45.0, 'deg'
)  # SWPLE, not in bwb.in, default 45.0 degrees


# inputs.set_val(Aircraft.Fuselage.WETTED_AREA, 0.0, 'ft**2')  # For BWB, see _BWBFuselage()
inputs.set_val(
    Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0
)  # SWETF not in bwb.in, set to Aviary default

# Horizontal Tail
# ---------------------------
inputs.set_val(Aircraft.HorizontalTail.AREA, 0.0, 'ft**2')  # SHT in bwb.in
inputs.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, 0.0)  # SHT in bwb.in
inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 0.0)  # TRHT in bwb.in
inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.0)  # TCHT in bwb.in
# inputs.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0.0)  # HHT not in bwb.in,
inputs.set_val(Aircraft.HorizontalTail.MASS_SCALER, 1.0)  # SHT in bwb.in
# inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA, 592.65, 'ft**2')  # SWTHT not in bwb.in
inputs.set_val(
    Aircraft.HorizontalTail.WETTED_AREA_SCALER, 1.0
)  # SWETH not in bwb.in, set to Aviary default
# inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0.0)  # SWPHT in bwb.in, but should not be here

# Hydraulics
# ---------------------------
inputs.set_val(
    Aircraft.Hydraulics.SYSTEM_PRESSURE, 3000.0, 'psi'
)  # HYDPR not in bwb.in, set to default
inputs.set_val(Aircraft.Hydraulics.MASS_SCALER, 1.0)  # WHYD in bwb.in

# Instruments
# ---------------------------
inputs.set_val(Aircraft.Instruments.MASS_SCALER, 1.0)  # WIN not in bwb.in, set to Aviary default

# Landing Gear
# ---------------------------
inputs.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 85.0, 'inch')  # XMLG in bwb.in
inputs.set_val(
    Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, 1.0
)  # FRLGM not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 87.0, 'inch')  # XNLG in bwb.in
inputs.set_val(
    Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, 1.0
)  # FRLGN not in bwb.in, set to Aviary default

# Nacelle
# ---------------------------
inputs.set_val(Aircraft.Nacelle.AVG_DIAMETER, 12.608, 'ft')  # DNAC in bwb.in
inputs.set_val(Aircraft.Nacelle.AVG_LENGTH, 17.433, 'ft')  # XNAC in bwb.in
inputs.set_val(Aircraft.Nacelle.MASS_SCALER, 0.0)  # FRNA in bwb.in
inputs.set_val(
    Aircraft.Nacelle.WETTED_AREA_SCALER, 1.0
)  # SWETN not in bwb.in, set to Aviary default

# Paint
# ---------------------------
inputs.set_val(Aircraft.Paint.MASS_PER_UNIT_AREA, 0.0, 'lbm/ft**2')  # WPAINT not in bwb.in,

# Propulsion and Engine
# ---------------------------
inputs.set_val(
    Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, 1.0
)  # WOIL not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 0.0)  # WPMSC in bwb.in

filename = get_path('models/engines/PAX300_baseline_ENGDEK.csv')

inputs.set_val(Aircraft.Engine.DATA_FILE, filename)
# inputs.set_val(Aircraft.Engine.MASS, 7400, 'lbm')  # not in bwb.in, not a FLOPS variable
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 22017, 'lbm')  # WENG in bwb.in
inputs.set_val(
    Aircraft.Engine.SCALED_SLS_THRUST, 70000.0, 'lbf'
)  # THRUST in bwb.in [70000, 1, 0, 0, 0, 0]
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 86459.2, 'lbf')  # THRSO in bwb.in
inputs.set_val(
    Aircraft.Engine.NUM_ENGINES, np.array([3])
)  # not in bwb.in, not a FLOPS variable, set to NEW+NEF
inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 3)  # NEF in bwb.in
inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, 0)  # NEW in bwb.in
inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)  # WTHR in bwb.in
inputs.set_val(Aircraft.Engine.WING_LOCATIONS, 0)  # ETAE not in bwb.in, not a FLOPS variable
inputs.set_val(
    Aircraft.Engine.SCALE_FACTOR, 0.8096304384
)  # not in bwb.in, not a FLOPS variable, set to THRUST/THRSO
inputs.set_val(
    Aircraft.Engine.SCALE_MASS, True
)  # not in bwb.in, not a FLOPS variable, set to Aviary default
inputs.set_val(Aircraft.Engine.MASS_SCALER, 1.0)  # EEXP in bwb.in
inputs.set_val(
    Aircraft.Engine.SCALE_PERFORMANCE, True
)  # not in bwb.in, not a FLOPS variable, set to Aviary default
inputs.set_val(
    Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0
)  # FFFSUB not in bwb.in, set to default
inputs.set_val(
    Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0
)  # FFFSUP not in bwb.in, set to default
inputs.set_val(
    Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0
)  # DFFAC not in bwb.in, set to default
inputs.set_val(
    Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 0.0
)  # FFFAC not in bwb.in, set to default
inputs.set_val(
    Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h'
)  # FLEAK not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0
)  # WPMISC not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)  # IDLE in bwb.in
inputs.set_val(
    Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False
)  # NONEG not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0
)  # not in bwb.in, not a FLOPS variable, set to Aviary default
inputs.set_val(
    Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0
)  # FIDMAX not in bwb.in, set to default
inputs.set_val(
    Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08
)  # FIDMIN not in bwb.in, set to default
inputs.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)  # IGEO not in bwb.in, set to default
inputs.set_val(
    Aircraft.Engine.INTERPOLATION_METHOD, 'slinear'
)  # not in bwb.in, not a FLOPS variable, set to Aviary default

# Vertical Tail
# ---------------------------
inputs.set_val(Aircraft.VerticalTail.NUM_TAILS, 0)  # NVERT in bwb.in
inputs.set_val(
    Aircraft.VerticalTail.AREA, 0.01, 'ft**2'
)  # SVT not in bwb.in, set to 0.01, not default
inputs.set_val(
    Aircraft.VerticalTail.ASPECT_RATIO, 0.01
)  # ARVT not in bwb.in, set to 0.01, not default ARHT/2 = 0/2
inputs.set_val(
    Aircraft.VerticalTail.TAPER_RATIO, 0.0
)  # TRVT not in bwb.in, set to default TRHT = 0
inputs.set_val(
    Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.0
)  # TCVT not in bwb.in, set to default TCHT = 0
inputs.set_val(Aircraft.VerticalTail.MASS_SCALER, 1.0)  # FRVT in bwb.in
# inputs.set_val(Aircraft.VerticalTail.WETTED_AREA, 581.13, 'ft**2')  # not in bwb.in, not a FLOPS variable
inputs.set_val(
    Aircraft.VerticalTail.WETTED_AREA_SCALER, 1.0
)  # SWETV not in bwb.in, set to Aviary default

# Wing
# ---------------------------
inputs.set_val(Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0)  # FAERT in bwb.in
inputs.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 2.0)  # AITEK in bwb.in
# inputs.set_val(Aircraft.Wing.AREA, 7621.66, 'ft**2')  # SW in bwb.in, always output for BWB
inputs.set_val(Aircraft.Wing.ASPECT_RATIO, 7.557)  # AR in bwb.in
inputs.set_val(Aircraft.Wing.ASPECT_RATIO_REF, 7.557)  # ARREF not in bwb.in, default to AR
inputs.set_val(
    Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0
)  # FRWI1 not in bwb.in, set to Aviary default

inputs.set_val(
    Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
    np.array(
        [
            -1.0,
            58.03,
            0.4491,
            0.3884,
            0.3317,
            0.2886,
            0.2537,
            0.2269,
            0.2121,
            0.1983,
            0.1843,
            0.1704,
            0.1565,
            0.1426,
            0.1287,
        ]
    ),
)  # CHD
inputs.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 1.0)  # FCOMP in bwb.in
# inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 137, 'ft**2')  # not in bwb.in, not a FLOPS variable
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333)  # FLAPR in bwb.in
inputs.set_val(Aircraft.Wing.DETAILED_WING, True)  # for BWB, always true
inputs.set_val(Aircraft.Wing.GLOVE_AND_BAT, 121.05, 'ft**2')  # GLOV in bwb.in

inputs.set_val(
    Aircraft.Wing.INPUT_STATION_DIST,
    np.array([0.0, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499, 0.7, 0.75, 0.8, 0.85, 0.8999, 0.95, 1]),
)  # ETAW

inputs.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0)  # PDIST not in bwb.in, set to default
inputs.set_val(Aircraft.Wing.LOAD_FRACTION, 1.0)  # PCTL not in bwb.in, set to default

inputs.set_val(
    Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
    np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 42.9, 42.9, 42.9, 42.9, 42.9, 42.9]),
    'deg',
)  # SWL
inputs.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 2.0)  # CAM in bwb.in
inputs.set_val(Aircraft.Wing.MISC_MASS_SCALER, 1.0)  # FRWI3 not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Wing.NUM_INTEGRATION_STATIONS, 50
)  # NSTD not in bwb.in, set to Aviary default
inputs.set_val(
    Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0
)  # FRWI2 not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333)  # FLAPR in bwb.in
inputs.set_val(Aircraft.Wing.SPAN, 253.720756, 'ft')  # SPAN not in bwb.in, SPAN = WF+OSSPAN*2
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, False)  # MIKE not in bwb.in, set to default
inputs.set_val(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0)  # FSTRT in bwb.in
inputs.set_val(
    Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1.0
)  # FRSC not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Wing.SWEEP, 35.7, 'deg')  # SWEEP in bwb.ins
inputs.set_val(Aircraft.Wing.TAPER_RATIO, 0.311)  # TR in bwb.in
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.11)  # TCA in bwb.in
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_REF, 0.11)  # TCREF not in bwb.in, default to TCA

inputs.set_val(
    Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
    np.array(
        [
            -1.0,
            0.15,
            0.1132,
            0.0928,
            0.0822,
            0.0764,
            0.0742,
            0.0746,
            0.0758,
            0.0758,
            0.0756,
            0.0756,
            0.0758,
            0.076,
            0.076,
        ]
    ),
)  # TOC
inputs.set_val(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75)  # ULF not in bwb.in, set to default
inputs.set_val(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0.0)  # VARSWP in bwb.in
inputs.set_val(Aircraft.Wing.MASS_SCALER, 1.0)  # FRWI in bwb.in
# inputs.set_val(Aircraft.Wing.WETTED_AREA, 2396.56, 'ft**2')  # SWETW not in bwb.in,
inputs.set_val(Aircraft.Wing.WETTED_AREA_SCALER, 1.0)  # SWETW not in bwb.in, set to Aviary default
inputs.set_val(Aircraft.Wing.DIHEDRAL, 3.0, 'deg')  # DIH in bwb.in
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 0.0)  # E in bwb.in

# Mission
# ---------------------------
inputs.set_val(Mission.Summary.CRUISE_MACH, 0.85)  # VCMN in bwb.in
inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)  # FACT not in bwb.in, set to default
inputs.set_val(Mission.Design.RANGE, 7750.0, 'NM')  # DESRNG in bwb.in
inputs.set_val(Mission.Constraints.MAX_MACH, 0.85)  # VMMO in bwb.in
# inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, 577, 'lbm')  # FTKOFL not in bwb.in

inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, 3.0)  # CLLDM not in bwb.in, set to default
inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2)  # CLTOM not in bwb.in, set to default
# inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG, 17.354)  # not in bwb.in, not a FLOPS variable
inputs.set_val(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 0.8)  # WRATIO in bwb.in
inputs.set_val(Mission.Landing.INITIAL_VELOCITY, 140.0, 'ft/s')  # VAPPR in bwb.in
inputs.set_val(
    Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.025
)  # ROLLMU not in bwb.in, set to default
# lbf TODO: where should this get connected from?
inputs.set_val(
    Mission.Design.THRUST_TAKEOFF_PER_ENG, 0.25, 'lbf'
)  # THROFF in bwb.in, output is 52724.3 lbf

# Settings
# ---------------------------
inputs.set_val(Settings.EQUATIONS_OF_MOTION, EquationsOfMotion.HEIGHT_ENERGY)
inputs.set_val(Settings.MASS_METHOD, LegacyCode.FLOPS)

# ---------------------------
#          OUTPUTS
# ---------------------------

# In FLOPS, DOWE = 411552.31557733245 because DOWE = WOWE.
outputs.set_val(Aircraft.Design.EMPTY_MASS, 390555.94982027, 'lbm')  # DOWE
outputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN, 0.0, 'lbm')  # WMARG
outputs.set_val(Aircraft.Design.OPERATING_MASS, 411552.34320647, 'lbm')  # WOWE
outputs.set_val(Aircraft.Propulsion.MASS, 58921.857380417721, 'lbm')  # WPRO
outputs.set_val(Aircraft.Design.STRUCTURE_MASS, 240989.14132753026, 'lbm')  # WSTRCT
outputs.set_val(Aircraft.Design.SYSTEMS_EQUIP_MASS, 90644.95111232, 'lbm')  # WSYS
outputs.set_val(Aircraft.Design.TOTAL_WETTED_AREA, 0.0, 'ft**2')  # TWET
outputs.set_val(Aircraft.Design.ZERO_FUEL_MASS, 509364.34320647, 'lbm')  # WZF
outputs.set_val(Mission.Design.FUEL_MASS, 364734.65679353, 'lbm')  # FUELM

outputs.set_val(
    Aircraft.Design.TOUCHDOWN_MASS, 699279.2, 'lbm'
)  # WLDG not in bwb.in, WLDG = GW*WRATIO

outputs.set_val(Aircraft.AirConditioning.MASS, 3897.6527857555625, 'lbm')  # WAC

outputs.set_val(Aircraft.AntiIcing.MASS, 562.09100951165135, 'lbm')  # WAI

outputs.set_val(Aircraft.APU.MASS, 2125.8280135763703, 'lbm')  # WAPU

outputs.set_val(Aircraft.Avionics.MASS, 2778.5110590964073, 'lbm')  # WAVONC

outputs.set_val(Aircraft.Canard.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL[-1]
outputs.set_val(Aircraft.Canard.FINENESS, 0.0)  # FR[-1]
outputs.set_val(Aircraft.Canard.WETTED_AREA, 0.0, 'ft**2')  # SWTCN
outputs.set_val(Aircraft.Canard.MASS, 0.0, 'lbm')  # WCAN

outputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS, 20592.0, 'lbm')  # WPBAG
outputs.set_val(Aircraft.CrewPayload.CARGO_MASS, 0.0, 'lbm')  # WCARGO
outputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS, 3850.0, 'lbm')  # WCON
outputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS, 450.0, 'lbm')  # WFLCRB
outputs.set_val(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, 3810.0, 'lbm')  # WSTUAB
outputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 10806.675950702213, 'lbm')  # WSRV
outputs.set_val(Aircraft.CrewPayload.PASSENGER_MASS, 77220.0, 'lbm')  # WPASS
outputs.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 97812.0, 'lbm')  # WPASS+WPBAG+WCARGO
outputs.set_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 97812.0, 'lbm')  # WPASS+WPBAG

outputs.set_val(Aircraft.Electrical.MASS, 4291.4778106479534, 'lbm')  # WELEC

outputs.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 1197720.2419621395, 'lbm')  # FMXTOT
outputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS, 5444.9572934402777, 'lbm')  # WFSYS
outputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 1732.78186198, 'lbm')  # WUF

outputs.set_val(Aircraft.Fins.MASS, 3159.3781042368792, 'lbm')  # WFIN

outputs.set_val(Aircraft.Furnishings.MASS, 57747.97136452, 'lbm')  # WFURN

outputs.set_val(Aircraft.BWB.NUM_BAYS, 7.0, 'unitless')  # NBAY
outputs.set_val(Aircraft.Fuselage.CABIN_AREA, 4697.33181006, 'ft**2')  # ACABIN
avg_diameter = 39.8525  # XD
avg_diameter_units = 'ft'
outputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, avg_diameter, avg_diameter_units)
outputs.set_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 137.5, 'ft')  # EL(4)
outputs.set_val(
    Aircraft.Fuselage.CROSS_SECTION, np.pi * (avg_diameter / 2.0) ** 2.0, f'{avg_diameter_units}**2'
)
outputs.set_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 0.16739117852998228)  # DB
outputs.set_val(Aircraft.Fuselage.FINENESS, 3.4502227)  # FR(4)
outputs.set_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER, 3.4502226961922089)  # BODYLD
outputs.set_val(Aircraft.Fuselage.MASS, 137935.30594648936, 'lbm')  # WFUSE
outputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 12.35302131, 'ft')  # DF
outputs.set_val(Aircraft.Fuselage.PLANFORM_AREA, 6710.4740143724875, 'ft**2')  # FPAREA
outputs.set_val(Aircraft.Fuselage.AFTBODY_MASS, 18736.55008878, 'lbm')  # WAFTB
outputs.set_val(Aircraft.Wing.BWB_AFTBODY_MASS, 15551.33657368, 'lbm')  # W4
outputs.set_val(Aircraft.Fuselage.LENGTH, 112.3001936860821, 'ft')  # XL
outputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 80.220756073526772, 'ft')  # WF
outputs.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 78.61013558, 'ft')  # XLP

outputs.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL(2)
outputs.set_val(Aircraft.HorizontalTail.FINENESS, 0.11)  # FR(2)
outputs.set_val(Aircraft.HorizontalTail.MASS, 0.0, 'lbm')  # WHT

outputs.set_val(Aircraft.Hydraulics.MASS, 6200.37391189, 'lbm')  # WHYD

outputs.set_val(Aircraft.Instruments.MASS, 1309.88942193, 'lbm')  # WIN

outputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS, 28200.322805698346, 'lbm')  # WLGM
outputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS, 2698.6740002098945, 'lbm')  # WLGN

outputs.set_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.array([15.68611614]), 'ft')  # EL(5)
outputs.set_val(Aircraft.Nacelle.FINENESS, np.array([1.38269353]))  # FR(5)
outputs.set_val(Aircraft.Nacelle.MASS, 0.0, 'lbm')  # WNAC

nacelle_wetted_area = np.array([498.26795086])  # SWET(5)
nacelle_wetted_area_units = 'ft**2'
outputs.set_val(Aircraft.Nacelle.WETTED_AREA, nacelle_wetted_area, nacelle_wetted_area_units)

outputs.set_val(
    Aircraft.Nacelle.TOTAL_WETTED_AREA, 3 * nacelle_wetted_area, nacelle_wetted_area_units
)

outputs.set_val(Aircraft.Paint.MASS, 0.0, 'lbm')  # WPAINT

outputs.set_val(
    Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 70000.0 * 3, 'lbf'
)  # output from propulsion,

outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 3)


outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, 346.93557352, 'lbm')  # WOIL
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 0)

outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 3)

outputs.set_val(Aircraft.Engine.MASS, 17825.63336233, 'lbm')  # WSP

# In FLOPS, WPOD = 18782.055842810063 because WSTART is scaled by
# WPMSC (Aircraft.Propulsion.MISC_MASS_SCALER) but not in Aviary.
# In Aviary, starter mass and engine controls are scaled later in EngineMiscMass()
# but it does not help for engine POD mass.
# In FLOPS, WSTART = 0.0 for the same reason (WPMSC = 0.0).
outputs.set_val(Aircraft.Engine.POD_MASS, 19307.96319637, 'lbm')  # WPOD
outputs.set_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 1526.1294678475103, 'lbm')  # WSTART
# In FLOPS, WEC = 0.0 because WEC is scaled by WPMSC (Aircraft.Propulsion.MISC_MASS_SCALER)
engine_ctrls_mass = 206.36860226  # WEC
outputs.set_val(Aircraft.Engine.CONTROLS_MASS, engine_ctrls_mass, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, engine_ctrls_mass, 'lbm')

outputs.set_val(Aircraft.Engine.ADDITIONAL_MASS, 0.0, 'lbm')  # WPMISC
outputs.set_val(Aircraft.Propulsion.TOTAL_MISC_MASS, 0.0, 'lbm')  # not in FLOPS
outputs.set_val(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, 0.0, 'lbm')  # sum of zeros
outputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS, 0.0, 'lbm')  # WTHR
# outputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 70000.0, 'lbf')  # THRUST

outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_MASS, 53476.90008698, 'lbm')  # WENG

outputs.set_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL(3)
outputs.set_val(Aircraft.VerticalTail.FINENESS, 0.11)  # FR(3)
outputs.set_val(Aircraft.VerticalTail.MASS, 0.0, 'lbm')  # WVT

outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR, 2.68745091)  # BT
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS, 6313.44762977, 'lbm')  # W1
outputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 69.53953418, 'ft')  # EL(1)
# Not in FLOPS output; calculated from inputs.
outputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 4032.5967, 'ft**2')  # SFLAP
outputs.set_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 1.0)  # CAYE
outputs.set_val(Aircraft.Wing.FINENESS, 0.11)  # FR(1)
outputs.set_val(Aircraft.Wing.MISC_MASS, 21498.83307778, 'lbm')  # W3
outputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS, 38779.21499739, 'lbm')  # W2
outputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS, 11731.15573539, 'lbm')  # WSC

outputs.set_val(Aircraft.Wing.MASS, 68995.460470895763, 'lbm')  # WWING
outputs.set_val(Aircraft.Wing.ROOT_CHORD, 38.5, 'ft')  # XLW
outputs.set_val(Aircraft.Wing.AREA, 12109.9, 'ft**2')  # SW, always computed for BWB

outputs.set_val(Mission.Design.MACH, 0.800)
# outputs.set_val(Mission.Design.LIFT_COEFFICIENT, -1.0)  # FCLDES
