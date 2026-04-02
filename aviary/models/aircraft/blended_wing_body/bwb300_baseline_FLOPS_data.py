import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.enums import AircraftTypes, EquationsOfMotion, LegacyCode
from aviary.variable_info.variables import Aircraft, Mission, Settings

BWB300FLOPS = {}
inputs = BWB300FLOPS['inputs'] = AviaryValues()
outputs = BWB300FLOPS['outputs'] = AviaryValues()

# Overall Aircraft
# ---------------------------
inputs.set_val(Aircraft.Design.BASE_AREA, 0.0, 'ft**2')  # SBASE
inputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN_SCALER, 0.0)  # EWMARG
inputs.set_val(
    Aircraft.Design.GROSS_MASS, 600000.0, 'lbm'
)  # DGW, value taken from GW which is not in Aviary
inputs.set_val(Aircraft.Design.USE_ALT_MASS, False)
inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)  # FCDI
inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)  # FCDSUB
inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)  # FCDSUP
inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)  # FCDO
inputs.set_val(Aircraft.Design.TYPE, AircraftTypes.BLENDED_WING_BODY)
inputs.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, False)
inputs.set_val(Aircraft.BWB.DETAILED_WING_PROVIDED, True)

# Air Conditioning
# ---------------------------
inputs.set_val(Aircraft.AirConditioning.MASS_SCALER, 1.0)  # WAC

# Anti-Icing
# ---------------------------
inputs.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)  # WAI

# APU
# ---------------------------
inputs.set_val(Aircraft.APU.MASS_SCALER, 1.0)  # WAPU

# Avionics
# ---------------------------
inputs.set_val(Aircraft.Avionics.MASS_SCALER, 1.0)  # WAVONC

# Canard
# ---------------------------
inputs.set_val(Aircraft.Canard.AREA, 0.0, 'ft**2')  # SCAN
inputs.set_val(Aircraft.Canard.ASPECT_RATIO, 0.0)  # ARCAN
inputs.set_val(Aircraft.Canard.THICKNESS_TO_CHORD, 0.0)  # TCCAN
inputs.set_val(Aircraft.Canard.LAMINAR_FLOW_LOWER, 0.0)  # TRLC
inputs.set_val(Aircraft.Canard.LAMINAR_FLOW_UPPER, 0.0)  # TRUC
inputs.set_val(Aircraft.Canard.MASS_SCALER, 1.0)  # FRCAN
inputs.set_val(Aircraft.Canard.TAPER_RATIO, 0.0)  # TRCAN
inputs.set_val(Aircraft.Canard.WETTED_AREA_SCALER, 1.0)  # SWETC

# Crew and Payload
# ---------------------------
inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 76)  # NPB
inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 24)  # NPF
inputs.set_val(Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS, 200)  # NPT
inputs.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 76)  # NPB
inputs.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 24)  # NPF
inputs.set_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS, 200)  # NPT
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 0)  # NBABR
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 0)  # NFABR
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 0)  # NTABR
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS, 0.0, 'inch')  # BPITCH
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 0.0, 'inch')  # FPITCH
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 0.0, 'inch')  # TPITCH

inputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, 1.0)  # WCON
inputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS, 23500.0, 'lbm')  # WCON
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, 2)  # NFLCR
inputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER, 1.0)  # WFLCRB
inputs.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, 2)  # NGALC
inputs.set_val(Aircraft.CrewPayload.MISC_CARGO, 0.0, 'lbm')  # CARGOF
inputs.set_val(Aircraft.CrewPayload.CABIN_CREW_MASS_SCALER, 1.0)  # WSTUAB
inputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, 1.0)  # WSRV
inputs.set_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 165.0, 'lbm')  # WPPASS
inputs.set_val(Aircraft.CrewPayload.WING_CARGO, 0.0, 'lbm')  # CARGOW

# Electrical
# ---------------------------
inputs.set_val(Aircraft.Electrical.MASS_SCALER, 1.0)  # WELEC

# Fins
# ---------------------------
inputs.set_val(Aircraft.Fins.AREA, 184.89, 'ft**2')  # SFIN
inputs.set_val(Aircraft.Fins.NUM_FINS, 2)  # NFIN
inputs.set_val(Aircraft.Fins.TAPER_RATIO, 0.464)  # TRFIN
inputs.set_val(Aircraft.Fins.MASS_SCALER, 1.0)  # FRFIN

# Fuel
# ---------------------------
inputs.set_val(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0, 'lbm')  # FULAUX
inputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0)  # WFSYS
inputs.set_val(Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0, 'lbm')  # FULFMX
inputs.set_val(Aircraft.Fuel.NUM_TANKS, 7)  # NTANK
inputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, 1.0)  # WUF
inputs.set_val(Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT, False)  # IFUFU
inputs.set_val(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6883549569366508, 'unitless')
inputs.set_val(Aircraft.Fuel.WING_REF_CAPACITY, 0.0, 'lbm')  # FUELRF
inputs.set_val(Aircraft.Fuel.WING_REF_CAPACITY_TERM_A, 0.0)  # FUSCLA
inputs.set_val(Aircraft.Fuel.WING_REF_CAPACITY_TERM_B, 0.0)  # FUSCLB


# Furnishings
# ---------------------------
inputs.set_val(Aircraft.Furnishings.MASS_SCALER, 1.118)  # WFURN

# Fuselage
# ---------------------------
inputs.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)  # NFUSE
inputs.set_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR, False)  # CARGF
inputs.set_val(Aircraft.Fuselage.MASS_SCALER, 1.0)  # FRFU
inputs.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 60.0, 'deg')  # SWPLE
inputs.set_val(Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD, 0.1792)  # TCF
inputs.set_val(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0)  # SWETF
inputs.set_val(Aircraft.Fuselage.LAMINAR_FLOW_LOWER, 0.0)  # TRLB
inputs.set_val(Aircraft.Fuselage.LAMINAR_FLOW_UPPER, 0.0)  # TRUB

# Horizontal Tail
# ---------------------------
inputs.set_val(Aircraft.HorizontalTail.AREA, 700.0, 'ft**2')  # SHT
inputs.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, 1.0)  # ARHT
inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 0.7140)  # TRHT
inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1)  # TCHT
inputs.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0.0)  # HHT
inputs.set_val(Aircraft.HorizontalTail.MASS_SCALER, 1.0)  # FRHT
inputs.set_val(Aircraft.HorizontalTail.NUM_TAILS, 1)
inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA_SCALER, 1.0)  # SWETH
inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0.0, 'deg')  # SWPHT
inputs.set_val(Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER, 0.0)  # TRLH
inputs.set_val(Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER, 0.0)  # TRUH

# Hydraulics
# ---------------------------
inputs.set_val(Aircraft.Hydraulics.SYSTEM_PRESSURE, 3000.0, 'psi')  # HYDPR
inputs.set_val(Aircraft.Hydraulics.MASS_SCALER, 1.0)  # WHYD

# Instruments
# ---------------------------
inputs.set_val(Aircraft.Instruments.MASS_SCALER, 1.0)  # WIN

# Landing Gear
# ---------------------------
inputs.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 85.0, 'inch')  # XMLG
inputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, 1.0)  # FRLGM
inputs.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 87.0, 'inch')  # XNLG
inputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, 1.0)  # FRLGN

# Nacelle
# ---------------------------
inputs.set_val(Aircraft.Nacelle.AVG_DIAMETER, 12.569, 'ft')  # DNAC
inputs.set_val(Aircraft.Nacelle.AVG_LENGTH, 17.297, 'ft')  # XNAC
inputs.set_val(Aircraft.Nacelle.MASS_SCALER, 0.0)  # FRNA
inputs.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, 1.0)  # SWETN
inputs.set_val(Aircraft.Nacelle.LAMINAR_FLOW_LOWER, 0.0)  # TRLN
inputs.set_val(Aircraft.Nacelle.LAMINAR_FLOW_UPPER, 0.0)  # TRUN

# Paint
# ---------------------------
inputs.set_val(Aircraft.Paint.MASS_PER_UNIT_AREA, 0.0, 'lbm/ft**2')  # WPAINT

# Propulsion and Engine
# ---------------------------
inputs.set_val(Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, 1.0)  # WOIL
inputs.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 0.0)  # WPMSC

filename = get_path('models/engines/PAX300_baseline_ENGDEK.csv')
inputs.set_val(Aircraft.Engine.DATA_FILE, filename)
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 22089.3, 'lbm')  # WENG
inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 87500.0, 'lbf')  # THRUST
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 86786.4, 'lbf')  # THRSO
inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 2)  # NEF
inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, np.array([0]))  # NEW
inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)  # WTHR
inputs.set_val(Aircraft.Engine.WING_LOCATIONS, np.array([0.2]))  # ETAE
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0082224864725349)  # THRUST/THRSO
inputs.set_val(Aircraft.Engine.SCALE_MASS, True)
inputs.set_val(Aircraft.Engine.MASS_SCALER, 1.0)  # EEXP
inputs.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
inputs.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)  # FFFSUB
inputs.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)  # FFFSUP
inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)  # DFFAC
inputs.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 0.0)  # FFFAC
inputs.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h')  # FLEAK
inputs.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0)  # WPMISC
inputs.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)  # IDLE
inputs.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)  # NONEG
inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)  # FIDMAX
inputs.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)  # FIDMIN
inputs.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)  # IGEO
inputs.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')

# Vertical Tail
# ---------------------------
inputs.set_val(Aircraft.VerticalTail.NUM_TAILS, 0)  # NVERT
inputs.set_val(Aircraft.VerticalTail.AREA, 0.0, 'ft**2')  # SVT
inputs.set_val(Aircraft.VerticalTail.ASPECT_RATIO, 0.5)  # ARVT
inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 0.714)  # TRVT
inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.11)  # TCVT
inputs.set_val(Aircraft.VerticalTail.MASS_SCALER, 1.0)  # FRVT
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA_SCALER, 1.0)  # SWETV
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA, 125.0, 'ft**2')  # SWETV
inputs.set_val(Aircraft.VerticalTail.SWEEP, 0.0, 'deg')  # SWPVT
inputs.set_val(Aircraft.VerticalTail.LAMINAR_FLOW_LOWER, 0.0)  # TRLV
inputs.set_val(Aircraft.VerticalTail.LAMINAR_FLOW_UPPER, 0.0)  # TRUV

# Wing
# ---------------------------
inputs.set_val(Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0)  # FAERT
inputs.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 2.0)  # AITEK
inputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0)  # FRWI1
inputs.set_val(Aircraft.Wing.SPAN, 186.631829293424, 'ft')  # SPAN
inputs.set_val(Aircraft.Wing.LOAD_FRACTION, 1.0)  # PCTL

inputs.set_val(
    Aircraft.Wing.CHORD_PER_SEMISPAN_DISTRIBUTION,
    np.array([-1.0, 48.25, 33.20, 18.97, 14.19, 10.20, 3.220]),
)  # CHD
inputs.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 0.85)  # FCOMP
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.3)  # FLAPR
inputs.set_val(Aircraft.Wing.DETAILED_WING, True)
inputs.set_val(
    Aircraft.Wing.GLOVE_AND_BAT, 1230.5, 'ft**2'
)  # GLOV, it was 0.0 as input and computed

inputs.set_val(
    Aircraft.Wing.INPUT_STATION_DISTRIBUTION,
    np.array([0.0, 0.0, 0.2075, 0.415, 0.6927, 0.928, 1.0]),  # ETAW
)

inputs.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0)  # PDIST

inputs.set_val(
    Aircraft.Wing.LOAD_PATH_SWEEP_DISTRIBUTION,
    np.array([0.0, 0.0, 0.0, 17.0, 17.0, 17.0]),
    'deg',  # SWL
)
inputs.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 2.0)  # CAM
inputs.set_val(Aircraft.Wing.MISC_MASS_SCALER, 1.0)  # FRWI3
inputs.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 50)  # NSTD
inputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0)  # FRWI2
inputs.set_val(Aircraft.Wing.OUTBOARD_SEMISPAN, 68.43, 'ft')  # OSSPAN
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, False)  # MIKE
inputs.set_val(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0)  # FSTRT
inputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1.0)  # FRSC
inputs.set_val(Aircraft.Wing.SWEEP, 35.7, 'deg')  # SWEEP
inputs.set_val(Aircraft.Wing.TAPER_RATIO, 0.311)  # TR
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.11)  # TCA
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE, 0.0)  # TCREF

inputs.set_val(
    Aircraft.Wing.THICKNESS_TO_CHORD_DISTRIBUTION,
    np.array([-1.0, 0.125, 0.125, 0.076, 0.076, 0.076, 0.06]),  # TOC
)
inputs.set_val(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75)  # ULF
inputs.set_val(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0.0)  # VARSWP
inputs.set_val(Aircraft.Wing.MASS_SCALER, 1.0)  # FRWI
inputs.set_val(
    Aircraft.Wing.WETTED_AREA_SCALER, 0.61
)  # SWETW was 1.0, changed to match the AERO from old version of FLOPS
inputs.set_val(Aircraft.Wing.DIHEDRAL, 3.0, 'deg')  # DIH
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 1.0)  # E
inputs.set_val(Aircraft.Wing.LAMINAR_FLOW_LOWER, 0.0)  # TRLW
inputs.set_val(Aircraft.Wing.LAMINAR_FLOW_UPPER, 0.0)  # TRUW
inputs.set_val(Aircraft.Wing.ASPECT_RATIO_REFERENCE, 0.0)  # ARREF

# Mission
# ---------------------------
inputs.set_val(Mission.Summary.CRUISE_MACH, 0.85)  # VCMN
inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)  # FACT
inputs.set_val(Aircraft.Design.RANGE, 7500.0, 'NM')  # DESRNG
inputs.set_val(Mission.Constraints.MAX_MACH, 0.9)  # VMMO

inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, 3.0)  # CLLDM
inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 1.3)  # CLTOM
inputs.set_val(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 0.7)  # WRATIO
inputs.set_val(Mission.Landing.INITIAL_VELOCITY, 150.0, 'ft/s')  # VAPPR
inputs.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.025)  # ROLLMU
inputs.set_val(Aircraft.Design.THRUST_TAKEOFF_PER_ENG, 0.0, 'lbf')  # THROFF
inputs.set_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 4.0, 'deg')  # ALPRUN
inputs.set_val(Mission.Takeoff.THRUST_INCIDENCE, 4.0, 'deg')  # TINC
inputs.set_val(Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT, 0.35)  # BRAKMU
inputs.set_val(Aircraft.LandingGear.DRAG_COEFFICIENT, 0.005)  # CDGEAR
inputs.set_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT, 0.0)  # CLSPOL
inputs.set_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT, 0.0)  # CDSPOL
inputs.set_val(Mission.Takeoff.DRAG_COEFFICIENT_MIN, 0.035258)  # CDMTO
inputs.set_val(Mission.Takeoff.FINAL_ALTITUDE, 35.0, 'ft')  # OBSTO
inputs.set_val(Mission.Landing.DRAG_COEFFICIENT_MIN, 0.0)  # CDMLD
inputs.set_val(Mission.Landing.FLARE_RATE, 3.0, 'deg/s')  # VANGLD

# Settings
# ---------------------------
inputs.set_val(Settings.EQUATIONS_OF_MOTION, EquationsOfMotion.HEIGHT_ENERGY)
inputs.set_val(Settings.MASS_METHOD, LegacyCode.FLOPS)
inputs.set_val(Settings.AERODYNAMICS_METHOD, LegacyCode.FLOPS)

# ---------------------------
#          OUTPUTS
# ---------------------------

outputs.set_val(Aircraft.Design.EMPTY_MASS, 286969.99768419, 'lbm')
outputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN, 0.0, 'lbm')  # WMARG
outputs.set_val(Aircraft.Design.STRUCTURE_MASS, 162969.90469722, 'lbm')  # WSTRCT 158921.83401643133
outputs.set_val(Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, 75801.466406974854, 'lbm')  # WSYS
outputs.set_val(Aircraft.Design.TOTAL_WETTED_AREA, 19637.79833526, 'ft**2')  # TWET
outputs.set_val(Aircraft.Design.TOUCHDOWN_MASS, 420000.0, 'lbm')  # WLDG = GW*WRATIO

outputs.set_val(Aircraft.AirConditioning.MASS, 3781.61256774, 'lbm')  # WAC
outputs.set_val(Aircraft.AntiIcing.MASS, 400.3921819029477, 'lbm')  # WAI
outputs.set_val(Aircraft.APU.MASS, 1578.8098560285962, 'lbm')  # WAPU
outputs.set_val(Aircraft.Avionics.MASS, 2280.13561342, 'lbm')  # WAVONC

outputs.set_val(Aircraft.BWB.NUM_BAYS, 4.0, 'unitless')  # NBAY
outputs.set_val(Aircraft.Canard.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL[-1]
outputs.set_val(Aircraft.Canard.FINENESS, 0.0)  # FR[-1]
outputs.set_val(Aircraft.Canard.WETTED_AREA, 0.0, 'ft**2')  # SWTCN
outputs.set_val(Aircraft.Canard.MASS, 0.0, 'lbm')  # WCAN

outputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS, 13200.0, 'lbm')  # WPBAG
outputs.set_val(Aircraft.CrewPayload.CARGO_MASS, 0.0, 'lbm')  # WCARGO
outputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS, 450.0, 'lbm')  # WFLCRB
outputs.set_val(Aircraft.CrewPayload.CABIN_CREW_MASS, 1640.0, 'lbm')  # WSTUAB
outputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, 7029.593528180887, 'lbm')  # WSRV
outputs.set_val(Aircraft.CrewPayload.PASSENGER_MASS_TOTAL, 49500.0, 'lbm')  # WPASS
outputs.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 62700.0, 'lbm')  # WPASS+WPBAG+WCARGO
outputs.set_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 62700.0, 'lbm')  # WPASS+WPBAG

outputs.set_val(Aircraft.Electrical.MASS, 2646.5272348061812, 'lbm')  # WELEC

outputs.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 787493.65267017565, 'lbm')  # FMXTOT
outputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS, 3656.7260445688612, 'lbm')  # WFSYS
outputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 1732.78186198, 'lbm')  # WUF
outputs.set_val(Aircraft.Fuel.WING_FUEL_CAPACITY, 787493.65267018, 'lbm')  # FULWMX

outputs.set_val(Aircraft.Fins.MASS, 2822.1415450307886, 'lbm')  # WFIN
outputs.set_val(Aircraft.Furnishings.MASS, 52096.553437128503, 'lbm')  # WFURN

outputs.set_val(Aircraft.Fuselage.CABIN_AREA, 2988.87966179, 'ft**2')  # ACABIN
ref_diameter = 35.331132876385297
outputs.set_val(Aircraft.Fuselage.REF_DIAMETER, ref_diameter, 'ft')  # XD
outputs.set_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 116.5760963133181, 'ft')  # EL(4)
outputs.set_val(Aircraft.Fuselage.CROSS_SECTION, np.pi * (ref_diameter / 2.0) ** 2.0, 'ft**2')
outputs.set_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 0.18930926)  # DB
outputs.set_val(Aircraft.Fuselage.FINENESS, 3.2995289656062941)  # FR(4)
outputs.set_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER, 3.29952897)  # BODYLD
outputs.set_val(Aircraft.Fuselage.MASS, 80216.313556241628, 'lbm')  # WFUSE
outputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 20.89043646, 'ft')  # DF
outputs.set_val(Aircraft.Fuselage.PLANFORM_AREA, 4269.82808827, 'ft**2')  # FPAREA
outputs.set_val(Aircraft.Fuselage.AFTBODY_MASS, 10384.964957095559, 'lbm')  # WAFTB
outputs.set_val(Aircraft.Fuselage.LENGTH, 116.57609631, 'ft')  # XL
outputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 49.77182929, 'ft')  # WF
outputs.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 81.60326742, 'ft')  # XLP

outputs.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 26.45751311065, 'ft')  # EL(2)
outputs.set_val(Aircraft.HorizontalTail.FINENESS, 0.1)  # FR(2)
outputs.set_val(Aircraft.HorizontalTail.MASS, 6444.9988831532046, 'lbm')  # WHT
outputs.set_val(Aircraft.HorizontalTail.WETTED_AREA, 983.26501, 'ft**2')  # SWTHT

outputs.set_val(Aircraft.Design.EMPENNAGE_MASS, 3159.3781042368792, 'lbm')

outputs.set_val(Aircraft.Hydraulics.MASS, 3962.6923427813854, 'lbm')  # WHYD

outputs.set_val(Aircraft.Instruments.MASS, 961.54346235801643, 'lbm')  # WIN

outputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS, 28200.322805698346, 'lbm')  # WLGM
outputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS, 2698.6740002098945, 'lbm')  # WLGN
outputs.set_val(Aircraft.LandingGear.TOTAL_MASS, 30898.996805908242, 'lbm')

outputs.set_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.array([17.367966592445]), 'ft')  # EL(5)
outputs.set_val(Aircraft.Nacelle.FINENESS, np.array([1.3761635770546583]))  # FR(5)
outputs.set_val(Aircraft.Nacelle.MASS, 0.0, 'lbm')  # WNAC
nacelle_wetted_area = np.array([613.74211034217353])  # SWET(5)
outputs.set_val(Aircraft.Nacelle.WETTED_AREA, nacelle_wetted_area, 'ft**2')
outputs.set_val(Aircraft.Nacelle.TOTAL_WETTED_AREA, 2 * nacelle_wetted_area, 'ft**2')

outputs.set_val(Aircraft.Paint.MASS, 0.0, 'lbm')  # WPAINT, WTPNT

outputs.set_val(Aircraft.Propulsion.MASS, 48198.583985444384, 'lbm')  # WPRO
outputs.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 70000.0 * 3, 'lbf')
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 3)
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, 267.39241429019251, 'lbm')  # WOIL
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 0)
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 2)

outputs.set_val(Aircraft.Engine.MASS, 44541.857940875525 / 2, 'lbm')  # WSP(10, 2)
# In FLOPS BWB, WPOD = 18782.055842810063 because WSTART is scaled by
# WPMSC (Aircraft.Propulsion.MISC_MASS_SCALER) but not in Aviary.
# In Aviary, starter mass and engine controls are scaled later in EngineMiscMass()
# but it does not help for engine POD mass.
# In FLOPS, WSTART = 0.0 for the same reason (WPMSC = 0.0).
outputs.set_val(Aircraft.Engine.POD_MASS, 19307.96319637, 'lbm')  # WPOD
# In FLOPS BWB, WEC = 0.0 because WEC is scaled by WPMSC (Aircraft.Propulsion.MISC_MASS_SCALER)
engine_ctrls_mass = 153.81807436  # WEC
outputs.set_val(Aircraft.Engine.ADDITIONAL_MASS, 0.0, 'lbm')  # WPMISC
outputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS, 0.0, 'lbm')  # WTHR
outputs.set_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 1526.1294678475103, 'lbm')  # WSTART
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, engine_ctrls_mass, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_MISC_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, 0.0, 'lbm')  # sum of zeros
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_MASS, 44541.857940875525, 'lbm')  # WENG

outputs.set_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL(3)
outputs.set_val(Aircraft.VerticalTail.FINENESS, 0.11)  # FR(3)
outputs.set_val(Aircraft.VerticalTail.MASS, 0.0, 'lbm')  # WVT
outputs.set_val(Aircraft.VerticalTail.WETTED_AREA, 125.0, 'ft**2')

outputs.set_val(Aircraft.Wing.BWB_AFTBODY_MASS, 8884.3375208, 'lbm')  # W4
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR, 5.20084141)  # FLOPS BT = 6.7996347825592336
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS, 13872.4182868, 'lbm')  # W1
outputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 45.124750222881779, 'ft')  # EL(1)
outputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 2526.5144041515805, 'ft**2')  # SFLAP
outputs.set_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 1.0)  # CAYE
outputs.set_val(Aircraft.Wing.FINENESS, 0.11)  # FR(1)
outputs.set_val(Aircraft.Wing.MISC_MASS, 6975.77622754, 'lbm')  # W3
outputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS, 24461.161868706797, 'lbm')  # W2
outputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS, 8093.1997108029764, 'lbm')  # WSC
outputs.set_val(Aircraft.Wing.ASPECT_RATIO, 4.84361005)  # AR
outputs.set_val(Aircraft.Wing.MASS, 68922.20579045, 'lbm')  # WWING 68995.460470895763
outputs.set_val(Aircraft.Wing.ROOT_CHORD, 38.5, 'ft')  # XLW
outputs.set_val(Aircraft.Wing.AREA, 8421.7146805052689, 'ft**2')  # SW
outputs.set_val(Aircraft.Wing.WETTED_AREA, 17302.04910213, 'ft**2')  # SWET(1)

outputs.set_val(Mission.Summary.USEFUL_LOAD, 20996.3933862, 'lbm')

# outputs.set_val(Aircraft.Design.MACH, 0.800)  # FMDES
outputs.set_val(Mission.Summary.OPERATING_MASS, 321171.82272983, 'lbm')  # DOWE
outputs.set_val(Mission.Summary.ZERO_FUEL_MASS, 383871.82272983, 'lbm')  # WZF
outputs.set_val(Mission.Summary.FUEL_MASS, 216128.17727028, 'lbm')  # FUELM
