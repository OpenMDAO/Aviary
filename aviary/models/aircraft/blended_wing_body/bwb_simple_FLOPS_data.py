import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.variable_info.enums import AircraftTypes, EquationsOfMotion, LegacyCode
from aviary.variable_info.variables import Aircraft, Mission, Settings

BWBsimpleFLOPS = {}
inputs = BWBsimpleFLOPS['inputs'] = AviaryValues()
outputs = BWBsimpleFLOPS['outputs'] = AviaryValues()

# Overall Aircraft
# ---------------------------
inputs.set_val(Aircraft.Design.BASE_AREA, 0.0, 'ft**2')  # SBASE
inputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN_SCALER, 0.0)  # EWMARG
inputs.set_val(Mission.Design.GROSS_MASS, 874099.0, 'lbm')  # DGW
inputs.set_val(Aircraft.Design.USE_ALT_MASS, False)
inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)  # FCDI
inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)  # FCDSUB
inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0)  # FCDSUP
inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)  # FCDO
inputs.set_val(Aircraft.Design.TYPE, AircraftTypes.BLENDED_WING_BODY)
inputs.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, True)
inputs.set_val(Aircraft.BWB.DETAILED_WING_PROVIDED, False)

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

# Crew and Payload
# ---------------------------
inputs.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 100)  # NPB
inputs.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 28)  # NPF
inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 468, units='unitless')  # NPB+NPF+NPT
inputs.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 340)  # NPT
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 4)  # NBABR
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 4)  # NFABR
inputs.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)  # NTABR
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS, 39, 'inch')  # BPITCH
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 61, 'inch')  # FPITCH
inputs.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 32, 'inch')  # TPITCH

inputs.set_val(Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER, 1.0)  # WCON
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, 22)  # NSTU
inputs.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, 2)  # NFLCR
inputs.set_val(Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER, 1.0)  # WFLCRB
inputs.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, 2)  # NGALC
inputs.set_val(Aircraft.CrewPayload.MISC_CARGO, 0.0, 'lbm')  # CARGOF
inputs.set_val(Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER, 1.0)  # WSTUAB
inputs.set_val(Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER, 1.0)  # WSRV
inputs.set_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 165.0, 'lbm')  # WPPASS
inputs.set_val(Aircraft.CrewPayload.WING_CARGO, 0.0, 'lbm')  # CARGOW
inputs.set_val(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 44.0, 'lbm')  # BPP

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
inputs.set_val(Aircraft.Fuel.WING_FUEL_FRACTION, 0.68835495693, 'unitless')

# Furnishings
# ---------------------------
inputs.set_val(Aircraft.Furnishings.MASS_SCALER, 1.0)  # WFURN

# Fuselage
# ---------------------------
inputs.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)  # NFUSE
inputs.set_val(Aircraft.Fuselage.LENGTH, 137.5, 'ft')  # X
inputs.set_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR, False)  # CARGF
inputs.set_val(Aircraft.Fuselage.MASS_SCALER, 1.0)  # FRFU
inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 64.58, 'ft')  # WF
inputs.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 45.0, 'deg')  # SWPLE
inputs.set_val(Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.11)  # TCF
# inputs.set_val(Aircraft.Fuselage.WETTED_AREA, 0.0, 'ft**2')  # see _BWBFuselage()
inputs.set_val(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0)  # SWETF

# Horizontal Tail
# ---------------------------
inputs.set_val(Aircraft.HorizontalTail.AREA, 0.0, 'ft**2')  # SHT
inputs.set_val(Aircraft.HorizontalTail.ASPECT_RATIO, 0.1)  # SHT
inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 0.0)  # TRHT
inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.11)  # TCHT
# inputs.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0.0)  # HHT
inputs.set_val(Aircraft.HorizontalTail.MASS_SCALER, 1.0)  # SHT
inputs.set_val(Aircraft.HorizontalTail.WETTED_AREA_SCALER, 1.0)  # SWETH
# inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0.0)  # SWPHT

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
inputs.set_val(Aircraft.Nacelle.AVG_DIAMETER, 12.608, 'ft')  # DNAC
inputs.set_val(Aircraft.Nacelle.AVG_LENGTH, 17.433, 'ft')  # XNAC
inputs.set_val(Aircraft.Nacelle.MASS_SCALER, 0.0)  # FRNA
inputs.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, 1.0)  # SWETN

# Paint
# ---------------------------
inputs.set_val(Aircraft.Paint.MASS_PER_UNIT_AREA, 0.0, 'lbm/ft**2')  # WPAINT

# Propulsion and Engine
# ---------------------------
inputs.set_val(Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER, 1.0)  # WOIL
inputs.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 0.0)  # WPMSC

filename = get_path('models/engines/PAX300_baseline_ENGDEK.csv')
inputs.set_val(Aircraft.Engine.DATA_FILE, filename)
inputs.set_val(Aircraft.Engine.REFERENCE_MASS, 22017, 'lbm')  # WENG
inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, 70000.0, 'lbf')  # THRUST
inputs.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 86459.2, 'lbf')  # THRSO
inputs.set_val(Aircraft.Engine.NUM_ENGINES, np.array([3]))  # NEW+NEF
inputs.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 3)  # NEF
inputs.set_val(Aircraft.Engine.NUM_WING_ENGINES, np.array([0]))  # NEW
inputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER, 0.0)  # WTHR
inputs.set_val(Aircraft.Engine.WING_LOCATIONS, 0)  # ETAE
inputs.set_val(Aircraft.Engine.SCALE_FACTOR, 0.8096304384)  # THRUST/THRSO
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
inputs.set_val(Aircraft.VerticalTail.ASPECT_RATIO, 0.1)  # ARVT
inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 0.0)  # TRVT
inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.11)  # TCVT
inputs.set_val(Aircraft.VerticalTail.MASS_SCALER, 1.0)  # FRVT
inputs.set_val(Aircraft.VerticalTail.WETTED_AREA_SCALER, 1.0)  # SWETV

# Wing
# ---------------------------
inputs.set_val(Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0)  # FAERT
inputs.set_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, 2.0)  # AITEK
inputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0)  # FRWI1
inputs.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 1.0)  # FCOMP
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333)  # FLAPR
inputs.set_val(Aircraft.Wing.DETAILED_WING, True)
inputs.set_val(Aircraft.Wing.GLOVE_AND_BAT, 121.05, 'ft**2')  # GLOV

inputs.set_val(Aircraft.Wing.INPUT_STATION_DIST, np.array([0.0, 0.5, 1.0]))  # ETAW
inputs.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0)  # PDIST
inputs.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 2.0)  # CAM
inputs.set_val(Aircraft.Wing.MISC_MASS_SCALER, 1.0)  # FRWI3
inputs.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 50)  # NSTD
inputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0)  # FRWI2
inputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333)  # FLAPR
inputs.set_val(Aircraft.Wing.SPAN, 238.08, 'ft')  # SPAN = WF+OSSPAN*2
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, False)  # MIKE
inputs.set_val(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0)  # FSTRT
inputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1.0)  # FRSC
inputs.set_val(Aircraft.Wing.SWEEP, 35.7, 'deg')  # SWEEP
inputs.set_val(Aircraft.Wing.TAPER_RATIO, 0.311)  # TR
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.11)  # TCA
inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_REF, 0.11)  # TCREF

inputs.set_val(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75)  # ULF
inputs.set_val(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0.0)  # VARSWP
inputs.set_val(Aircraft.Wing.MASS_SCALER, 1.0)  # FRWI
inputs.set_val(Aircraft.Wing.WETTED_AREA_SCALER, 1.0)  # SWETW
inputs.set_val(Aircraft.Wing.DIHEDRAL, 3.0, 'deg')  # DIH
inputs.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 0.0)  # E

# Mission
# ---------------------------
inputs.set_val(Mission.Summary.CRUISE_MACH, 0.85)  # VCMN
inputs.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)  # FACT
inputs.set_val(Mission.Design.RANGE, 7750.0, 'NM')  # DESRNG
inputs.set_val(Mission.Constraints.MAX_MACH, 0.85)  # VMMO
# inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, 577, 'lbm')  # FTKOFL

# inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, 3.0)  # CLLDM
inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2)  # CLTOM
# inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG, 17.354)
inputs.set_val(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, 0.8)  # WRATIO
inputs.set_val(Mission.Landing.INITIAL_VELOCITY, 140.0, 'ft/s')  # VAPPR
inputs.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.025)  # ROLLMU
inputs.set_val(Mission.Design.THRUST_TAKEOFF_PER_ENG, 0.25, 'lbf')  # THROFF

# Settings
# ---------------------------
inputs.set_val(Settings.EQUATIONS_OF_MOTION, EquationsOfMotion.HEIGHT_ENERGY)
inputs.set_val(Settings.MASS_METHOD, LegacyCode.FLOPS)

# ---------------------------
#          OUTPUTS
# ---------------------------

# In FLOPS, DOWE = 455464.65969526308 because DOWE = WOWE.
outputs.set_val(Aircraft.Design.EMPTY_MASS, 434037.32820147, 'lbm')  # DOWE
outputs.set_val(Aircraft.Design.EMPTY_MASS_MARGIN, 0.0, 'lbm')  # WMARG
outputs.set_val(Aircraft.Design.STRUCTURE_MASS, 273591.31917826, 'lbm')  # WSTRCT
outputs.set_val(Aircraft.Design.SYSTEMS_EQUIP_MASS, 98848.9061107412710, 'lbm')  # WSYS
outputs.set_val(Aircraft.Design.TOTAL_WETTED_AREA, 35311.53118076, 'ft**2')  # TWET
outputs.set_val(Aircraft.Design.TOUCHDOWN_MASS, 699279.2, 'lbm')  # WLDG = GW*WRATIO

outputs.set_val(Aircraft.AirConditioning.MASS, 4383.96064972, 'lbm')  # WAC
outputs.set_val(Aircraft.AntiIcing.MASS, 519.37038003, 'lbm')  # WAI
outputs.set_val(Aircraft.APU.MASS, 2148.13002234, 'lbm')  # WAPU
outputs.set_val(Aircraft.Avionics.MASS, 2896.223816950469, 'lbm')  # WAVONC

outputs.set_val(Aircraft.BWB.NUM_BAYS, 5.0, 'unitless')  # NBAY
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

outputs.set_val(Aircraft.Electrical.MASS, 4514.28869169, 'lbm')  # WELEC

outputs.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 2385712.4988316689, 'lbm')  # FMXTOT
outputs.set_val(Aircraft.Fuel.FUEL_SYSTEM_MASS, 8120.2023807944415, 'lbm')  # WFSYS
outputs.set_val(Aircraft.Fuel.UNUSABLE_FUEL_MASS, 2163.999415070652, 'lbm')  # WUF
outputs.set_val(Aircraft.Fuel.WING_FUEL_CAPACITY, 2385712.4988316689, 'lbm')  # FULWMX

outputs.set_val(Aircraft.Fins.MASS, 3159.3781042368792, 'lbm')  # WFIN
outputs.set_val(Aircraft.Furnishings.MASS, 61482.097969438299, 'lbm')  # WFURN

outputs.set_val(Aircraft.Fuselage.CABIN_AREA, 5173.187202504683, 'ft**2')  # ACABIN
ref_diameter = 39.8525
outputs.set_val(Aircraft.Fuselage.REF_DIAMETER, ref_diameter, 'ft')  # XD
outputs.set_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 137.5, 'ft')  # EL(4)
outputs.set_val(Aircraft.Fuselage.CROSS_SECTION, np.pi * (ref_diameter / 2.0) ** 2.0, 'ft**2')
outputs.set_val(Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, 0.16739117852998228)  # DB
outputs.set_val(Aircraft.Fuselage.FINENESS, 3.4502227)  # FR(4)
outputs.set_val(Aircraft.Fuselage.LENGTH_TO_DIAMETER, 3.4502226961922089)  # BODYLD
outputs.set_val(Aircraft.Fuselage.MASS, 152790.66300003964, 'lbm')  # WFUSE
outputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 15.125, 'ft')  # DF
outputs.set_val(Aircraft.Fuselage.PLANFORM_AREA, 7390.267432149546, 'ft**2')  # FPAREA
outputs.set_val(Aircraft.Fuselage.AFTBODY_MASS, 24278.05868511, 'lbm')  # WAFTB
outputs.set_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, 96.25, 'ft')

outputs.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL(2)
outputs.set_val(Aircraft.HorizontalTail.FINENESS, 0.11)  # FR(2)
outputs.set_val(Aircraft.HorizontalTail.MASS, 0.0, 'lbm')  # WHT
outputs.set_val(Aircraft.HorizontalTail.WETTED_AREA, 0.0, 'ft**2')  # SWTHT

outputs.set_val(Aircraft.Hydraulics.MASS, 7368.5077321194321, 'lbm')  # WHYD

outputs.set_val(Aircraft.Instruments.MASS, 1383.9538229392606, 'lbm')  # WIN

outputs.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS, 28200.322805698346, 'lbm')  # WLGM
outputs.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS, 2698.6740002098945, 'lbm')  # WLGN

outputs.set_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.array([15.68611614]), 'ft')  # EL(5)
outputs.set_val(Aircraft.Nacelle.FINENESS, np.array([1.38269353]))  # FR(5)
outputs.set_val(Aircraft.Nacelle.MASS, 0.0, 'lbm')  # WNAC
nacelle_wetted_area = np.array([498.26822066])  # SWET(5)
outputs.set_val(Aircraft.Nacelle.WETTED_AREA, nacelle_wetted_area, 'ft**2')
outputs.set_val(Aircraft.Nacelle.TOTAL_WETTED_AREA, 3 * nacelle_wetted_area, 'ft**2')

outputs.set_val(Aircraft.Paint.MASS, 0.0, 'lbm')  # WPAINT, WTPNT

outputs.set_val(Aircraft.Propulsion.MASS, 61597.102467771889, 'lbm')  # WPRO
outputs.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 70000.0 * 3, 'lbf')
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 3)
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, 346.93557352, 'lbm')  # WOIL
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 0)
outputs.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, 3)

outputs.set_val(Aircraft.Engine.MASS, 17825.63336233, 'lbm')  # WSP
# In FLOPS BWB, WPOD = 19067.983045931636 because WSTART is scaled by
# WPMSC (Aircraft.Propulsion.MISC_MASS_SCALER) but not in Aviary.
# In Aviary, starter mass and engine controls are scaled later in EngineMiscMass()
# but it does not help for engine POD mass.
# In FLOPS, WSTART = 0.0 for the same reason (WPMSC = 0.0).
outputs.set_val(Aircraft.Engine.POD_MASS, 19593.89025207, 'lbm')  # WPOD
# In FLOPS BWB, WEC = 0.0 because WEC is scaled by WPMSC (Aircraft.Propulsion.MISC_MASS_SCALER)
engine_ctrls_mass = 206.36860226  # WEC
outputs.set_val(Aircraft.Engine.CONTROLS_MASS, engine_ctrls_mass, 'lbm')
outputs.set_val(Aircraft.Engine.ADDITIONAL_MASS, 0.0, 'lbm')  # WPMISC
outputs.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS, 0.0, 'lbm')  # WTHR
outputs.set_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 1526.1294678475103, 'lbm')  # WSTART
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, engine_ctrls_mass, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_MISC_MASS, 0.0, 'lbm')
outputs.set_val(Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, 0.0, 'lbm')  # sum of zeros
outputs.set_val(Aircraft.Propulsion.TOTAL_ENGINE_MASS, 53476.90008698, 'lbm')  # WENG

outputs.set_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 0.0, 'ft')  # EL(3)
outputs.set_val(Aircraft.VerticalTail.FINENESS, 0.11)  # FR(3)
outputs.set_val(Aircraft.VerticalTail.MASS, 0.0, 'lbm')  # WVT
outputs.set_val(Aircraft.VerticalTail.WETTED_AREA, 0.0, 'ft**2')

outputs.set_val(Aircraft.Wing.BWB_AFTBODY_MASS, 20150.78870864, 'lbm')  # W4
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR, 2.68745091)  # FLOPS BT = 2.7108906906553494
outputs.set_val(Aircraft.Wing.BENDING_MATERIAL_MASS, 6313.44762977, 'lbm')  # W1
outputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 69.53953418, 'ft')  # EL(1)
outputs.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 5513.13877521, 'ft**2')  # SFLAP
outputs.set_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 1.0)  # CAYE
outputs.set_val(Aircraft.Wing.FINENESS, 0.11)  # FR(1)
outputs.set_val(Aircraft.Wing.MISC_MASS, 21498.83307778, 'lbm')  # W3
outputs.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS, 38779.21499739, 'lbm')  # W2
outputs.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS, 14152.3734702, 'lbm')  # WSC
outputs.set_val(Aircraft.Wing.ASPECT_RATIO, 3.4488813)  # AR
outputs.set_val(Aircraft.Wing.ASPECT_RATIO_REF, 3.4488813)  # ARREF
outputs.set_val(Aircraft.Wing.MASS, 86742.28126808, 'lbm')  # WWING
outputs.set_val(Aircraft.Wing.ROOT_CHORD, 63.96, 'ft')  # XLW
outputs.set_val(Aircraft.Wing.AREA, 16555.972297926455, 'ft**2')  # SW
outputs.set_val(Aircraft.Wing.LOAD_FRACTION, 0.53107166)  # PCTL
outputs.set_val(Aircraft.Wing.WETTED_AREA, 33816.732336575638, 'ft**2')  # SWET(1)

outputs.set_val(Mission.Design.MACH, 0.800)
outputs.set_val(Mission.Summary.OPERATING_MASS, 455464.65969526308, 'lbm')  # WOWE
outputs.set_val(Mission.Summary.ZERO_FUEL_MASS, 553276.65969526302, 'lbm')  # WZF
outputs.set_val(Mission.Summary.FUEL_MASS, 320822.34030473698, 'lbm')  # FUELM
