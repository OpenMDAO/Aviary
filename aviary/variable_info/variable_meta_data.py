"""
Define meta data associated with variables in the Aviary data hierarchy.
"""

from copy import deepcopy
import numpy as np

from aviary.utils.develop_metadata import add_meta_data
from aviary.variable_info.enums import (
    AircraftTypes,
    EquationsOfMotion,
    FlapType,
    GASPEngineType,
    LegacyCode,
    ProblemType,
    Verbosity,
    AtmosphereModel,
)
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

# ---------------------------
# Meta data associated with variables in the aircraft data hierarchy.
# Please add variables in alphabetical order to match order in variables.py.
#
# ASCII art from http://patorjk.com/software/taag/#p=display&h=0&f=Big&t=
# Super categories such as aircraft and mission are in 'Blocks' font
# Sub categories such as AntiIcing and Wing are in 'Big' font
# Additional sub categories are in 'Small' font
# ---------------------------
_MetaData = {}

# TODO Metadata descriptions should contain which core subsystems that variable appears
#      in. A standardized format for this should be created that takes advantage of
#      newlines, tabs, etc. kind of like a docstring.
# ================================================================================================================================================================
# .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
# | |      __      | || |     _____    | || |  _______     | || |     ______   | || |  _______     | || |      __      | || |  _________   | || |  _________   | |
# | |     /  \     | || |    |_   _|   | || | |_   __ \    | || |   .' ___  |  | || | |_   __ \    | || |     /  \     | || | |_   ___  |  | || | |  _   _  |  | |
# | |    / /\ \    | || |      | |     | || |   | |__) |   | || |  / .'   \_|  | || |   | |__) |   | || |    / /\ \    | || |   | |_  \_|  | || | |_/ | | \_|  | |
# | |   / ____ \   | || |      | |     | || |   |  __ /    | || |  | |         | || |   |  __ /    | || |   / ____ \   | || |   |  _|      | || |     | |      | |
# | | _/ /    \ \_ | || |     _| |_    | || |  _| |  \ \_  | || |  \ `.___.'\  | || |  _| |  \ \_  | || | _/ /    \ \_ | || |  _| |_       | || |    _| |_     | |
# | ||____|  |____|| || |    |_____|   | || | |____| |___| | || |   `._____.'  | || | |____| |___| | || ||____|  |____|| || | |_____|      | || |   |_____|    | |
# | |              | || |              | || |              | || |              | || |              | || |              | || |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'
# ================================================================================================================================================================

#             _           _____                       _   _   _     _                   _
#     /\     (_)         / ____|                     | | (_) | |   (_)                 (_)
#    /  \     _   _ __  | |        ___    _ __     __| |  _  | |_   _    ___    _ __    _   _ __     __ _
#   / /\ \   | | | '__| | |       / _ \  | '_ \   / _` | | | | __| | |  / _ \  | '_ \  | | | '_ \   / _` |
#  / ____ \  | | | |    | |____  | (_) | | | | | | (_| | | | | |_  | | | (_) | | | | | | | | | | | | (_| |
# /_/    \_\ |_| |_|     \_____|  \___/  |_| |_|  \__,_| |_|  \__| |_|  \___/  |_| |_| |_| |_| |_|  \__, |
#                                                                                                    __/ |
#                                                                                                   |___/
# ========================================================================================================

add_meta_data(
    # Note user override
    #    - see also: Aircraft.AirConditioning.MASS_SCALER
    Aircraft.AirConditioning.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(23, 2)', '~WEIGHT.WAC', '~WTSTAT.WSP(23, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Environmental control mass (air conditioning)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.AirConditioning.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(6)', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of air conditioning',
    default_value=1.0,
)

add_meta_data(
    Aircraft.AirConditioning.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WAC', 'MISWT.WAC', 'MISWT.OAC'],
        'FLOPS': 'WTIN.WAC',
    },
    units='unitless',
    desc='air conditioning system mass scaler',
    default_value=1.0,
)

#                     _     _   _____          _
#     /\             | |   (_) |_   _|        (_)
#    /  \     _ __   | |_   _    | |     ___   _   _ __     __ _
#   / /\ \   | '_ \  | __| | |   | |    / __| | | | '_ \   / _` |
#  / ____ \  | | | | | |_  | |  _| |_  | (__  | | | | | | | (_| |
# /_/    \_\ |_| |_|  \__| |_| |_____|  \___| |_| |_| |_|  \__, |
#                                                           __/ |
#                                                          |___/
# ===============================================================

add_meta_data(
    # NOTE user override
    #    - see also: Aircraft.AntiIcing.MASS_SCALER
    Aircraft.AntiIcing.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CW(7)',
        # The following note is for FLOPS
        # ['WTS.WSP(24, 2)', '~WEIGHT.WAI', '~WTSTAT.WSP(24, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Anti-icing system mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.AntiIcing.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WAI', 'MISWT.WAI', 'MISWT.OAI'],
        'FLOPS': 'WTIN.WAI',
    },
    units='unitless',
    desc='anti-icing system mass scaler',
    default_value=1.0,
)

#             _____    _    _
#     /\     |  __ \  | |  | |
#    /  \    | |__) | | |  | |
#   / /\ \   |  ___/  | |  | |
#  / ____ \  | |      | |__| |
# /_/    \_\ |_|       \____/
# ============================

add_meta_data(
    # Note user override
    #    - see also: Aircraft.APU.MASS_SCALER
    Aircraft.APU.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CW(1)',
        # ['WTS.WSP(17, 2)', '~WEIGHT.WAPU', '~WTSTAT.WSP(17, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of auxiliary power unit',
    default_value=0.0,
)

add_meta_data(
    Aircraft.APU.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WAPU', 'MISWT.WAPU', 'MISWT.OAPU'],
        'FLOPS': 'WTIN.WAPU',
    },
    units='unitless',
    desc='mass scaler for auxiliary power unit',
    default_value=1.0,
)

#                     _                   _
#     /\             (_)                 (_)
#    /  \    __   __  _    ___    _ __    _    ___   ___
#   / /\ \   \ \ / / | |  / _ \  | '_ \  | |  / __| / __|
#  / ____ \   \ V /  | | | (_) | | | | | | | | (__  \__ \
# /_/    \_\   \_/   |_|  \___/  |_| |_| |_|  \___| |___/
# =======================================================

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Avionics.MASS_SCALER
    Aircraft.Avionics.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CW(5)',
        # ['WTS.WSP(21, 2)', '~WEIGHT.WAVONC', '~WTSTAT.WSP(21, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Avionics group mass. Includes equipment and installation mass.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Avionics.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WAVONC', 'MISWT.WAVONC', 'MISWT.OAVONC'],
        'FLOPS': 'WTIN.WAVONC',
    },
    units='unitless',
    desc='avionics mass scaler',
    default_value=1.0,
)

#  ____            _     _
# |  _ \          | |   | |
# | |_) |   __ _  | |_  | |_    ___   _ __   _   _
# |  _ <   / _` | | __| | __|  / _ \ | '__| | | | |
# | |_) | | (_| | | |_  | |_  |  __/ | |    | |_| |
# |____/   \__,_|  \__|  \__|  \___| |_|     \__, |
#                                             __/ |
#                                            |___/
# =================================================
add_meta_data(
    Aircraft.Battery.ADDITIONAL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of non energy-storing parts of the battery',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Battery.DISCHARGE_LIMIT,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SOCMIN',
        'FLOPS': None,
    },
    units='unitless',
    desc='default constraint on how far the battery can discharge, as a proportion of '
    'total energy capacity',
    default_value=0.2,
)

add_meta_data(
    Aircraft.Battery.EFFICIENCY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.EFF_BAT', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='battery pack efficiency',
)

add_meta_data(
    Aircraft.Battery.ENERGY_CAPACITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'EBATTAVL', 'FLOPS': None},
    units='kJ',
    desc='total energy the battery can store',
)

add_meta_data(
    Aircraft.Battery.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.WBATTIN',
        'FLOPS': None,
    },
    units='lbm',
    desc='total mass of the battery',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Battery.PACK_ENERGY_DENSITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ENGYDEN',
        'FLOPS': None,
    },
    units='W*h/kg',
    desc='specific energy density of the battery pack',
    default_value=1.0,
)


add_meta_data(
    Aircraft.Battery.PACK_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='mass of the energy-storing components of the battery',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Battery.PACK_VOLUMETRIC_DENSITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='kW*h/L',
    desc='volumetric density of the battery pack',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Battery.VOLUME,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft*3',
    desc='total volume of the battery pack',
    default_value=0.0,
)


#  ____    _                      _              _    __          __  _                     ____                _
# |  _ \  | |                    | |            | |   \ \        / / (_)                   |  _ \              | |
# | |_) | | |   ___   _ __     __| |   ___    __| |    \ \  /\  / /   _   _ __     __ _    | |_) |   ___     __| |  _   _
# |  _ <  | |  / _ \ | '_ \   / _` |  / _ \  / _` |     \ \/  \/ /   | | | '_ \   / _` |   |  _ <   / _ \   / _` | | | | |
# | |_) | | | |  __/ | | | | | (_| | |  __/ | (_| |      \  /\  /    | | | | | | | (_| |   | |_) | | (_) | | (_| | | |_| |
# |____/  |_|  \___| |_| |_|  \__,_|  \___|  \__,_|       \/  \/     |_| |_| |_|  \__, |   |____/   \___/   \__,_|  \__, |
#                                                                                  __/ |                             __/ |
#                                                                                 |___/                             |___/
# ========================================================================================================================

add_meta_data(
    Aircraft.BWB.DETAILED_WING_PROVIDED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Flag if the detailed wing model is provided',
    option=True,
    types=bool,
    default_value=True,
)

add_meta_data(
    Aircraft.BWB.MAX_BAY_WIDTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'FUSEIN.BAYWMX',
        'LEAPS1': None,
    },
    units='ft',
    desc='maximum bay width',
    types=float,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.BWB.MAX_NUM_BAYS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'FUSEIN.NBAYMX',  # ['&DEFINE.FUSEIN.NBAYMX', 'FUSDTA.NBAYMX'],
    },
    units='unitless',
    desc='fixed number of bays',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.BWB.NUM_BAYS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'FUSEIN.NBAY',  # ['&DEFINE.FUSEIN.NBAY', 'FUSDTA.NBAY'],
    },
    units='unitless',
    desc='fixed number of passenger bays',
    types=int,
    multivalue=True,
    option=False,
    default_value=[0],
)

add_meta_data(
    Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.SWP_FB'],
        # ['&DEFINE.FUSEIN.SWPLE', 'FUSDTA.SWPLE'],
        'FLOPS': 'FUSEIN.SWPLE',
    },
    units='deg',
    desc='forebody sweep angle',
    default_value=0.0,
)

#   _____                                      _
#  / ____|                                    | |
# | |        __ _   _ __     __ _   _ __    __| |
# | |       / _` | | '_ \   / _` | | '__|  / _` |
# | |____  | (_| | | | | | | (_| | | |    | (_| |
#  \_____|  \__,_| |_| |_|  \__,_| |_|     \__,_|
# ===============================================
add_meta_data(
    Aircraft.Canard.AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.SCAN',  # ['&DEFINE.WTIN.SCAN', 'EDETIN.SCAN'],
    },
    units='ft**2',
    desc='canard theoretical area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.ASPECT_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.ARCAN',  # ['&DEFINE.WTIN.ARCAN', 'EDETIN.ARCAN'],
    },
    units='unitless',
    desc='canard theoretical aspect ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[-1]',
    },
    units='ft',
    desc='Reynolds characteristic length for the canard',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[-1]',
    },
    units='unitless',
    desc='canard fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRLC',  # ['&DEFINE.AERIN.TRLC', 'XLAM.TRLC', ],
    },
    units='unitless',
    desc='define percent laminar flow for canard lower surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRUC',  # ['&DEFINE.AERIN.TRUC', 'XLAM.TRUC', ],
    },
    units='unitless',
    desc='define percent laminar flow for canard upper surface',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Canard.MASS_SCALER
    Aircraft.Canard.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(5, 2)', '~WEIGHT.WCAN', '~WTSTAT.WSP(5, 2)', '~INERT.WCAN'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of canards',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRCAN',  # ['&DEFINE.WTIN.FRCAN', 'WTS.FRCAN', ],
    },
    units='unitless',
    desc='mass scaler for canard structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Canard.TAPER_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TRCAN',  # ['&DEFINE.WTIN.TRCAN', 'WTS.TRCAN'],
    },
    units='unitless',
    desc='canard theoretical taper ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TCCAN',  # ['&DEFINE.WTIN.TCCAN', 'EDETIN.TCCAN'],
    },
    units='unitless',
    desc='canard thickness-chord ratio',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Canard.WETTED_AREA_SCALER
    Aircraft.Canard.WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['ACTWET.SWTCN', 'MISSA.SWET[-1]'],
    },
    units='ft**2',
    desc='canard wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Canard.WETTED_AREA_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.SWETC',  # ['&DEFINE.AERIN.SWETC', 'AWETO.SWETC', ],
    },
    units='unitless',
    desc='canard wetted area scaler',
    default_value=1.0,
)

#    _____                   _                    _
#   / ____|                 | |                  | |
#  | |        ___    _ __   | |_   _ __    ___   | |  ___
#  | |       / _ \  | '_ \  | __| | '__|  / _ \  | | / __|
#  | |____  | (_) | | | | | | |_  | |    | (_) | | | \__ \
#   \_____|  \___/  |_| |_|  \__| |_|     \___/  |_| |___/
# ========================================================

add_meta_data(
    Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK15', 'FLOPS': None},
    units='unitless',
    desc='technology factor on cockpit controls mass',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Controls.CONTROL_MASS_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELWFC', 'FLOPS': None},
    units='lbm',
    desc='incremental flight controls mass',
    default_value=0,
)

add_meta_data(
    Aircraft.Controls.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFC', 'FLOPS': None},
    units='lbm',
    desc='Flight controls group mass. Contains cockpit controls, automatic flight control system '
    'and system controls.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKSAS', 'FLOPS': None},
    units='lbm',
    desc='mass of stability augmentation system',
    default_value=0,
)

add_meta_data(
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK19', 'FLOPS': None},
    units='unitless',
    desc='technology factor on stability augmentation system mass',
    default_value=1,
)

#   _____                            _____                    _                       _
#  / ____|                          |  __ \                  | |                     | |
# | |       _ __    ___  __      __ | |__) |   __ _   _   _  | |   ___     __ _    __| |
# | |      | '__|  / _ \ \ \ /\ / / |  ___/   / _` | | | | | | |  / _ \   / _` |  / _` |
# | |____  | |    |  __/  \ V  V /  | |      | (_| | | |_| | | | | (_) | | (_| | | (_| |
#  \_____| |_|     \___|   \_/\_/   |_|       \__,_|  \__, | |_|  \___/   \__,_|  \__,_|
#                                                      __/ |
#                                                     |___/
# ======================================================================================

add_meta_data(
    Aircraft.CrewPayload.BAGGAGE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(35,2)', '~WEIGHT.WPBAG', '~WTSTAT.WSP(35,2)', '~INERT.WPBAG'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of passenger baggage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.BPP',  # ['&DEFINE.WTIN.BPP', 'WPAB.BPP'],
    },
    units='lbm',
    desc='baggage mass per passenger',
    option=True,
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.CrewPayload.CABIN_CREW_MASS_SCALER
    Aircraft.CrewPayload.CABIN_CREW_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(28,2)', '~WEIGHT.WSTUAB', '~WTSTAT.WSP(28, 2)', '~INERT.WSTUAB'],
        'FLOPS': None,
    },
    units='lbm',
    desc='total mass of the non-flight crew and their baggage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.CABIN_CREW_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WSTUAB', 'MISWT.WSTUAB', 'MISWT.OSTUAB'],
        'FLOPS': 'WTIN.WSTUAB',
    },
    units='unitless',
    desc='scaler for total mass of the non-flight crew and their baggage',
    default_value=1.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER
    Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(32,2)', '~WEIGHT.WCON', '~WTSTAT.WSP(32,2)', '~INERT.WCON',],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of cargo containers',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WCON', 'MISWT.WCON', 'MISWT.OCON'],
        'FLOPS': 'WTIN.WCON',
    },
    units='unitless',
    desc='Scaler for mass of cargo containers',
    default_value=1.0,
)

add_meta_data(
    Aircraft.CrewPayload.CARGO_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total mass of as-flown cargo',
)

add_meta_data(
    Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(12)', 'FLOPS': None},
    units='lbm',
    desc='mass of catering items per passenger',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER
    Aircraft.CrewPayload.FLIGHT_CREW_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(27, 2)', '~WEIGHT.WFLCRB', '~WTSTAT.WSP(27, 2)', '~INERT.WFLCRB'],
        'FLOPS': None,
    },
    units='lbm',
    desc='total mass of the flight crew and their baggage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.FLIGHT_CREW_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WFLCRB', 'MISWT.WFLCRB', 'MISWT.OFLCRB'],
        'FLOPS': 'WTIN.WFLCRB',
    },
    units='unitless',
    desc='scaler for total mass of the flight crew and their baggage',
    default_value=1.0,
)

add_meta_data(
    Aircraft.CrewPayload.MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.WPPASS',  # ['&DEFINE.WTIN.WPPASS', 'WPAB.WPPASS'],
    },
    units='lbm',
    desc='mass per passenger',
    option=True,
    default_value=165.0,
)

add_meta_data(
    Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.UWPAX', 'FLOPS': None},
    units='lbm',
    desc='total mass of one passenger and their bags',
    option=True,
    default_value=200,
)

add_meta_data(
    Aircraft.CrewPayload.MISC_CARGO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.CARGOF',  # ['&DEFINE.WTIN.CARGOF', 'WTS.CARGOF'],
    },
    units='lbm',
    desc='cargo (other than passenger baggage) carried in fuselage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['&DEFINE.WTIN.NPB', 'WTS.NPB'],
    },
    units='unitless',
    desc='number of business class passengers',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_CABIN_CREW,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Total number of cabin crew. In FLOPS this includes galley and flight attendants',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_ECONOMY_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['&DEFINE.WTIN.NPT', 'WTS.NPT'],
    },
    units='unitless',
    desc='number of economy class passengers',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_FIRST_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['&DEFINE.WTIN.NPF', 'WTS.NPF'],
    },
    units='unitless',
    desc='number of first class passengers.',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NSTU',  # ['&DEFINE.WTIN.NSTU', 'WTS.NSTU'],
    },
    units='unitless',
    desc='number of flight attendants',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_FLIGHT_CREW,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.NFLCR', 'WTS.NFLCR', '~WTSTAT.NFLCR'],
        'FLOPS': 'WTIN.NFLCR',
    },
    units='unitless',
    desc='number of flight crew',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_GALLEY_CREW,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NGALC',  # ['&DEFINE.WTIN.NGALC', 'WTS.NGALC'],
    },
    units='unitless',
    desc='number of galley crew',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_PASSENGERS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,  # 'INGASP.PAX' here we assume previous studies were changing Design.num_pax not as-flown
        'FLOPS': None,  # ['CSTDAT.NSV', '~WEIGHT.NPASS', '~WTSTAT.NPASS'],
    },
    units='unitless',
    desc='total number of passengers',
    option=True,
    default_value=0,
    types=int,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_MASS_TOTAL,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(34, 2)', '~WEIGHT.WPASS', '~WTSTAT.WSP(34, 2)', '~INERT.WPASS'],
        'FLOPS': None,
    },
    units='lbm',
    desc='TBD: total mass of all passengers without their baggage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
    meta_data=_MetaData,
    # note: this GASP variable does not include cargo, but it does include
    # passenger baggage
    historical_name={'GASP': 'INGASP.WPL', 'FLOPS': None},
    units='lbm',
    desc='mass of passenger payload, including passengers, passenger baggage',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override
    #    - see also: Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(31, 2)', '~WEIGHT.WSRV', '~WTSTAT.WSP(31, 2)', '~INERT.WSRV'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of passenger service equipment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(9)', 'FLOPS': None},
    default_value=0.0,
    units='lbm',
    desc='mass of passenger service items mass per passenger',
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WSRV', 'MISWT.WSRV', 'MISWT.OSRV'],
        'FLOPS': 'WTIN.WSRV',
    },
    units='unitless',
    desc='scaler for mass of passenger service equipment',
    default_value=1.0,
)

add_meta_data(
    Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total mass of payload, including passengers, passenger baggage, and cargo',
)

add_meta_data(
    Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(14)', 'FLOPS': None},
    units='lbm',
    desc='unit mass of ULD (unit load device) for cargo handling per passenger',
    default_value=0.0,
    types=float,
    option=True,
)

add_meta_data(
    Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(10)', 'FLOPS': None},
    default_value=1.0,
    units='lbm',
    desc='mass of water per occupant (passengers, pilots, and flight attendants)',
)

add_meta_data(
    Aircraft.CrewPayload.WING_CARGO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.CARGOW',  # ['&DEFINE.WTIN.CARGOW', 'WTS.CARGOW'],
    },
    units='lbm',
    desc='cargo carried in wing',
    default_value=0.0,
)


#   ___               _
#  |   \   ___   ___ (_)  __ _   _ _
#  | |) | / -_) (_-< | | / _` | | ' \
#  |___/  \___| /__/ |_| \__, | |_||_|
#  ====================== |___/ ======

add_meta_data(
    Aircraft.CrewPayload.Design.CARGO_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total mass of cargo flown on design mission',
)

add_meta_data(
    Aircraft.CrewPayload.Design.MAX_CARGO_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.WCARGO',
        # ['WTS.WSP(36,2)', '~WEIGHT.WCARGO', '~WTSTAT.WSP(36,2)', '~INERT.WCARGO',],
        'FLOPS': None,
    },
    units='lbm',
    desc='maximum mass of cargo',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NPB',  # ['&DEFINE.WTIN.NPB', 'WTS.NPB'],
    },
    units='unitless',
    desc='number of business class passengers that the aircraft is designed to accommodate',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NPT',  # ['&DEFINE.WTIN.NPT', 'WTS.NPT'],
    },
    units='unitless',
    desc='number of economy class passengers that the aircraft is designed to accommodate',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.PCT_FC',
        'FLOPS': 'WTIN.NPF',  # ['&DEFINE.WTIN.NPF', 'WTS.NPF'],
    },
    units='unitless',
    desc='number of first class passengers that the aircraft is designed to accommodate. In GASP, the input is the percentage of total number of passengers.',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_PASSENGERS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.PAX',  # number of passenger seats excluding crew
        'FLOPS': None,  # ['CSTDAT.NSV', '~WEIGHT.NPASS', '~WTSTAT.NPASS'],
    },
    units='unitless',
    desc='total number of passengers that the aircraft is designed to accommodate',
    option=True,
    default_value=0,
    types=int,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.NBABR'},
    units='unitless',
    desc='Number of business class seats abreast.',
    types=int,
    option=True,
    default_value=5,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SAB', 'FLOPS': 'FUSEIN.NTABR'},
    units='unitless',
    desc='Number of economy class seats abreast.',
    types=int,
    option=True,
    default_value=6,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.NFABR'},
    units='unitless',
    desc='Number of first class seats abreast.',
    types=int,
    option=True,
    default_value=4,
)

add_meta_data(
    Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.BPITCH'},
    units='inch',
    desc='pitch of the business class seats.',
    option=True,
    default_value=39.0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.PS', 'FLOPS': 'FUSEIN.TPITCH'},
    units='inch',
    desc='pitch of the economy class seats.',
    option=True,
    default_value=32.0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.FPITCH'},
    units='inch',
    desc='pitch of the first class seats.',
    option=True,
    default_value=61.0,
)

#  _____                 _
# |  __ \               (_)
# | |  | |   ___   ___   _    __ _   _ __
# | |  | |  / _ \ / __| | |  / _` | | '_ \
# | |__| | |  __/ \__ \ | | | (_| | | | | |
# |_____/   \___| |___/ |_|  \__, | |_| |_|
#                             __/ |
#                            |___/
# =========================================
add_meta_data(
    Aircraft.Design.BASE_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.SBASE',
        #  [  # inputs
        #      '&DEFINE.AERIN.SBASE', 'EDETIN.SBASE',
        #      # outputs
        #      'MISSA.SBASE', 'MISSA.SBASEX',
        #  ],
    },
    units='ft**2',
    desc='Aircraft base area (total exit cross-section area minus inlet '
    'capture areas for internally mounted engines)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.CG_DELTA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELCG', 'FLOPS': None},
    units='unitless',
    desc='allowable center-of-gravity (cg) travel as a fraction of the mean aerodynamic chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.CHARACTERISTIC_LENGTHS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL',
    },
    units='ft',
    desc='Reynolds characteristic length for each component',
)

add_meta_data(
    Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKCC', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of cockpit controls',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.COMPRESSIBILITY_DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCMPC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='compressibility aero calibration factor',
)

add_meta_data(
    Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='if true, use empirical tail volume coefficient equation. This is '
    'true if VBARHX is 0 in GASP.',
)

add_meta_data(
    Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='if true, use empirical tail volume coefficient equation. This is '
    'true if VBARVX is 0 in GASP.',
)

add_meta_data(
    Aircraft.Design.CRUISE_ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRALT', 'FLOPS': None},
    units='ft',
    option=True,
    default_value=25000.0,
    desc='design mission cruise altitude',
)

add_meta_data(
    Aircraft.Design.CRUISE_MACH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'CONFIN.VCMN',
        #  [  # inputs
        #      '&DEFINE.CONFIN.VCMN', 'PARVAR.DVD(1,8)',
        #      # outputs
        #      'CONFIG.VCMN', 'CONFIG.DVA(8)', '~FLOPS.DVA(8)', '~ANALYS.DVA(8)',
        #      # other
        #      'MISSA.VCMIN',
        #  ],
    },
    units='unitless',
    desc='aircraft cruise Mach number',
    default_value=0.0,  # TODO: required
)

add_meta_data(
    Aircraft.Design.DRAG_COEFFICIENT_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELCD', 'FLOPS': None},
    units='unitless',
    desc='increment to the profile drag coefficient',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.DRAG_DIVERGENCE_SHIFT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SCFAC', 'FLOPS': None},
    units='unitless',
    desc='shift in drag divergence Mach number due to supercritical design',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.DRAG_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Drag polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.EMERGENCY_EQUIPMENT_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(11)', 'FLOPS': None},
    units='lbm',
    desc='mass of emergency equipment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.EMPENNAGE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
    },
    units='lbm',
    desc='Empennage group mass. Contains mass of canards, horizontal/vertical stabilizers '
    'and fins, and ventral fins, and any supporting structure for mounted engines on those surfaces.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.EMPTY_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFMSS.MISSIN.DOWE', '&FLOPS.RERUN.DOWE', 'ESB.DOWE'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Empty mass of the aircraft. Includes structure group, propulsion group, and total systems '
    'and equipment mass.',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Design.EMPTY_MASS_MARGIN_SCALER
    Aircraft.Design.EMPTY_MASS_MARGIN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'DARM.WMARG',
    },
    units='lbm',
    desc='empty mass margin',
    default_value=0.0,
)

add_meta_data(
    # Note users must enable this feature, or the associated calculation is
    # discarded. Default to 0.0
    Aircraft.Design.EMPTY_MASS_MARGIN_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.EWMARG',  # ['&DEFINE.WTIN.EWMARG', 'DARM.EWMARG'],
    },
    units='unitless',
    desc='empty mass margin scaler',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.EXCRESCENCE_DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FEXCRT', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='excrescence aero drag factor',
)

add_meta_data(
    Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
    historical_name={
        'GASP': None,
        'FLOPS': None,
    },
    meta_data=_MetaData,
    units='lbm',
    desc='Total mass of all user-defined external subsystems. These are bookkept as part of empty '
    'mass.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR',
    },
    units='unitless',
    desc='table of component fineness ratios',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.GROSS_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.WG',
        # ['&DEFINE.WTIN.DGW', 'WTS.DGW', '~WEIGHT.DG', '~WWGHT.DG'],
        'FLOPS': 'WTIN.DGW',
    },
    units='lbm',
    desc='Design gross mass of the aircraft. Includes zero fuel mass plus useable fuel.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.IJEFF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.IJEFF', 'FLOPS': None},
    desc='A flag used by Jeff V. Bowles to debug GASP code during his 53 years supporting the '
    'development of GASP. This flag is planted here to thank him for his hard work and dedication, '
    "Aviary wouldn't be what it is today without his help.",
)

add_meta_data(
    Aircraft.Design.INTERFERENCE_DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCKIC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='interference aero calibration factor (including technology factor INGASP.FCKIT)',
)

# TODO expected types and default value?
add_meta_data(
    Aircraft.Design.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.TRL',
    },
    units='unitless',
    desc='table of percent laminar flow over lower component surfaces',
)

# TODO expected types and default values?
add_meta_data(
    Aircraft.Design.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.TRU',
    },
    units='unitless',
    desc='table of percent laminar flow over upper component surfaces',
)

add_meta_data(
    # Note user override (no scaling)
    Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.WRATIO',  # ['&DEFINE.AERIN.WRATIO', 'ESB.WRATIO'],
    },
    units='unitless',
    desc='ratio of maximum landing mass to maximum takeoff mass',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override (no scaling)
    Aircraft.Design.LIFT_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.FCLDES',
        #  [  # inputs
        #      '&DEFINE.AERIN.FCLDES', 'OSWALD.FCLDES',
        #      # outputs
        #      '~EDET.CLDES', '~CLDESN.CLDES', '~MDESN.CLDES'
        #  ],
    },
    units='unitless',
    desc='Fixed design lift coefficient. If input, overrides design lift '
    'coefficient computed by EDET.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.CLMWFU', 'INGASP.CLMAX'],
        'FLOPS': None,
    },
    units='unitless',
    desc='maximum lift coefficient from flaps model when flaps are up (not deployed)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.LIFT_CURVE_SLOPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLALPH', 'FLOPS': None},
    units='1/rad',
    desc='lift curve slope at cruise Mach number',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.FSA7C',
        'FLOPS': 'MISSIN.FCDI',  # '~DRGFCT.FCDI',
    },
    units='unitless',
    default_value=1.0,
    desc='Scaling factor for lift-dependent drag coefficient',
)

add_meta_data(
    Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Lift dependent drag polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Lift independent drag polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.LIFT_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Lift polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    # NOTE: user override (no scaling)
    Aircraft.Design.MACH,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CRMACH',
        'FLOPS': 'AERIN.FMDES',
        #  [  # inputs
        #      '&DEFINE.AERIN.FMDES', 'OSWALD.FMDES'
        #      # outputs
        #      '~EDET.DESM', '~MDESN.DESM'
        #  ],
    },
    units='unitless',
    desc='aircraft design Mach number',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.THEMAX', 'FLOPS': None},
    units='deg',
    desc='maximum fuselage pitch allowed',
    default_value=15,
)

add_meta_data(
    Aircraft.Design.MAX_STRUCTURAL_SPEED,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VMLFSL', 'FLOPS': None},
    units='mi/h',
    desc='maximum structural design flight speed in miles per hour',
    default_value=0,
)

add_meta_data(
    Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CATD', 'FLOPS': None},
    option=True,
    default_value=3,
    types=int,
    units='unitless',
    desc='part 25 structural category',
)

add_meta_data(
    Aircraft.Design.PERCENT_EXCRESCENCE_DRAG,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.PCT_EXCR', 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=0.0,
    desc='excrescence drag as percentage of fuselage, wing, nacelle, (winglet), empennage and strut',
)

add_meta_data(
    Aircraft.Design.RANGE,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ARNGE',
        # ['&DEFINE.CONFIN.DESRNG', 'CONFIG.DESRNG'],
        'FLOPS': 'CONFIN.DESRNG',
    },
    units='NM',
    desc='The design range of the aircraft used for sizing of FLOPS based subsystems and mission target length if not provided in phase_info',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    option=True,
    default_value=False,
    types=bool,
    units='unitless',
    desc='eliminates discontinuities in GASP-based mass estimation code if true',
)

add_meta_data(
    Aircraft.Design.STATIC_MARGIN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STATIC', 'FLOPS': None},
    units='unitless',
    desc='aircraft static margin as a fraction of mean aerodynamic chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.STRUCTURAL_MASS_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELWST', 'FLOPS': None},
    units='lbm',
    desc='structural mass increment that is added (or removed) after the structural mass is calculated',
    default_value=0,
)

add_meta_data(
    Aircraft.Design.STRUCTURE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(9, 2)', '~WEIGHT.WSTRCT', '~WTSTAT.WSP(9, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Total structure group mass. Includes the following groups: wing, epennage, fuselage, '
    'landing gear, air induction.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'MISSIN.FCDSUB',  # '~DRGFCT.FCDSUB',
    },
    units='unitless',
    default_value=1.0,
    desc='Scaling factor for subsonic drag',
)

add_meta_data(
    Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'MISSIN.FCDSUP',  # '~DRGFCT.FCDSUP',
    },
    units='unitless',
    default_value=1.0,
    desc='Scaling factor for supersonic drag',
)

add_meta_data(
    Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(25, 2)', '~WEIGHT.WSYS', '~WTSTAT.WSP(25, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Systems and equipment group mass. Includes flight controls, auxilary power, instruments, '
    'hydraulics, pneumatics, electrical, avionics, furnishings and equipment, environmental control, '
    'and anti-icing mass.',
    default_value=0.0,
)

# TODO intermediate calculated values with no uses by other systems may not belong in the
#      variable hierarchy
add_meta_data(
    # Note in FLOPS, this is the same variable as
    # Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, because FLOPS overwrite the
    # value during calculations; in Aviary, these must be separate variables
    Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Total systems & equipment group mass without additional 1% of empty mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.THRUST_TAKEOFF_PER_ENG,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.THROFF',
        # LEAPS1 used the average thrust_takeoff of all operational engines
        # actually on the airplane, possibly after resizing (as with FLOPS)
    },
    units='lbf',
    desc='Thrust per engine, used for energy state simple takeoff calculation',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.THRUST_TO_WEIGHT_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
        # NOTE TWR != THRUST_TO_WEIGHT_RATIO because Aviary\'s value is the actual T/W, while TWR is
        #      the desired T/W ratio
        # 'FLOPS': 'CONFIN.TWR',
    },
    units='unitless',
    default_value=0.0,
    types=float,
    desc='ratio of total sea-level-static thrust to aircraft takeoff gross weight',
)

add_meta_data(
    Aircraft.Design.TOTAL_WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WEIGHT.TWET',
    },
    units='ft**2',
    desc='total aircraft wetted area',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override (no scaling)
    Aircraft.Design.TOUCHDOWN_MASS_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.WLDG',
        #  [  # inputs
        #      '&DEFINE.WTIN.WLDG', 'WTS.WLDG',
        #      # outputs
        #      'CMODLW.WLDGO',
        #  ],
    },
    units='lbm',
    desc='Maximum mass at touchdown used to size landing gear',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.IHWB', 'FLOPS': ['OPTION.IFITE']},
    units='unitless',
    types=AircraftTypes,
    option=True,
    default_value=AircraftTypes.TRANSPORT,
    desc='aircraft type: BWB for blended wing body, transport otherwise',
)

add_meta_data(
    Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER,
    meta_data=_MetaData,
    historical_name={'GASP': 'CATD', 'FLOPS': None},
    option=True,
    default_value=False,
    types=bool,
    units='unitless',
    desc='if true, ULF (ultimate load factor) is forced to be calculated from '
    'the maneuver load factor, even if the gust load factor is larger. '
    'This was set to true with a negative CATD in GASP.',
)

add_meta_data(
    Aircraft.Design.USE_ALT_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.IALTWT',
    },
    units='unitless',
    desc='control whether the alternate mass equations are to be used or not',
    option=True,
    types=bool,
    default_value=False,
)

add_meta_data(
    Aircraft.Design.WETTED_AREAS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.SWET',
    },
    units='ft**2',
    desc='table of component wetted areas',
)

add_meta_data(
    Aircraft.Design.WING_LOADING,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.WGS', 'INGASP.WOS'],
        'FLOPS': None,  # 'MISSA.SWET',
    },
    default_value=0,
    types=float,
    units='lbf/ft**2',
    desc='ratio of aircraft gross takeoff weight to projected wing area',
)

add_meta_data(
    Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'MISSIN.FCDO',  # '~DRGFCT.FCDO',
    },
    units='unitless',
    default_value=1.0,
    desc='Scaling factor for zero-lift drag coefficient',
)

#
#  ______   _                 _            _                  _
# |  ____| | |               | |          (_)                | |
# | |__    | |   ___    ___  | |_   _ __   _    ___    __ _  | |
# |  __|   | |  / _ \  / __| | __| | '__| | |  / __|  / _` | | |
# | |____  | | |  __/ | (__  | |_  | |    | | | (__  | (_| | | |
# |______| |_|  \___|  \___|  \__| |_|    |_|  \___|  \__,_| |_|
# ==============================================================

add_meta_data(
    Aircraft.Electrical.HAS_HYBRID_SYSTEM,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='if true there is an augmented electrical system',
)

add_meta_data(
    Aircraft.Electrical.HYBRID_CABLE_LENGTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.LCABLE', 'FLOPS': None},
    units='ft',
    desc='length of cable for hybrid electric augmented system',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override
    #    - see also: Aircraft.Electrical.MASS_SCALER
    Aircraft.Electrical.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(20, 2)', '~WEIGHT.WELEC', '~WTSTAT.WSP(20, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of the electrical system',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Electrical.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WELEC', 'MISWT.WELEC', 'MISWT.OELEC'],
        'FLOPS': 'WTIN.WELEC',
    },
    units='unitless',
    desc='mass scaler for the electrical system',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(15)', 'FLOPS': None},
    units='lbm',
    desc='electrical system weight per passenger. In GASP, default 16.0',
    default_value=0.0,
)

#  ______                   _
# |  ____|                 (_)
# | |__     _ __     __ _   _   _ __     ___
# |  __|   | '_ \   / _` | | | | '_ \   / _ \
# | |____  | | | | | (_| | | | | | | | |  __/
# |______| |_| |_|  \__, | |_| |_| |_|  \___|
#                    __/ |
#                   |___/
# ===========================================

# Note user override
#    - see also: Aircraft.Engine.ADDITIONAL_MASS_FRACTION
add_meta_data(
    Aircraft.Engine.ADDITIONAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='additional engine mass not counted by existing categories (such as engine control and '
    'starter mass in FLOPS). In GASP, this is engine installation mass.',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.ADDITIONAL_MASS_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SKPEI',
        'FLOPS': 'WTIN.WPMISC',  # ['&DEFINE.WTIN.WPMISC', 'FAWT.WPMISC'],
    },
    units='unitless',
    option=True,
    desc='fraction of (scaled) engine mass used to calculate additional engine mass (see '
    'Aircraft.Engine.ADDITIONAL_MASS)',
    types=(float, int, np.ndarray),
    multivalue=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'MISSIN.FLEAK',
    },
    option=True,
    units='lbm/h',
    desc='Additional constant fuel flow. This value is not scaled with the engine',
    default_value=0.0,
    multivalue=True,
)

# TODO there should be a GASP name that pairs here
add_meta_data(
    Aircraft.Engine.DATA_FILE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'ENGDIN.EIFILE'},
    units='unitless',
    types=str,
    default_value=None,
    option=True,
    desc='filepath to data file containing engine performance tables',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FIXED_RPM,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='rpm',
    default_value=1.0,
    desc='RPM the engine is set to be running at. Overrides RPM provided by '
    'engine model or chosen by optimizer. Typically used when pairing a motor or '
    'turboshaft using a fixed operating RPM with a propeller.',
    multivalue=True,
    option=True,
)

add_meta_data(
    Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.FIDMAX',
    },
    units='unitless',
    option=True,
    default_value=1.0,
    desc='If Aircraft.Engine.GENERATE_FLIGHT_IDLE is True, bounds engine '
    'performance outputs (other than thrust) at flight idle to be below a '
    'decimal fraction of the max value of that output produced by the engine '
    'at each flight condition.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.FIDMIN',
    },
    units='unitless',
    option=True,
    default_value=0.08,
    desc='If Aircraft.Engine.GENERATE_FLIGHT_IDLE is True, bounds engine '
    'performance outputs (other than thrust) at flight idle to be above a '
    'decimal fraction of the max value of that output produced by the engine '
    'at each flight condition.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=0.0,
    desc='If Aircraft.Engine.GENERATE_FLIGHT_IDLE is True, defines idle thrust '
    'condition as a decimal fraction of max thrust produced by the engine at each '
    'flight condition.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.DFFAC',
    },
    units='unitless',
    option=True,
    desc='Constant term in fuel flow scaling equation',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.FFFAC',
    },
    units='unitless',
    desc='Linear term in fuel flow scaling equation',
    default_value=0.0,
    option=True,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.GENERATE_FLIGHT_IDLE,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.IDLE',
    },
    meta_data=_MetaData,
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='If True, generate flight idle data by extrapolating from engine data. Flight '
    'idle is defined as engine performance when thrust is reduced to the level '
    'defined by Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION. Other engine outputs are '
    'extrapolated to this thrust level, bounded by '
    'Aircraft.Engine.FLIGHT_IDLE_MIN_FRACT and Aircraft.Engine.FLIGHT_IDLE_MAX_FRACT',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.GEOPOTENTIAL_ALT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.IGEO',
    },
    units='unitless',
    option=True,
    desc='If True, engine deck altitudes are geopotential and will be converted to '
    'geometric altitudes. If False, engine deck altitudes are geometric.',
    types=bool,
    default_value=False,
    multivalue=True,
)

# Global hybrid throttle is also False by default to account for parallel-hybrid engines
# that can't operate at every power level at every condition due to other constraints
add_meta_data(
    Aircraft.Engine.GLOBAL_HYBRID_THROTTLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Flag for engine decks if the range of provided hybrid throttles is consistent '
    'across all flight conditions (e.g. the maximum hybrid throttle seen in the entire '
    'deck is 1.0, but a given flight condition only goes to 0.9 -> GLOBAL_HYBRID_THROTTLE '
    '= TRUE means the engine can be extrapolated out to 1.0 at that point. If '
    "GLOBAL_HYBRID_THROTTLE is False, then each flight condition's hybrid throttle range is "
    'individually normalized from 0 to 1 independent of other points on the deck).',
    default_value=False,
    types=bool,
    option=True,
    multivalue=True,
)

# TODO Disabling global throttle ranges is preferred (therefore default) to prevent
# unintended extrapolation, but breaks missions using GASP-based engines that have uneven
# throttle ranges (need t4 constraint on mission to truly fix).
add_meta_data(
    Aircraft.Engine.GLOBAL_THROTTLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Flag for engine decks if the range of provided throttles is consistent '
    'across all flight conditions (e.g. the maximum throttle seen in the entire '
    'deck is 1.0, but a given flight condition only goes to 0.9 -> GLOBAL_THROTTLE '
    '= TRUE means the engine can be extrapolated out to 1.0 at that point. If '
    "GLOBAL_THROTTLE is False, then each flight condition's throttle range is "
    'individually normalized from 0 to 1 independent of other points on the deck).',
    default_value=False,
    types=bool,
    option=True,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.IGNORE_NEGATIVE_THRUST,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.NONEG',
    },
    option=True,
    units='unitless',
    default_value=False,
    types=bool,
    desc='If False, all input or generated points are used, otherwise points in the '
    'engine deck with negative net thrust are ignored.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.INLET_AREA_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    default_value=0.0002,  # default in GASP
    types=float,
    desc='engine inlet area coefficient. Suggested values: 0.000375 for modern engines.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.INTERPOLATION_METHOD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value='slinear',
    types=str,
    desc="method used for interpolation on an engine deck's data file, allowable values are "
    'table methods from openmdao.components.interp_util.interp. Engine models only use the '
    'methods avilable to the MetaModelSemiStructuredComp component. These are listed here: '
    'https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/'
    'metamodelsemistructured_comp.html',
    multivalue=True,
)

# TODO if altitude is more robust, then we can modify how EngineDeck sorts things
add_meta_data(
    Aircraft.Engine.INTERPOLATION_SORT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value='mach',
    types=str,
    desc='Specify the first interpolation variable in the semi-structured metamodel. '
    'Choose from mach or altitude. Mach is usually the first column in the deck, but '
    'altitude is more robust for semi-structured data.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['WTS.WSP(10, 2)', '~WTSTAT.WSP(10, 2)'],
    },
    units='lbm',
    desc='Scaled mass of a single engine. Engine mass includes installation mass, accessory gear '
    'boxes & drive, exhaust system, engine cooling, water injection, engien controls starting '
    'system, propeller/fan installation, lubricating system, and the drive system. Drive '
    'system mass contains gearboxes including lubrication and rotor brakes, transmission drive, '
    'rotor shaft, and gas drive. Fuel system is bookept as a separate line item in the propulsion '
    'group. For nonconventional engines, such as all-electric, engine mass should also contain '
    'masses appropriate for per-engine mass bookeeping (such as motor mass).',
    default_value=0.0,
    multivalue=True,
)

# TODO FLOPS based equation scale factor needs to be separated out into a different
#      variable, which will also eliminate logic branch in engine.py where two
#      different equations are used depending on the value of MASS_SCALER
add_meta_data(
    Aircraft.Engine.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CK5',
        'FLOPS': 'WTIN.EEXP',  # '~WEIGHT.EEXP',
    },
    units='unitless',
    desc='scaler for engine mass',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.MASS_SPECIFIC,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SWSLS', 'FLOPS': None},
    units='lbm/lbf',
    desc='specific mass of one engine (engine weight/SLS thrust)',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.NUM_ENGINES,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ENP',
        'FLOPS': None,  # ['~ANALYS.NENG', 'LANDG.XENG', ],
    },
    units='unitless',
    desc='total number of engines per model on the aircraft (fuselage, wing, or otherwise)',
    types=int,
    multivalue=True,
    option=True,
    default_value=2,
)

add_meta_data(
    Aircraft.Engine.NUM_FUSELAGE_ENGINES,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NEF',  # ['&DEFINE.WTIN.NEF', 'EDETIN.NEF'],
    },
    units='unitless',
    desc='number of fuselage mounted engines per model',
    option=True,
    types=int,
    multivalue=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Engine.NUM_WING_ENGINES,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.NEW', 'EDETIN.NEW', '~WWGHT.NEW'],
        'FLOPS': 'WTIN.NEW',
    },
    units='unitless',
    desc='number of wing mounted engines per model',
    option=True,
    types=int,
    multivalue=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Engine.POD_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['~WEIGHT.WPOD', '~WWGHT.WPOD'],
    },
    units='lbm',
    desc='engine pod mass including nacelles',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.POD_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK14', 'FLOPS': None},
    units='unitless',
    desc='technology factor on mass of engine pods',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.POSITION_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKEPOS', 'FLOPS': None},
    units='unitless',
    desc='engine position factor',
    default_value=0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.PYLON_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FPYL', 'FLOPS': None},
    units='unitless',
    desc='factor for turbofan engine pylon mass',
    default_value=0.7,
    multivalue=True,
)

# NOTE This unscaled turbine (engine) weight is an input provided by the user, and is not
#      an override. It is scaled by Aircraft.Engine.SCALE_FACTOR (a calculated value) to
#      produce Aircraft.Engine.MASS
add_meta_data(
    Aircraft.Engine.REFERENCE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.WENG',  # '~WEIGHT.WENG',
    },
    units='lbm',
    desc='Unscaled mass of a single engine. See Aircraft.Engine.MASS for breakdown of what is '
    'included in engine mass.',
    default_value=0.0,
    option=True,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.REFERENCE_SLS_THRUST,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.FN_REF',  # no FN_REF in GASP
        'FLOPS': 'WTIN.THRSO',
    },
    units='lbf',
    desc='Maximum sea-level static thrust of an unscaled engine. Optional. In '
    'EngineDecks, reference thrust will be found from performance data if not provided '
    'by user. User-provided values override SLS point found in performance data.',
    default_value=0.0,
    option=True,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.RPM_DESIGN,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INPROP.XNMAX',  # maximum engine speed, rpm
        'FLOPS': None,
    },
    units='rpm',
    desc='the designed output RPM from the engine for fixed-RPM shafts',
    multivalue=True,
    option=True,
)

add_meta_data(
    Aircraft.Engine.SCALE_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='A scaling factor used to scale engine performance data during mission analysis.',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.SCALE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
    },
    desc='Toggle for enabling scaling of engine mass based on Aircraft.Engine.SCALE_FACTOR',
    option=True,
    types=bool,
    multivalue=True,
    default_value=True,
)

add_meta_data(
    Aircraft.Engine.SCALED_SLS_THRUST,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.THIN',
        'FLOPS': 'CONFIN.THRUST',
    },
    units='lbf',
    desc='Maximum sea-level static thrust of an engine after scaling. Optional for '
    'EngineDecks if Aircraft.Engine.SCALE_FACTOR is provided, in which case this '
    'variable is computed.',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CKFF',
        'FLOPS': 'ENGDIN.FFFSUB',
    },
    units='unitless',
    desc='scaling factor on fuel flow when Mach number is subsonic',
    default_value=1.0,
    option=True,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CKFF',
        'FLOPS': 'ENGDIN.FFFSUP',
    },
    units='unitless',
    desc='scaling factor on fuel flow when Mach number is supersonic',
    default_value=1.0,
    option=True,
    multivalue=True,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER
    Aircraft.Engine.THRUST_REVERSERS_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(11, 2)', '~WEIGHT.WTHR', '~WTSTAT.WSP(11, 2)', '~INERT.WTHR'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of thrust reversers on engines',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    # Note users must enable this feature, or the associated calculation is
    # discarded
    Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WTHR', 'MISWT.WTHR', 'MISWT.OTHR'],
        'FLOPS': 'WTIN.WTHR',
    },
    units='unitless',
    desc='scaler for mass of thrust reversers on engines. In FLOPS default to 0.0',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.NTYE', 'FLOPS': None},
    option=True,
    default_value=GASPEngineType.TURBOJET,
    types=GASPEngineType,
    multivalue=True,
    units='unitless',
    desc='specifies engine type used for GASP-based engine mass calculation',
)

add_meta_data(
    Aircraft.Engine.WING_LOCATIONS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.YP',
        'FLOPS': 'WTIN.ETAE',  # ['&DEFINE.WTIN.ETAE', 'WDEF.ETAE'],
    },
    units='unitless',
    desc='Engine wing mount locations as fractions of semispan; (NUM_WING_ENGINES)/2 values '
    'are input',
    types=(float, list, np.ndarray),
    default_value=[0.0],
    multivalue=True,
)

#   ___                      _
#  / __|  ___   __ _   _ _  | |__   ___  __ __
# | (_ | / -_) / _` | | '_| | '_ \ / _ \ \ \ /
#  \___| \___| \__,_| |_|   |_.__/ \___/ /_\_\
# ============================================

add_meta_data(
    Aircraft.Engine.Gearbox.EFFICIENCY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='The efficiency of the gearbox.',
    default_value=1.0,
    multivalue=True,
)
add_meta_data(
    Aircraft.Engine.Gearbox.GEAR_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},  # 1 / INPROP.GR
    units='unitless',
    desc='Reduction gear ratio, or the ratio of the RPM_in divided by the RPM_out for the gearbox.',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Gearbox.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='The mass of the gearbox.',
    default_value=0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INPROP.HPMSLS',  # max sea level static horsepower, hp
        'FLOPS': None,
    },
    units='hp',
    desc='A guess for the maximum power that will be transmitted through the gearbox during the mission (max shp input).',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Gearbox.SPECIFIC_TORQUE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf*ft/lbm',
    desc='The specific torque of the gearbox, used to calculate gearbox mass. ',
    default_value=100,
    multivalue=True,
)

#  __  __         _
# |  \/  |  ___  | |_   ___   _ _
# | |\/| | / _ \ |  _| / _ \ | '_|
# |_|  |_| \___/  \__| \___/ |_|
# ================================

add_meta_data(
    Aircraft.Engine.Motor.DATA_FILE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'None'},
    units='unitless',
    types=str,
    default_value=None,
    option=True,
    desc='filepath to data file containing electric motor performance table',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Motor.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'WMOTOR', 'FLOPS': None},
    units='lbm',
    desc='Total motor mass (considers number of motors)',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Motor.TORQUE_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf*ft',
    desc='Max torque value that can be output from a single motor. Used to determine '
    'motor mass in pre-mission',
    multivalue=True,
)

#   ___                            _   _
#  | _ \  _ _   ___   _ __   ___  | | | |  ___   _ _
#  |  _/ | '_| / _ \ | '_ \ / -_) | | | | / -_) | '_|
#  |_|   |_|   \___/ | .__/ \___| |_| |_| \___| |_|
#                    |_|
# ===================================================

add_meta_data(
    Aircraft.Engine.Propeller.ACTIVITY_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.AF', 'FLOPS': None},
    units='unitless',
    desc='propeller actitivty factor per Blade (Range: 80 to 200)',
    default_value=0.0,
    multivalue=True,
)

# NOTE if FT < 0, this bool is true, if >= 0, this is false and the value of FT is used
# as the installation loss factor
add_meta_data(
    Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.FT', 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=True,
    types=bool,
    multivalue=True,
    desc='if true, compute installation loss factor based on blockage factor',
)

add_meta_data(
    Aircraft.Engine.Propeller.DATA_FILE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    types=str,
    default_value=None,
    option=True,
    desc='filepath to data file containing propeller data map',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.DIAMETER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.DPROP', 'FLOPS': None},
    units='ft',
    desc='propeller diameter',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.CLI', 'FLOPS': None},
    units='unitless',
    desc='propeller blade integrated design lift coefficient (Range: 0.3 to 0.8)',
    default_value=0.5,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.MASS,
    meta_data=_MetaData,
    # TODO Check if GASP has a variable for this
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of propellers on engine (sum of all blades)',
    option=False,
    types=float,
    multivalue=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Engine.Propeller.NUM_BLADES,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.BL', 'FLOPS': None},
    units='unitless',
    desc='number of blades per propeller',
    option=True,
    types=int,
    multivalue=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Engine.Propeller.TIP_MACH_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,  # TODO this needs verification
        'FLOPS': None,
    },
    units='unitless',
    desc='maximum allowable Mach number at propeller tip (based on helical speed)',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.TIP_SPEED_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INPROP.TSPDMX', 'INPROP.TPSPDMXe'],
        'FLOPS': None,
    },
    units='ft/s',
    desc='maximum allowable propeller linear tip speed',
    default_value=800.0,
    multivalue=True,
)

#  ______   _
# |  ____| (_)
# | |__     _   _ __    ___
# |  __|   | | | '_ \  / __|
# | |      | | | | | | \__ \
# |_|      |_| |_| |_| |___/
# ===========================

add_meta_data(
    Aircraft.Fins.AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.SFIN',  # ['&DEFINE.WTIN.SFIN', 'WTS.SFIN'],
    },
    units='ft**2',
    desc='vertical fin theoretical area',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Fins.MASS_SCALER
    Aircraft.Fins.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(4, 2)', '~WEIGHT.WFIN', '~WTSTAT.WSP(4, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of vertical fins',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fins.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRFIN',  # ['&DEFINE.WTIN.FRFIN', 'WTS.FRFIN'],
    },
    units='unitless',
    desc='mass scaler for fin structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Fins.NUM_FINS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NFIN',  # ['&DEFINE.WTIN.NFIN', 'WTS.NFIN'],
    },
    units='unitless',
    desc='number of fins',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Fins.TAPER_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TRFIN',  # ['&DEFINE.WTIN.TRFIN', 'WTS.TRFIN'],
    },
    units='unitless',
    desc='vertical fin theoretical taper ratio',
    default_value=0.0,
)

#  ______                  _
# |  ____|                | |
# | |__     _   _    ___  | |
# |  __|   | | | |  / _ \ | |
# | |      | |_| | |  __/ | |
# |_|       \__,_|  \___| |_|
# ===========================

add_meta_data(
    Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FULAUX',  # ['&DEFINE.WTIN.FULAUX', 'FAWT.FULAUX'],
    },
    units='lbm',
    desc='fuel capacity of the auxiliary tank',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.BURN_PER_PASSENGER_MILE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/NM',
    desc='average fuel burn per passenger per mile flown',
)

add_meta_data(
    Aircraft.Fuel.DENSITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FUELD', 'FLOPS': 'WTIN.FULDEN'},
    units='lbm/galUS',
    desc='fuel density (jet fuel typical density of 6.7 lbm/galUS used in the calculation of wing_capacity'
    '(if wing_capacity is not input) and in the calculation of fuel system weight.',
    default_value=6.7,
)

add_meta_data(
    # NOTE: user override
    #    - see also: Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER
    Aircraft.Fuel.FUEL_SYSTEM_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(13, 2)', '~WEIGHT.WFSYS', '~WTSTAT.WSP(13, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Fuel system mass. Includes tanks (both protected and unprotected), plumbing, and '
    'similar masses.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKFS', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of fuel system',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CK21',
        # ['&DEFINE.WTIN.WFSYS', 'MISWT.WFSYS', 'MISWT.OFSYS'],
        'FLOPS': 'WTIN.WFSYS',
    },
    units='unitless',
    desc='scaler for fuel system mass',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.FULFMX', 'WTS.FULFMX', '~WEIGHT.FUFU'],
        'FLOPS': 'WTIN.FULFMX',
    },
    units='lbm',
    desc='fuel capacity of the fuselage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'WTIN.IFUFU'},
    units='unitless',
    desc='Flag to control enforcement of fuel_capacity constraint. '
    'If False (default) Aviary will add the excess fuel constraint and only converge if there is enough fuel capacity to complete the mission.'
    'If set True Aviary will ignore this constraint, and allow mission fuel > total_fuel_capacity. Use carefully!',
    default_value=False,
    types=bool,
)

add_meta_data(
    Aircraft.Fuel.NUM_TANKS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NTANK',  # ['&DEFINE.WTIN.NTANK', 'WTS.NTANK'],
    },
    units='unitless',
    desc='number of fuel tanks',
    types=int,
    option=True,
    default_value=7,
)

add_meta_data(
    Aircraft.Fuel.TOTAL_CAPACITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FMXTOT',  # ['&DEFINE.WTIN.FMXTOT', 'PLRNG.FMXTOT'],
    },
    units='lbm',
    desc='Total fuel capacity of the aircraft including wing, fuselage and '
    'auxiliary tanks. Used in generating payload-range diagram (Default = '
    'wing_capacity + fuselage_capacity + aux_capacity)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.TOTAL_VOLUME,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WEIGHT.ZFEQ',
    },
    units='galUS',  # need to check this
    desc='Total fuel volume',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER
    Aircraft.Fuel.UNUSABLE_FUEL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(29, 2)', '~WEIGHT.WUF', '~WTSTAT.WSP(29, 2)', '~INERT.WUF'],
        'FLOPS': None,
    },
    units='lbm',
    desc='unusable fuel mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(13)', 'FLOPS': None},
    default_value=0.0,
    units='unitless',
    desc='mass trend coefficient of trapped fuel factor',
)

add_meta_data(
    Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WUF', 'MISWT.WUF', 'MISWT.OUF'],
        'FLOPS': 'WTIN.WUF',
    },
    units='unitless',
    desc='scaler for Unusable fuel mass',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Fuel.VOLUME_MARGIN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOL_MRG', 'FLOPS': None},
    units='unitless',
    desc='Extra volume required in the wing fuel tank as a percentage of design mission fuel mass.'
    'Only used in GASP wing tank mass and fuel system mass sizing calculations.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_FUEL_CAPACITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FULWMX',  # ['&DEFINE.WTIN.FULWMX', 'WTS.FULWMX'],
    },
    units='lbm',
    desc='fuel capacity of the auxiliary tank',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_FUEL_FRACTION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKWF', 'FLOPS': None},
    units='unitless',
    desc='fraction of total theoretical wing volume used for wing fuel',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_REF_CAPACITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FUELRF',  # ['&DEFINE.WTIN.FUELRF', 'WPAB.FUELRF'],
    },
    units='lbm',  # TODO FLOPS says lbm, sfwate.f line 827
    desc='reference fuel volume',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_REF_CAPACITY_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FSWREF',  # ['&DEFINE.WTIN.FSWREF', 'WPAB.FSWREF'],
    },
    units='unitless',  # TODO FLOPS says unitless, sfwate.f line 828
    desc='reference wing area for fuel capacity',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_REF_CAPACITY_TERM_A,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FUSCLA',  # ['&DEFINE.WTIN.FUSCLA', 'WPAB.FUSCLA'],
    },
    units='unitless',
    desc='scaling factor A',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_REF_CAPACITY_TERM_B,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FUSCLB',  # ['&DEFINE.WTIN.FUSCLB', 'WPAB.FUSCLB'],
    },
    units='unitless',
    desc='scaling factor B',
    default_value=0.0,
)

# add_meta_data(
#     Aircraft.Fuel.WING_VOLUME,
#     meta_data=_MetaData,
#     historical_name={"GASP": None,
#                      "FLOPS": None,
#                     },
#     FLOPS_name=None,
#     GASP_name='INGASP.FVOLW',
#     units='ft**3',
#     desc='wing tank fuel volume',
# )

add_meta_data(
    Aircraft.Fuel.WING_VOLUME_DESIGN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOLREQ', 'FLOPS': None},
    units='ft**3',
    desc='wing tank fuel volume when carrying design fuel plus fuel margin',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOLW_GEOM', 'FLOPS': None},
    units='ft**3',
    desc='wing tank fuel volume based on geometry',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOLW_MAX', 'FLOPS': None},
    units='ft**3',
    desc='wing tank volume based on maximum wing fuel weight',
    default_value=0.0,
)


#
#   ______                          _         _       _
#  |  ____|                        (_)       | |     (_)
#  | |__     _   _   _ __   _ __    _   ___  | |__    _   _ __     __ _   ___
#  |  __|   | | | | | '__| | '_ \  | | / __| | '_ \  | | | '_ \   / _` | / __|
#  | |      | |_| | | |    | | | | | | \__ \ | | | | | | | | | | | (_| | \__ \
#  |_|       \__,_| |_|    |_| |_| |_| |___/ |_| |_| |_| |_| |_|  \__, | |___/
#                                                                  __/ |
#                                                                 |___/
# ============================================================================

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Furnishings.MASS_SCALER
    Aircraft.Furnishings.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CW(8)',
        # ['WTS.WSP(22, 2)', '~WEIGHT.WFURN', '~WTSTAT.WSP(22, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Total furnishings mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Furnishings.MASS_BASE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='For FLOPS based, base furnishings system mass without additional 1% empty mass',
    default_value=0.0,
)

# TODO the GASP use of this variable is misleading (optional coefficient for additional
#      furnishing mass, which is activated by Aircraft.Furnishings.USE_EMPERICAL_EQUATION). Create
#      new variable (Aircraft.Furnishings.MASS_COEFFICIENT?) for GASP side
add_meta_data(
    Aircraft.Furnishings.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WFURN', 'MISWT.WFURN', 'MISWT.OFURN'],
        'FLOPS': 'WTIN.WFURN',
    },
    units='unitless',
    desc='Furnishings system mass scaler. In GASP based, it is applicale if gross mass '
    '> 10000 lbs and number of passengers >= 50. Set it to 0.0 if not use.',
    default_value=1.0,
)

# Misnamed. This sets if Aircraft.Furnishings.MASS_SCALER is used as a coefficient for additional
# furnishings weight and the alternative (False) is to use the emperical equation. The variable toggle
# based on gross mass and num_pax is bad Aviary behavior and should occur in fortran_to_aviary instead
add_meta_data(
    Aircraft.Furnishings.USE_EMPIRICAL_EQUATION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='In GASP based, indicate whether use commonly used empirical furnishing weight equation. '
    'This applies only when gross mass > 10000 and number of passengers >= 50.',
    types=bool,
    option=True,
    default_value=True,
)

#  ______                        _
# |  ____|                      | |
# | |__     _   _   ___    ___  | |   __ _    __ _    ___
# |  __|   | | | | / __|  / _ \ | |  / _` |  / _` |  / _ \
# | |      | |_| | \__ \ |  __/ | | | (_| | | (_| | |  __/
# |_|       \__,_| |___/  \___| |_|  \__,_|  \__, |  \___|
#                                             __/ |
#                                            |___/
# ========================================================

add_meta_data(
    Aircraft.Fuselage.AFTBODY_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'WGT_AB', 'FLOPS': None},
    units='lbm',
    default_value=0.0,
    desc='aftbody mass',
)

add_meta_data(
    Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.UWT_AFT', 'FLOPS': None},
    units='lbm/ft**2',
    default_value=0.0,
    desc='aftbody structural areal unit weight',
)

add_meta_data(
    Aircraft.Fuselage.AISLE_WIDTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WAS', 'FLOPS': None},
    units='inch',
    desc='width of the aisles in the passenger cabin',
    option=True,
    default_value=24,
)

add_meta_data(
    Aircraft.Fuselage.AVG_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.WC', 'INGASP.SWF'],
        'FLOPS': None,
    },
    units='ft',
    desc='average fuselage diameter',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.CABIN_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.FUSEIN.ACABIN', 'WDEF.ACABIN'],
        'FLOPS': 'FUSEIN.ACABIN',
    },
    units='ft**2',
    desc='fixed area of passenger cabin for blended wing body transports',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[4]',
    },
    units='ft',
    desc='Reynolds characteristic length for the fuselage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.CROSS_SECTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['MISSA.SPI', '~CDCC.SPI'],
    },
    units='ft**2',
    desc='fuselage cross sectional area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.DELTA_DIAMETER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HCK', 'FLOPS': None},
    units='ft',
    desc='mean fuselage cabin diameter minus mean fuselage nose diameter',
    default_value=0.0,
)

# TODO this should be a design parameter? As it combines two physical categories?
add_meta_data(
    Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['MISSA.DB', '~CDCC.DB'],
    },
    units='unitless',
    desc='fuselage diameter to wing span ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCFFC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='fuselage aero calibration factor (including technology factor INGASP.FCFFT)',
)

add_meta_data(
    Aircraft.Fuselage.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[4]',
    },
    units='unitless',
    desc='fuselage fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELFE', 'FLOPS': None},
    units='ft**2',
    desc='increment to fuselage flat plate area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.FOREBODY_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'WGT_FB', 'FLOPS': None},
    units='lbm',
    default_value=0.0,
    desc='forebody mass',
)

add_meta_data(
    Aircraft.Fuselage.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKF', 'FLOPS': None},
    units='unitless',
    desc='fuselage form factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HGTqWID', 'FLOPS': 'WTIN.TCF'},
    units='unitless',
    types=float,
    default_value=1.0,
    desc='fuselage height-to-width ratio',
)

add_meta_data(
    Aircraft.Fuselage.HYDRAULIC_DIAMETER,
    meta_data=_MetaData,
    historical_name={'GASP': 'DHYDRAL', 'FLOPS': None},
    units='ft',
    types=float,
    default_value=0.0,
    desc='the geometric mean of cabin height and cabin width',
)

add_meta_data(
    Aircraft.Fuselage.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRLB',  # ['&DEFINE.AERIN.TRLB', 'XLAM.TRLB', ],
    },
    units='unitless',
    desc='define percent laminar flow for fuselage lower surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRUB',  # ['&DEFINE.AERIN.TRUB', 'XLAM.TRUB', ],
    },
    units='unitless',
    desc='define percent laminar flow for fuselage upper surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ELF',
        'FLOPS': 'WTIN.XL',
        #  [  # inputs
        #      '&DEFINE.WTIN.XL', 'WTS.XL',
        #      # outputs
        #      'EDETIN.BL', '~DEFAER.BL',
        #  ],
    },
    units='ft',
    desc='Define the Fuselage total length. If total_length is not input for a '
    'passenger transport, FLOPS will calculate the fuselage length, width and '
    'depth and the length of the passenger compartment.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.LENGTH_TO_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['MISSA.BODYLD', '~CDCC.BODYLD'],
    },
    units='unitless',
    desc='fuselage length to diameter ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.LIFT_COEFFICIENT_RATIO_BODY_TO_WING,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLBqCLW', 'FLOPS': None},
    units='unitless',
    types=float,
    default_value=0.0,
    desc='lift coefficient of body over lift coefficient of wing ratio',
)

add_meta_data(
    Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLALPH_B0', 'FLOPS': None},
    units='1/rad',
    default_value=0.0,
    desc='lift curve slope of fuselage at Mach 0',
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Fuselage.MASS_SCALER
    Aircraft.Fuselage.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(6, 2)', '~WEIGHT.WFUSE', '~WTSTAT.WSP(6, 2)', '~INERT.WFUSE', ],
        'FLOPS': None,
    },
    units='lbm',
    desc='Fuselage group mass. Contains basic structure and secondary structures such as '
    'enclosures, flooring, doors, ramps, panels, etc.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKB', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of fuselage',
    default_value=136,
)

add_meta_data(
    Aircraft.Fuselage.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRFU',  # ['&DEFINE.WTIN.FRFU', 'WTS.FRFU'],
    },
    units='unitless',
    desc='mass scaler of the fuselage structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Fuselage.MAX_HEIGHT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.DF',  # ['&DEFINE.WTIN.DF', 'WTS.DF'],
    },
    units='ft',
    desc='maximum fuselage height',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.MAX_WIDTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.WF',
        #  [  # inputs
        #      '&DEFINE.WTIN.WF', 'WTS.WF',
        #      # outputs
        #      'MIMOD.FWID',
        #  ],
    },
    units='ft',
    desc='maximum fuselage width',
    default_value=0.0,
)

# TODO are we keeping military cargo?
add_meta_data(
    Aircraft.Fuselage.MILITARY_CARGO_FLOOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.CARGF',  # ['&DEFINE.WTIN.CARGF', 'WTS.CARGF'],
    },
    units='unitless',
    desc='indicate whether or not there is a military cargo aircraft floor',
    option=True,
    types=bool,
    default_value=False,
)

add_meta_data(
    Aircraft.Fuselage.NOSE_FINENESS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELODN', 'FLOPS': None},
    units='unitless',
    desc='length to diameter ratio of nose cone',
    default_value=1,
)

add_meta_data(
    Aircraft.Fuselage.NUM_AISLES,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.AS', 'FLOPS': None},
    units='unitless',
    desc='number of aisles in the passenger cabin',
    types=int,
    option=True,
    default_value=1,
)

add_meta_data(
    Aircraft.Fuselage.NUM_FUSELAGES,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.NFUSE', 'EDETIN.NFUSE', '~WWGHT.NFUSE'],
        'FLOPS': 'WTIN.NFUSE',
    },
    units='unitless',
    desc='number of fuselages',
    types=int,
    option=True,
    default_value=1,
)

add_meta_data(
    Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.XLP',  # ['&DEFINE.WTIN.XLP', 'WTS.XLP'],
    },
    units='ft',
    desc='length of passenger compartment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELPC', 'FLOPS': None},
    units='ft',
    desc='length of the pilot compartment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PLANFORM_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'SPF_BODY',
        'FLOPS': None,  # '~WEIGHT.FPAREA',
    },
    units='ft**2',
    desc='fuselage planform area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELP', 'FLOPS': None},
    units='psi',
    desc='fuselage pressure differential during cruise',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WPRFUS', 'FLOPS': None},
    units='ft',
    default_value=0.0,
    desc='additional pressurized fuselage width for cargo bay',
)

add_meta_data(
    Aircraft.Fuselage.REF_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': ['EDETIN.XD'],
    },
    units='ft',
    desc='A coarse average diameter calculated using the mean of max width and depth.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.SEAT_WIDTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WS', 'FLOPS': None},
    units='inch',
    desc='width of the economy class seats',
    option=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'WTIN.TCSOB', 'LEAPS1': None},
    units='unitless',
    desc='fuselage thickness/chord ratio at side of body',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.SIMPLE_LAYOUT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    types=bool,
    units='unitless',
    desc='carry out simple or detailed layout of fuselage (for FLOPS based geometry).',
    option=True,
    default_value=True,
)

add_meta_data(
    Aircraft.Fuselage.TAIL_FINENESS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELODT', 'FLOPS': None},
    units='unitless',
    desc='length to diameter ratio of tail cone',
    default_value=1.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Fuselage.WETTED_AREA_SCALER
    Aircraft.Fuselage.WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SF',
        'FLOPS': None,  # ['ACTWET.SWTFU', 'MISSA.SWET[4]'],
    },
    units='ft**2',
    desc='fuselage wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SAFTqS', 'FLOPS': None},
    units='unitless',
    types=float,
    default_value=0.0,
    desc='aftbody wetted area to total body wetted area',
)

add_meta_data(
    Aircraft.Fuselage.WETTED_AREA_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SF_FAC',
        'FLOPS': 'AERIN.SWETF',  # ['&DEFINE.AERIN.SWETF', 'AWETO.SWETF', ],
    },
    units='unitless',
    desc='fuselage wetted area scaler',
    default_value=1.0,
)

#  _    _                  _                         _             _   _______           _   _
# | |  | |                (_)                       | |           | | |__   __|         (_) | |
# | |__| |   ___    _ __   _   ____   ___    _ __   | |_    __ _  | |    | |      __ _   _  | |
# |  __  |  / _ \  | '__| | | |_  /  / _ \  | '_ \  | __|  / _` | | |    | |     / _` | | | | |
# | |  | | | (_) | | |    | |  / /  | (_) | | | | | | |_  | (_| | | |    | |    | (_| | | | | |
# |_|  |_|  \___/  |_|    |_| /___|  \___/  |_| |_|  \__|  \__,_| |_|    |_|     \__,_| |_| |_|
# =============================================================================================

add_meta_data(
    Aircraft.HorizontalTail.AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SHT',  # not an input in GASP
        'FLOPS': 'WTIN.SHT',  # ['&DEFINE.WTIN.SHT', 'EDETIN.SHT'],
    },
    units='ft**2',
    desc='horizontal tail theoretical area; overridden by vol_coeff, if '
    'vol_coeff > 0.0',  # TODO: this appears to never be calculated in Aviary, need to show users the overriding capability of Aviary
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.ASPECT_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ARHT',
        'FLOPS': 'WTIN.ARHT',  # ['&DEFINE.WTIN.ARHT', 'EDETIN.ARHT'],
    },
    units='unitless',
    desc='horizontal tail theoretical aspect ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.AVERAGE_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CBARHT', 'FLOPS': None},
    units='ft',
    desc='mean aerodynamic chord of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[2]',
    },
    units='ft',
    desc='Reynolds characteristic length for the horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCFHTC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='horizontal tail aero calibration factor (including technology factor INGASP.FCFHTT)',
)

add_meta_data(
    Aircraft.HorizontalTail.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[2]',
    },
    units='unitless',
    desc='horizontal tail fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKHT', 'FLOPS': None},
    units='unitless',
    desc='horizontal tail form factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRLH',  # ['&DEFINE.AERIN.TRLH', 'XLAM.TRLH', ],
    },
    units='unitless',
    desc='define percent laminar flow for horizontal tail lower surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRUH',  # ['&DEFINE.AERIN.TRUH', 'XLAM.TRUH', ],
    },
    units='unitless',
    desc='define percent laminar flow for horizontal tail upper surface',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.HorizontalTail.MASS_SCALER
    Aircraft.HorizontalTail.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(2, 2)', '~WEIGHT.WHT', '~WTSTAT.WSP(2, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKY', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRHT',  # ['&DEFINE.WTIN.FRHT', 'WTS.FRHT'],
    },
    units='unitless',
    desc='mass scaler of the horizontal tail structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MOMENT_ARM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELTH', 'FLOPS': None},
    units='ft',
    desc='moment arm of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MOMENT_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.COELTH', 'FLOPS': None},
    units='unitless',
    desc='Ratio of wing chord to horizontal tail moment arm',
)

add_meta_data(
    Aircraft.HorizontalTail.NUM_TAILS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='number of horizontal tails',
    types=int,
    option=True,
    default_value=1,
)

add_meta_data(
    Aircraft.HorizontalTail.ROOT_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRCLHT', 'FLOPS': None},
    units='ft',
    desc='horizontal tail root chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.SPAN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BHT', 'FLOPS': None},
    units='ft',
    desc='span of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.SWEEP,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.DWPQCH',
        'FLOPS': 'WTIN.SWPHT',  # , 'WTS.SWPHT'],
    },
    units='deg',
    desc='quarter-chord sweep of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.TAPER_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SLMH',
        'FLOPS': 'WTIN.TRHT',  # , 'EDETIN.TRHT'],
    },
    units='unitless',
    desc='horizontal tail theoretical taper ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.TCHT',
        'FLOPS': 'WTIN.TCHT',  # , 'EDETIN.TCHT'],
    },
    units='unitless',
    desc='horizontal tail thickness-chord ratio',
    default_value=0.0,
)

# TODO preprocessing for this variable on FLOPS side
add_meta_data(
    Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SAH',
        'FLOPS': 'WTIN.HHT',  # ['&DEFINE.WTIN.HHT', 'EDETIN.HHT'],
    },
    units='unitless',
    desc='Define the decimal fraction of vertical tail span where horizontal '
    'tail is mounted. Defaults: 0.0 == for body mounted (default for '
    'transport with all engines on wing); 1.0 == for T tail '
    '(default for transport with multiple engines on fuselage)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VBARHX', 'FLOPS': None},
    units='unitless',
    desc='tail volume coefficicient of horizontal tail',
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.HorizontalTail.WETTED_AREA_SCALER
    Aircraft.HorizontalTail.WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['ACTWET.SWTHT', 'MISSA.SWET[2]'],
    },
    units='ft**2',
    desc='horizontal tail wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.WETTED_AREA_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.SWETH',  # ['&DEFINE.AERIN.SWETH', 'AWETO.SWETH', ],
    },
    units='unitless',
    desc='horizontal tail wetted area scaler',
    default_value=1.0,
)

#  _    _               _                          _   _
# | |  | |             | |                        | | (_)
# | |__| |  _   _    __| |  _ __    __ _   _   _  | |  _    ___   ___
# |  __  | | | | |  / _` | | '__|  / _` | | | | | | | | |  / __| / __|
# | |  | | | |_| | | (_| | | |    | (_| | | |_| | | | | | | (__  \__ \
# |_|  |_|  \__, |  \__,_| |_|     \__,_|  \__,_| |_| |_|  \___| |___/
#            __/ |
#           |___/
# ====================================================================

add_meta_data(
    Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(3)', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of hydraulics for flight control system',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(4)', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of hydraulics for landing gear',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Hydraulics.MASS_SCALER
    Aircraft.Hydraulics.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(19, 2)', '~WEIGHT.WHYD', '~WTSTAT.WSP(19, 2)', '~INERT.WHYD'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of hydraulic system',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Hydraulics.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WHYD', 'MISWT.WHYD', 'MISWT.OHYD'],
        'FLOPS': 'WTIN.WHYD',
    },
    units='unitless',
    desc='mass scaler of the hydraulic system',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Hydraulics.SYSTEM_PRESSURE,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.HYDPR',  # ['&DEFINE.WTIN.HYDPR', 'WTS.HYDPR'],
    },
    units='psi',
    desc='hydraulic system pressure',
    default_value=0.0,
)

#
#  _____                 _                                               _
# |_   _|               | |                                             | |
#   | |    _ __    ___  | |_   _ __   _   _   _ __ ___     ___   _ __   | |_   ___
#   | |   | '_ \  / __| | __| | '__| | | | | | '_ ` _ \   / _ \ | '_ \  | __| / __|
#  _| |_  | | | | \__ \ | |_  | |    | |_| | | | | | | | |  __/ | | | | | |_  \__ \
# |_____| |_| |_| |___/  \__| |_|     \__,_| |_| |_| |_|  \___| |_| |_|  \__| |___/
# ================================================================================================

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Instruments.MASS_SCALER
    Aircraft.Instruments.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(18, 2)', '~WEIGHT.WIN', '~WTSTAT.WSP(18, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='instrument group mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Instruments.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(2)', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of instruments',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Instruments.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WIN', 'MISWT.WIN', 'MISWT.OIN'],
        'FLOPS': 'WTIN.WIN',
    },
    units='unitless',
    desc='mass scaler of the instrument group',
    default_value=1.0,
)

#  _                            _   _                    _____
# | |                          | | (_)                  / ____|
# | |        __ _   _ __     __| |  _   _ __     __ _  | |  __    ___    __ _   _ __
# | |       / _` | | '_ \   / _` | | | | '_ \   / _` | | | |_ |  / _ \  / _` | | '__|
# | |____  | (_| | | | | | | (_| | | | | | | | | (_| | | |__| | |  __/ | (_| | | |
# |______|  \__,_| |_| |_|  \__,_| |_| |_| |_|  \__, |  \_____|  \___|  \__,_| |_|
#                                                __/ |
#                                               |___/
# ===================================================================================
add_meta_data(
    Aircraft.LandingGear.DRAG_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        # ['&DEFTOL.TOLIN.CDGEAR', '~DEFTOL.CDGEAR', 'ROTDAT.CDGEAR'],
        'FLOPS': 'TOLIN.CDGEAR',
        'GASP': None,
    },
    option=True,
    default_value=0.0,
    units='unitless',
    desc='landing gear drag coefficient',
)

add_meta_data(
    Aircraft.LandingGear.FIXED_GEAR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.IGEAR', 'FLOPS': None},
    option=True,
    default_value=True,
    types=bool,
    units='unitless',
    desc='Type of landing gear. In GASP, 0 is retractable and 1 is fixed. Here, '
    'false is retractable and true is fixed.',
)

add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.YMG', 'FLOPS': None},
    units='unitless',
    desc='span fraction of main gear on wing (0=on fuselage, 1=at tip)',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER
    Aircraft.LandingGear.MAIN_GEAR_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'INI.WLGM',
    },
    units='lbm',
    desc='mass of main landing gear (WMG in GASP)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKMG', 'FLOPS': None},
    units='unitless',
    desc='fraction of total landing gear mass that is main gear mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRLGM',  # ['&DEFINE.WTIN.FRLGM', 'WTS.FRLGM'],
    },
    units='unitless',
    desc='mass scaler of the main landing gear structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.XMLG',  # ['&DEFINE.WTIN.XMLG', 'WTS.XMLG'],
    },
    units='inch',
    desc='length of extended main landing gear oleo',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKLG', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of landing gear',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER
    Aircraft.LandingGear.NOSE_GEAR_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WEIGHT.WLGN',
    },
    units='lbm',
    desc='mass of nose landing gear',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRLGN',  # ['&DEFINE.WTIN.FRLGN', 'WTS.FRLGN'],
    },
    units='unitless',
    desc='mass scaler of the nose landing gear structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.XNLG',  # ['&DEFINE.WTIN.XNLG', 'WTS.XNLG'],
    },
    units='inch',
    desc='length of extended nose landing gear oleo',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKTL', 'FLOPS': None},
    units='unitless',
    desc='factor on tail mass for arresting hook',
    default_value=1,
)

add_meta_data(
    Aircraft.LandingGear.TOTAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WLG', 'FLOPS': None},
    units='lbm',
    desc='total mass of landing gear',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.TOTAL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK12', 'FLOPS': None},
    units='unitless',
    desc='technology factor on landing gear mass',
    default_value=1.0,
)

#  _   _                         _   _
# | \ | |                       | | | |
# |  \| |   __ _    ___    ___  | | | |   ___
# | . ` |  / _` |  / __|  / _ \ | | | |  / _ \
# | |\  | | (_| | | (__  |  __/ | | | | |  __/
# |_| \_|  \__,_|  \___|  \___| |_| |_|  \___|
# ============================================

add_meta_data(
    Aircraft.Nacelle.AVG_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.DBARN',
        'FLOPS': 'WTIN.DNAC',  # ['&DEFINE.WTIN.DNAC', 'EDETIN.DNAC'],
    },
    units='ft',
    desc='Average diameter of engine nacelles for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.AVG_LENGTH,
    meta_data=_MetaData,
    # NOTE this is not specified as an average in GASP, but calculations make
    # it appear to be one
    historical_name={
        'GASP': 'INGASP.ELN',
        'FLOPS': 'WTIN.XNAC',  # ['&DEFINE.WTIN.XNAC', 'EDETIN.XNAC'],
    },
    units='ft',
    desc='Average length of nacelles for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[5]',
    },
    units='ft',
    desc='Reynolds characteristic length for nacelle for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.CLEARANCE_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLEARqDN', 'FLOPS': None},
    units='unitless',
    desc='the minimum number of nacelle diameters above the ground that the bottom of the nacelle must be',
    default_value=0.0,  # should be at least 0.2
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.CORE_DIAMETER_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DNQDE', 'FLOPS': None},
    units='unitless',
    desc='ratio of nacelle diameter to engine core diameter',
    default_value=1.25,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCFNC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='nacelle aero calibration factor (including technology factor INGASP.FCFNT)',
)

add_meta_data(
    Aircraft.Nacelle.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.XLQDE',
        'FLOPS': None,  # 'MISSA.FR[5]',
    },
    units='unitless',
    desc='nacelle fineness ratio',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKN', 'FLOPS': None},
    units='unitless',
    desc='nacelle form factor',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRLN',  # ['&DEFINE.AERIN.TRLN', 'XLAM.TRLN', ],
    },
    units='unitless',
    desc='define percent laminar flow for nacelle lower surface for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRUN',  # ['&DEFINE.AERIN.TRUN', 'XLAM.TRUN', ],
    },
    units='unitless',
    desc='define percent laminar flow for nacelle upper surface for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Nacelle.MASS_SCALER
    Aircraft.Nacelle.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(8, 2)', '~WEIGHT.WNAC', '~WTSTAT.WSP(8, 2)', '~INERT.WNAC'],
        'FLOPS': None,
    },
    units='lbm',
    desc='estimated mass of the nacelles for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRNA',  # ['&DEFINE.WTIN.FRNA', 'WTS.FRNA'],
    },
    units='unitless',
    desc='mass scaler of the nacelle structure for each engine model',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.MASS_SPECIFIC,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.UWNAC', 'FLOPS': None},
    units='lbm/ft**2',
    desc='nacelle mass/nacelle surface area; lbm per sq ft.',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HEBQDN', 'FLOPS': None},
    units='unitless',
    desc='percentage of nacelle diameter buried in fuselage over nacelle diameter',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.PYLON_DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FPYLND', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='pylon aero calibration factor',
)

add_meta_data(
    Aircraft.Nacelle.SURFACE_AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SN', 'FLOPS': None},  # SN is wetted area
    units='ft**2',
    desc='surface area of the outside of one entire nacelle, not just the wetted area',
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.TOTAL_WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'ACTWET.SWTNA',
    },
    units='ft**2',
    desc='total nacelles wetted area',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.SWET[5]',
    },
    units='ft**2',
    desc='wetted area of a single nacelle for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.WETTED_AREA_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.SWETN',  # ['&DEFINE.AERIN.SWETN', 'AWETO.SWETN', ],
    },
    units='unitless',
    desc='nacelle wetted area scaler for each engine model',
    default_value=1.0,
    multivalue=True,
)

#   ____                                             _____                 _
#  / __ \                                           / ____|               | |
# | |  | | __  __  _   _    __ _    ___   _ __     | (___    _   _   ___  | |_    ___   _ __ ___
# | |  | | \ \/ / | | | |  / _` |  / _ \ | '_ \     \___ \  | | | | / __| | __|  / _ \ | '_ ` _ \
# | |__| |  >  <  | |_| | | (_| | |  __/ | | | |    ____) | | |_| | \__ \ | |_  |  __/ | | | | | |
#  \____/  /_/\_\  \__, |  \__, |  \___| |_| |_|   |_____/   \__, | |___/  \__|  \___| |_| |_| |_|
#                   __/ |   __/ |                             __/ |
#                  |___/   |___/                             |___/
# ================================================================================================

add_meta_data(
    Aircraft.OxygenSystem.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Mass of passenger oxygen system',
    default_value=0.0,
)

add_meta_data(
    Aircraft.OxygenSystem.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Mass Scaler for the Passenger Oxygen System',
    default_value=0.0,
)

#  _____            _           _
# |  __ \          (_)         | |
# | |__) |   __ _   _   _ __   | |_
# |  ___/   / _` | | | | '_ \  | __|
# | |      | (_| | | | | | | | | |_
# |_|       \__,_| |_| |_| |_|  \__|
# ==================================

add_meta_data(
    Aircraft.Paint.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'DARM.WTPNT',
    },
    units='lbm',
    desc='mass of paint for all wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Paint.MASS_PER_UNIT_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.WPAINT',  # ['&DEFINE.WTIN.WPAINT', 'DARM.WPAINT'],
    },
    units='lbm/ft**2',
    desc='mass of paint per unit area for all wetted area',
    default_value=0.0,
)

#  _____                                   _         _
# |  __ \                                 | |       (_)
# | |__) |  _ __    ___    _ __    _   _  | |  ___   _    ___    _ __
# |  ___/  | '__|  / _ \  | '_ \  | | | | | | / __| | |  / _ \  | '_ \
# | |      | |    | (_) | | |_) | | |_| | | | \__ \ | | | (_) | | | | |
# |_|      |_|     \___/  | .__/   \__,_| |_| |___/ |_|  \___/  |_| |_|
#                         | |
#                         |_|
# =====================================================================
# NOTE variables under propulsion are aircraft-level values
add_meta_data(
    Aircraft.Propulsion.ENERGY_SYSTEM_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
    },
    units='lbm',
    desc='Energy system mass. Contains mass for energy storage and transmission, including the fuel '
    'system, battery, and electric powertrain.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WOIL', 'MISWT.WOIL', 'MISWT.OOIL'],
        'FLOPS': 'WTIN.WOIL',
    },
    units='unitless',
    desc='Scaler for engine oil mass',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Propulsion.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(15, 2)', '~WEIGHT.WPRO', '~WTSTAT.WSP(15, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Propulsion group mass. Total mass of all engines on the aircraft, as well as energy system '
    'mass.',
    default_value=0.0,
)

# TODO clash with per-engine scaling, need to resolve w/ heterogeneous engine
# TODO in GASP this applies to ADDITIONAL_MASS (installation weight), confusing because that also
#      uses ADDITIONAL_MASS_FRACTION - also applies globally to all engines there, which is wrong
add_meta_data(
    Aircraft.Propulsion.MISC_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WPMSC', 'MISWT.WPMSC', 'MISWT.OPMSC'],
        'FLOPS': 'WTIN.WPMSC',
    },
    units='unitless',
    desc='scaler applied to miscellaneous engine mass (in FLOPS, sum of engine control, starter, '
    'and additional mass. In GASP, applied to ADDITIONAL_MASS, which is engine installation mass)',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total estimated mass of the engine controls for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_ENGINE_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WEP', 'FLOPS': None},
    units='lbm',
    desc='total mass of all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER
    Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(30, 2)', '~WEIGHT.WOIL', '~WTSTAT.WSP(30, 2)', '~INERT.WOIL'],
        'FLOPS': None,
    },
    units='lbm',
    desc='engine oil mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total engine pod mass for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Propulsion.MISC_WEIGHT_SCALER
    Aircraft.Propulsion.TOTAL_MISC_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='sum of engine control, starter, and additional mass for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_NUM_ENGINES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='total number of engines for the aircraft (fuselage, wing, or otherwise)',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='total number of fuselage-mounted engines for the aircraft',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='total number of wing-mounted engines for the aircraft',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_REFERENCE_SLS_THRUST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='total maximum thrust of all unscalsed engines on aircraft, sea-level static',
    option=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='total maximum thrust of all scaled engines on aircraft, sea-level static',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_STARTER_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total mass of starters for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER
    Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='total mass of thrust reversers for all engines on aircraft',
    default_value=0.0,
)

#   _____   _                    _
#  / ____| | |                  | |
# | (___   | |_   _ __   _   _  | |_
#  \___ \  | __| | '__| | | | | | __|
#  ____) | | |_  | |    | |_| | | |_
# |_____/   \__| |_|     \__,_|  \__|
# ===================================

add_meta_data(
    Aircraft.Strut.AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STRTWS', 'FLOPS': None},
    units='ft**2',
    desc='strut area',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.AREA_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SSTQSW', 'FLOPS': None},
    units='unitless',
    desc='ratio of strut area to wing area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.ATTACHMENT_LOCATION,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.STRUT', 'INGASP.STRUTX', 'INGASP.XSTRUT'],
        'FLOPS': None,
    },
    units='ft',
    desc='attachment location of strut the full attachment-to-attachment span',
    default_value=0.0,
)

# related to Aircraft.Strut.ATTACHMENT_LOCATION
add_meta_data(
    Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='attachment location of strut as fraction of the half-span',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STRTCHD', 'FLOPS': None},
    units='ft',
    desc='chord of the strut',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=True,
    types=bool,
    desc='if true the location of the strut is given dimensionally, otherwise '
    'it is given non-dimensionally. In GASP this depended on STRUT',
)

add_meta_data(
    Aircraft.Strut.DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCFSTRC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='strut aero calibration factor (including technology factor INGASP.FCFSTRT)',
)

add_meta_data(
    Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKSTRT', 'FLOPS': None},
    units='unitless',
    desc='strut/fuselage interference factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.LENGTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STRTLNG', 'FLOPS': None},
    units='ft',
    desc='length of the strut',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WSTRUT', 'FLOPS': None},
    units='lbm',
    desc='mass of the strut',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKSTRUT', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of the strut',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TCSTRT', 'FLOPS': None},
    units='unitless',
    desc='thickness to chord ratio of the strut',
    default_value=0,
)

#  ____
# |  _ \
# | |_) |   ___     ___    _ __ ___
# |  _ <   / _ \   / _ \  | '_ ` _ \
# | |_) | | (_) | | (_) | | | | | | |
# |____/   \___/   \___/  |_| |_| |_|
# ===================================

add_meta_data(
    Aircraft.TailBoom.LENGTH,  # tail boom support is not included.
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELFFC', 'FLOPS': None},
    units='ft',
    # tail boom support is not implemented yet.
    desc='cabin length for the tail boom fuselage',
    default_value=0.0,
)


# __      __                _     _                  _   _______           _   _
# \ \    / /               | |   (_)                | | |__   __|         (_) | |
#  \ \  / /    ___   _ __  | |_   _    ___    __ _  | |    | |      __ _   _  | |
#   \ \/ /    / _ \ | '__| | __| | |  / __|  / _` | | |    | |     / _` | | | | |
#    \  /    |  __/ | |    | |_  | | | (__  | (_| | | |    | |    | (_| | | | | |
#     \/      \___| |_|     \__| |_|  \___|  \__,_| |_|    |_|     \__,_| |_| |_|
# ===============================================================================

add_meta_data(
    Aircraft.VerticalTail.AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SVT',  # not an input in GASP
        'FLOPS': 'WTIN.SVT',  # ['&DEFINE.WTIN.SVT', 'EDETIN.SVT'],
    },
    units='ft**2',
    desc='vertical tail theoretical area (per tail); overridden by vol_coeff if vol_coeff > 0.0',
    # this appears to never be calculated in Aviary, need to make user aware
    # of Aviary overriding support
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.ASPECT_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ARVT',
        'FLOPS': 'WTIN.ARVT',  # ['&DEFINE.WTIN.ARVT', 'EDETIN.ARVT'],
    },
    units='unitless',
    desc='vertical tail theoretical aspect ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.AVERAGE_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CBARVT', 'FLOPS': None},
    units='ft',
    desc='mean aerodynamic chord of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[3]',
    },
    units='ft',
    desc='Reynolds characteristic length for the vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCFVTC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='vertical tail aero calibration factor (including technology factor INGASP.FCFVTT)',
)

add_meta_data(
    Aircraft.VerticalTail.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[3]',
    },
    units='unitless',
    desc='vertical tail fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKVT', 'FLOPS': None},
    units='unitless',
    desc='vertical tail form factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRLV',  # ['&DEFINE.AERIN.TRLV', 'XLAM.TRLV', ],
    },
    units='unitless',
    desc='define percent laminar flow for vertical tail lower surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRUV',  # ['&DEFINE.AERIN.TRUV', 'XLAM.TRUV', ],
    },
    units='unitless',
    desc='define percent laminar flow for vertical tail upper surface',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.VerticalTail.MASS_SCALER
    Aircraft.VerticalTail.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(3, 2)', '~WEIGHT.WVT', '~WTSTAT.WSP(3, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKZ', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of the vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRVT',  # ['&DEFINE.WTIN.FRVT', 'WTS.FRVT'],
    },
    units='unitless',
    desc='mass scaler of the vertical tail structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.VerticalTail.MOMENT_ARM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELTV', 'FLOPS': None},
    units='ft',
    desc='moment arm of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.MOMENT_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BOELTV', 'FLOPS': None},
    units='unitless',
    desc='ratio of wing span to vertical tail moment arm',
)

add_meta_data(
    Aircraft.VerticalTail.NUM_TAILS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NVERT',  # ['&DEFINE.WTIN.NVERT', 'EDETIN.NVERT'],
    },
    units='unitless',
    desc='number of vertical tails',
    types=int,
    option=True,
    default_value=1,
)

add_meta_data(
    Aircraft.VerticalTail.ROOT_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRCLVT', 'FLOPS': None},
    units='ft',
    desc='root chord of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.SPAN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BVT', 'FLOPS': None},
    units='ft',
    desc='span of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.SWEEP,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.DWPQCV',
        'FLOPS': 'WTIN.SWPVT',  # ['&DEFINE.WTIN.SWPVT', 'WTS.SWPVT'],
    },
    units='deg',
    desc='quarter-chord sweep of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.TAPER_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SLMV',
        'FLOPS': 'WTIN.TRVT',  # ['&DEFINE.WTIN.TRVT', 'EDETIN.TRVT'],
    },
    units='unitless',
    desc='vertical tail theoretical taper ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.TCVT',
        'FLOPS': 'WTIN.TCVT',  # ['&DEFINE.WTIN.TCVT', 'EDETIN.TCVT', ],
    },
    units='unitless',
    desc='vertical tail thickness-chord ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.VOLUME_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VBARVX', 'FLOPS': None},
    units='unitless',
    desc='tail volume coefficient of the vertical tail',
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.VerticalTail.WETTED_AREA_SCALER
    Aircraft.VerticalTail.WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['ACTWET.SWTVT', 'MISSA.SWET[3]', ],
    },
    units='ft**2',
    desc='vertical tails wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.WETTED_AREA_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.SWETV',  # ['&DEFINE.AERIN.SWETV', 'AWETO.SWETV', ],
    },
    units='unitless',
    desc='vertical tail wetted area scaler',
    default_value=1.0,
)

# __          __  _
# \ \        / / (_)
#  \ \  /\  / /   _   _ __     __ _
#   \ \/  \/ /   | | | '_ \   / _` |
#    \  /\  /    | | | | | | | (_| |
#     \/  \/     |_| |_| |_|  \__, |
#                              __/ |
#                             |___/
# ==================================

add_meta_data(
    Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.FAERT', 'WTS.FAERT', '~WWGHT.FAERT', '~BNDMAT.FAERT'],
        'FLOPS': 'WTIN.FAERT',
    },
    units='unitless',
    desc='Define the decimal fraction of amount of aeroelastic tailoring used '
    'in design of wing where: 0.0 == no aeroelastic tailoring; '
    '1.0 == maximum aeroelastic tailoring.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.AIRFOIL_TECHNOLOGY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.AITEK',
        #  [  # inputs
        #      '&DEFINE.AERIN.AITEK', 'EDETIN.AITEK',
        #      # outputs
        #      'MISSA.AITEK', 'MISSA.AITEKX',
        #  ],
    },
    units='unitless',
    desc='Airfoil technology parameter. Limiting values are: 1.0 represents '
    'conventional technology wing (Default); 2.0 represents advanced '
    'technology wing.',
    default_value=1.0,
    option=True,
)

add_meta_data(
    Aircraft.Wing.AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SW',
        'FLOPS': 'CONFIN.SW',
        #  [  # inputs
        #      '&DEFINE.CONFIN.SW', 'PARVAR.DVD(1,4)',
        #      # outputs
        #      'CONFIG.SW', 'CONFIG.DVA(4)', '~FLOPS.DVA(4)', '~ANALYS.DVA(4)',
        #      '~TOFF.SW', '~LNDING.SW', '~PROFIL.SW', '~INMDAT.SW', '~WWGHT.SW',
        #      # other
        #      'MISSA.SREF', '~CDCC.SREF',
        #  ],
    },
    units='ft**2',
    desc='reference wing area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.ASPECT_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.AR',
        'FLOPS': 'CONFIN.AR',
        #  [  # inputs
        #      '&DEFINE.CONFIN.AR', 'PARVAR.DVD(1, 2)', '~BUFFET.AR', '~CDPP.AR',
        #      '~DPREP.ARX',
        #      # outputs
        #      'CONFIG.AR', 'CONFIG.DVA(2)', '~FLOPS.DVA(2)', '~ANALYS.DVA(2)',
        #      '~TOFF.ARN', '~LNDING.ARN', '~PROFIL.ARN', '~WWGHT.ARN', '~INERT.ARN',
        #      # other
        #      'MISSA.AR', 'MISSA.ARX', '~CDCC.AR', '~CLDESN.AR', '~MDESN.AR',
        #  ],
    },
    units='unitless',
    desc='ratio of the wing span to its mean chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.ASPECT_RATIO_REFERENCE,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.ARREF',  # ['&DEFINE.WTIN.ARREF'],
    },
    units='unitless',
    desc='Reference aspect ratio, used for detailed wing mass estimation.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.AVERAGE_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CBARW', 'FLOPS': None},
    units='ft',
    desc='mean aerodynamic chord of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.BENDING_MATERIAL_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['~WWGHT.BT', '~BNDMAT.W'],
    },
    units='unitless',
    desc='Wing bending material factor with sweep adjustment. Used to compute '
    'Aircraft.Wing.BENDING_MATERIAL_MASS',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER
    Aircraft.Wing.BENDING_MATERIAL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WWGHT.W1',
    },
    units='lbm',
    desc='wing mass breakdown term 1',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRWI1',  # ['&DEFINE.WTIN.FRWI1', 'WIOR3.FRWI1'],
    },
    units='unitless',
    desc='mass scaler of the bending wing mass term',
    default_value=1.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Wing.BWB_AFTBODY_MASS_SCALER
    Aircraft.Wing.BWB_AFTBODY_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WWGHT.W4',
    },
    units='lbm',
    desc='wing mass breakdown term 4',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.BWB_AFTBODY_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRWI4',  # ['&DEFINE.WTIN.FRWI4', 'WIOR3.FRWI4'],
    },
    units='unitless',
    desc='mass scaler of the blended-wing-body aft-body wing mass term',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.CENTER_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRCLW', 'FLOPS': None},
    units='ft',
    desc='wing chord at fuselage centerline, usually called root chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.CENTER_DISTANCE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XWQLF', 'FLOPS': None},
    units='unitless',
    desc='distance (percent fuselage length) from nose to the wing aerodynamic center',
)

add_meta_data(
    Aircraft.Wing.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[1]',
    },
    units='ft',
    desc='Reynolds characteristic length for the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.CHOOSE_FOLD_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    default_value=True,
    types=bool,
    option=True,
    desc='if true, fold location is based on your chosen value, otherwise it is '
    'based on strut location. In GASP this depended on STRUT or YWFOLD',
)

add_meta_data(
    Aircraft.Wing.CHORD_PER_SEMISPAN_DISTRIBUTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.CHD',  # ['&DEFINE.WTIN.CHD', 'WDEF.CHD'],
    },
    units='unitless',
    desc='chord lengths as fractions of semispan at station locations; '
    'overwrites station_chord_lengths',
    types=float,
    default_value=[0.0],
    multivalue=True,
)

add_meta_data(
    Aircraft.Wing.COMPOSITE_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.FCOMP', 'WTS.FCOMP', '~WWGHT.FCOMP'],
        'FLOPS': 'WTIN.FCOMP',
    },
    units='unitless',
    desc='Define the decimal fraction of amount of composites used in wing '
    'structure where: 0.0 == no composites; 1.0 == maximum use of composites, '
    'approximately equivalent bending_mat_weight=.6, '
    'struct_weights=.83, misc_weight=.7 '
    '(not necessarily all composite).',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.CONTROL_SURFACE_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WEIGHT.SFLAP',  # TODO ~WWGHT.SFLAP: similar, but separate calculation
    },
    units='ft**2',
    desc='area of wing control surfaces',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.FLAPR', 'WTS.FLAPR', '~WWGHT.FLAPR'],
        'FLOPS': 'WTIN.FLAPR',
    },
    units='unitless',
    desc='Defines the ratio of total moveable wing control surface areas '
    '(flaps, elevators, spoilers, etc.) to reference wing area.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.DETAILED_WING,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Flag that sets if FLOPS mass should use the detailed wing model',
    option=True,
    types=bool,
    default_value=False,
)

add_meta_data(
    Aircraft.Wing.DIHEDRAL,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.DIH',  # ['&DEFINE.WTIN.DIH', 'WTS.DIH', ],
    },
    units='deg',
    desc='wing dihedral (positive) or anhedral (negative) angle',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.DRAG_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FCFWC', 'FLOPS': None},
    units='unitless',
    default_value=1.0,
    desc='wing aero calibration factor (including technology factor INGASP.FCFWT)',
)

add_meta_data(
    Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WWGHT.CAYE',
    },
    units='unitless',
    desc='Engine inertia relief factor for wingspan inboard of engine locations. Used '
    'to compute Aircraft.Wing.BENDING_MATERIAL_MASS',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.EXPOSED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'SW_EXP',
        'FLOPS': None,
    },
    units='ft**2',
    desc='exposed wing area, i.e. wing area outside the fuselage, True for both Tube&Wing and HWB',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[1]',
    },
    units='unitless',
    desc='wing fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FLAP_CHORD_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CFOC', 'FLOPS': None},
    units='unitless',
    desc='ratio of flap chord to wing chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FLAP_DEFLECTION_LANDING,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DFLPLD', 'FLOPS': None},
    units='deg',
    desc='Deflection of flaps for landing',
)

add_meta_data(
    Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DFLPTO', 'FLOPS': None},
    units='deg',
    desc='Deflection of flaps for takeoff',
)

add_meta_data(
    Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCDOTE', 'FLOPS': None},
    units='unitless',
    desc='drag coefficient increment due to optimally deflected trailing edge flaps (default depends on flap type)',
)

add_meta_data(
    Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCLMTE', 'FLOPS': None},
    units='unitless',
    desc='lift coefficient increment due to optimally deflected trailing edge flaps (default depends on flap type)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FLAP_SPAN_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BTEOB', 'FLOPS': None},
    units='unitless',
    desc='fraction of wing trailing edge with flaps',
    default_value=0.65,
)

add_meta_data(
    Aircraft.Wing.FLAP_TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.JFLTYP', 'FLOPS': None},
    units='unitless',
    default_value=FlapType.DOUBLE_SLOTTED,
    types=FlapType,
    multivalue=True,
    option=True,
    desc='Set the flap type. Available choices are: plain, split, single_slotted, '
    'double_slotted, triple_slotted, fowler, and double_slotted_fowler. '
    'In GASP this was JFLTYP and was provided as an int from 1-7',
)

add_meta_data(
    Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    default_value=False,
    types=bool,
    option=True,
    desc='if true, fold location from the chosen input is an actual fold span, '
    'if false it is normalized to the half span. In GASP this depended on STRUT or YWFOLD',
)

add_meta_data(
    Aircraft.Wing.FOLD_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WWFOLD', 'FLOPS': None},
    units='lbm',
    desc='mass of the folding area of the wing',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.FOLD_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKWFOLD', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of the wing fold',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.FOLDED_SPAN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.YWFOLD', 'FLOPS': None},
    units='ft',
    desc='folded wingspan',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='folded wingspan',
    default_value=1,
)

add_meta_data(
    Aircraft.Wing.FOLDING_AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SWFOLD', 'FLOPS': None},
    units='ft**2',
    desc='wing area of folding part of wings',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKW', 'FLOPS': None},
    units='unitless',
    desc='wing form factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKI', 'FLOPS': None},
    units='unitless',
    desc='wing/fuselage interference factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.GLOVE_AND_BAT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.GLOV',
        # ['&DEFINE.WTIN.GLOV', 'EDETIN.GLOV', '~TOFF.GLOV', '~LNDING.GLOV',
        #    '~PROFIL.GLOV'
        #    ],
    },
    units='ft**2',
    desc='total glove and bat area beyond theoretical wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.HAS_FOLD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    desc='if true a fold will be included in the wing',
    default_value=False,
    types=bool,
)

add_meta_data(
    Aircraft.Wing.HAS_STRUT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    option=True,
    units='unitless',
    default_value=False,
    types=bool,
    desc='if true then aircraft has a strut. In GASP this depended on STRUT',
)

add_meta_data(
    Aircraft.Wing.HEIGHT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HTG', 'FLOPS': None},
    units='ft',
    desc='wing height above ground during ground run, measured at roughly '
    'location of mean aerodynamic chord at the mid plane of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.HIGH_LIFT_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WHLDEV', 'FLOPS': None},
    units='lbm',
    desc='mass of the high lift devices',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WCFLAP', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of high lift devices (default depends on flap type)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.INCIDENCE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.EYEW', 'FLOPS': None},
    units='deg',
    desc='incidence angle of the wings with respect to the fuselage',
    default_value=0.0,
)

add_meta_data(
    # see also: station_locations
    # NOTE required for blended-wing-body type aircraft
    Aircraft.Wing.INPUT_STATION_DISTRIBUTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.ETAW',  # ['&DEFINE.WTIN.ETAW', 'WDEF.ETAW'],
    },
    units='unitless',
    desc='wing station locations as fractions of semispan; overwrites station_locations',
    types=float,
    multivalue=True,
    option=True,
    default_value=[0.0],
)

add_meta_data(
    Aircraft.Wing.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRLW',  # ['&DEFINE.AERIN.TRLW', 'XLAM.TRLW', ],
    },
    units='unitless',
    desc='define percent laminar flow for wing lower surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.LAMINAR_FLOW_UPPER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.TRUW',  # ['&DEFINE.AERIN.TRUW', 'XLAM.TRUW', ],
    },
    units='unitless',
    desc='define percent laminar flow for wing upper surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.LEADING_EDGE_SWEEP,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SWPLE', 'FLOPS': None},
    units='rad',
    desc='sweep angle at leading edge of wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.PDIST',  # ['&DEFINE.WTIN.PDIST', 'WDEF.PDIST'],
    },
    units='unitless',
    desc='controls spatial distribution of integration stations for detailed wing',
    default_value=2.0,
    option=True,
)

add_meta_data(
    Aircraft.Wing.LOAD_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.PCTL',  # ['&DEFINE.WTIN.PCTL', 'WDEF.PCTL'],
    },
    units='unitless',
    desc='fraction of load carried by defined wing',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.LOAD_PATH_SWEEP_DISTRIBUTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.SWL',  # ['&DEFINE.WTIN.SWL', 'WDEF.SWL'],
    },
    units='deg',
    types=float,
    desc='Define the sweep of load path at station locations. Typically '
    'parallel to rear spar tending toward max t/c of airfoil. The Ith value '
    'is used between wing stations I and I+1.',
    default_value=[0.0],
    multivalue=True,
)

# TODO this variable may be uneccessary since we can just check wing loading's value where needed
add_meta_data(
    Aircraft.Wing.LOADING_ABOVE_20,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='if true the wing loading is stated to be above 20 psf. In GASP this depended on WGS',
    option=True,
    default_value=True,
    types=bool,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Wing.MASS_SCALER
    Aircraft.Wing.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(1, 2)', '~WEIGHT.WWING', '~WTSTAT.WSP(1, 2)', '~WWGHT.WWING'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Wing group mass. Contains basic & secondary structures, ailerons/elevons, spoilers, flaps, '
    'and slats.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKWW', 'FLOPS': None},
    units='unitless',
    desc='mass trend coefficient of the wing without high lift devices',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRWI',  # ['&DEFINE.WTIN.FRWI', 'WTS.FRWI'],
    },
    units='unitless',
    desc='mass scaler of the overall wing',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.MATERIAL_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKNO', 'FLOPS': None},
    units='unitless',
    desc='correction factor for the use of non optimum material',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.CAM',
        #  [  # inputs
        #      '&DEFINE.AERIN.CAM', 'EDETIN.CAM',
        #      # outputs
        #      'MISSA.CAM', 'MISSA.CAMX',
        #  ],
    },
    units='unitless',
    desc='Maximum camber at 70 percent semispan, percent of local chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MAX_LIFT_REF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.RCLMAX', 'FLOPS': None},
    units='unitless',
    desc='input reference maximum lift coefficient for basic wing',
)

add_meta_data(
    Aircraft.Wing.MAX_SLAT_DEFLECTION_LANDING,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELLED', 'FLOPS': None},
    units='deg',
    desc='leading edge slat deflection during landing',
    default_value=10,
)

add_meta_data(
    Aircraft.Wing.MAX_SLAT_DEFLECTION_TAKEOFF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELLED', 'FLOPS': None},
    units='deg',
    desc='leading edge slat deflection during takeoff',
    default_value=10,
)

add_meta_data(
    Aircraft.Wing.MAX_THICKNESS_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XCTCMX', 'FLOPS': None},
    units='unitless',
    desc='location (percent chord) of max wing thickness',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MIN_PRESSURE_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XCPS', 'FLOPS': None},
    units='unitless',
    desc='location (percent chord) of peak suction',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override
    #    - see also: Aircraft.Wing.MISC_MASS_SCALER
    Aircraft.Wing.MISC_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WWGHT.W3',
    },
    units='lbm',
    desc='wing mass breakdown term 3',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MISC_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRWI3',  # ['&DEFINE.WTIN.FRWI3', 'WIOR3.FRWI3'],
    },
    units='unitless',
    desc='mass scaler of the miscellaneous wing mass term',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.NUM_FLAP_SEGMENTS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FLAPN', 'FLOPS': None},
    units='unitless',
    desc='number of flap segments per wing panel',
    types=int,
    option=True,
    default_value=2,
)

add_meta_data(
    Aircraft.Wing.NUM_INTEGRATION_STATIONS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.NSTD', 'WDEF.NSTD', '~BNDMAT.NSD', '~DETA.NSD'],
        'FLOPS': 'WTIN.NSTD',
    },
    units='unitless',
    desc='number of integration stations',
    types=int,
    option=True,
    default_value=50,
)

add_meta_data(
    Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELTEO', 'FLOPS': None},
    units='deg',
    desc='optimum flap deflection angle (default depends on flap type)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELLEO', 'FLOPS': None},
    units='deg',
    desc='optimum slat deflection angle',
    default_value=20,
)

add_meta_data(
    Aircraft.Wing.OUTBOARD_SEMISPAN,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.OSSPAN'},
    units='ft',
    desc='Outboard semispan (used if a detailed wing outboard is being added to a BWB fuselage)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.ROOT_CHORD,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CROOTW',
        'FLOPS': 'WTIN.XLW',
    },
    units='ft',
    desc='wing chord length at at the wing/fuselage intersection',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override
    #    - see also: Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER
    Aircraft.Wing.SHEAR_CONTROL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WWGHT.W2',
    },
    units='lbm',
    desc='wing mass breakdown term 2',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRWI2',  # ['&DEFINE.WTIN.FRWI2', 'WIOR3.FRWI2'],
    },
    units='unitless',
    desc='mass scaler of the shear and control term',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.SLAT_CHORD_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLEOC', 'FLOPS': None},
    units='unitless',
    desc='ratio of slat chord to wing chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCLMLE', 'FLOPS': None},
    units='unitless',
    desc='lift coefficient increment due to optimally deflected LE slats',
)

add_meta_data(
    Aircraft.Wing.SLAT_SPAN_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BLEOB', 'FLOPS': None},
    units='unitless',
    desc='fraction of wing leading edge with slats',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SPAN,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.B',
        'FLOPS': 'WTIN.SPAN',
        #  [  # inputs
        #      '&DEFINE.WTIN.SPAN',
        #      # outputs
        #      '~WEIGHT.B', '~WWGHT.B', '~GESURF.B'
        #  ],
    },
    units='ft',
    desc='span of main wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SPAN_EFFICIENCY_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.E',  # ['&DEFINE.AERIN.E', 'OSWALD.E', ],
    },
    units='unitless',
    desc='coefficient for calculating span efficiency for extreme taper ratios',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.MIKE',  # ['&DEFINE.AERIN.MIKE', 'MIMOD.MIKE'],
    },
    units='unitless',
    desc='Define a switch for span efficiency reduction for extreme taper '
    'ratios: True == a span efficiency factor '
    '(*wing_span_efficiency_factor0*) is calculated based on wing taper ratio '
    'and aspect ratio; False == a span efficiency factor '
    '(*wing_span_efficiency_factor0*) is set to 1.0.',
    option=True,
    types=bool,
    default_value=False,
)

add_meta_data(
    Aircraft.Wing.STRUT_BRACING_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.FSTRT', 'WTS.FSTRT', '~WWGHT.FSTRT', '~BNDMAT.FSTRT'],
        'FLOPS': 'WTIN.FSTRT',
    },
    units='unitless',
    desc='Define the wing strut-bracing factor where: 0.0 == no wing-strut; '
    '1.0 == full benefit from strut bracing.',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER
    Aircraft.Wing.SURFACE_CONTROL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(16, 2)', '~WEIGHT.WSC', '~WTSTAT.WSP(16, 2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='mass of surface controls',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKFW', 'FLOPS': None},
    units='unitless',
    desc='Surface controls weight coefficient',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRSC',  # ['&DEFINE.WTIN.FRSC', 'WTS.FRSC'],
    },
    units='unitless',
    desc='Surface controls mass scaler',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.SWEEP,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.DLMC4',
        'FLOPS': 'CONFIN.SWEEP',
        #  [  # inputs
        #      '&DEFINE.CONFIN.SWEEP', 'PARVAR.DVD(1,6)',
        #      # outputs
        #      'CONFIG.SWEEP', 'CONFIG.DVA(6)', '~FLOPS.DVA(6)', '~ANALYS.DVA(6)',
        #      '~WWGHT.SWEEP', '~INERT.SWEEP',
        #      # other
        #      'MISSA.SW25', '~BUFFET.SW25', '~CDCC.SW25', '~CLDESN.SW25',
        #      '~MDESN.SW25',
        #  ],
    },
    units='deg',
    desc='quarter-chord sweep angle of the wing',
    default_value=0.0,  # TODO required.
)

add_meta_data(
    Aircraft.Wing.TAPER_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SLM',
        'FLOPS': 'CONFIN.TR',
        #  [  # inputs
        #      '&DEFINE.CONFIN.TR', 'PARVAR.DVD(1,5)',
        #      # outputs
        #      'CONFIG.TR', 'CONFIG.DVA(5)', 'CONFIG.TR1', '~FLOPS.DVA(5)',
        #      '~ANALYS.DVA(5)', '~GESURF.TR', '~WWGHT.TR', '~INERT.TR',
        #      # other
        #      'MISSA.TAPER', '~CDCC.TAPER', '~MDESN.TAPER',
        #  ],
    },
    units='unitless',
    desc='taper ratio of the wing',
    default_value=0.0,  # TODO required.
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'CONFIN.TCA',
        #  [  # inputs
        #      '&DEFINE.CONFIN.TCA', 'PARVAR.DVD(1,7)',
        #      # outputs
        #      'CONFIG.TCA', 'CONFIG.DVA(7)', '~FLOPS.DVA(7)', '~ANALYS.DVA(7)',
        #      '~WWGHT.TCA',
        #      # other
        #      'MISSA.TC', '~BUFFET.TC', '~CDCC.TC', '~CDPP.TC', '~CLDESN.TC',
        #      '~MDESN.TC',
        #  ],
    },
    units='unitless',
    desc='wing thickness-chord ratio (weighted average)',
    default_value=0.0,  # TODO required
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_DISTRIBUTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TOC',  # ['&DEFINE.WTIN.TOC', 'WDEF.TOC'],
    },
    units='unitless',
    desc='the thickeness-chord ratios at station locations',
    default_value=[0.0],
    types=float,
    multivalue=True,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TCREF',  # ['&DEFINE.WTIN.TCREF'],
    },
    units='unitless',
    desc='Reference thickness-to-chord ratio, used for detailed wing mass estimation.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TCR', 'FLOPS': None},
    units='unitless',
    desc='thickness-to-chord ratio at the root of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TCT', 'FLOPS': None},
    units='unitless',
    desc='thickness-to-chord ratio at the tip of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TC', 'FLOPS': None},
    units='unitless',
    desc='wing thickness-chord ratio at the wing station of the mean aerodynamic chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ULF',
        # ['&DEFINE.WTIN.ULF', 'WTS.ULF', '~WWGHT.ULF'],
        'FLOPS': 'WTIN.ULF',
    },
    units='unitless',
    desc='structural ultimate load factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.VAR_SWEEP_MASS_PENALTY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.VARSWP', 'FAWT.VARSWP', '~WWGHT.VARSWP'],
        'FLOPS': 'WTIN.VARSWP',
    },
    units='unitless',
    desc='Define the fraction of wing variable sweep mass penalty where: '
    '0.0 == fixed-geometry wing; 1.0 == full variable-sweep wing.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.VERTICAL_MOUNT_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HWING', 'FLOPS': None},
    units='unitless',
    desc='vertical wing mount location on fuselage (0 = low wing, 1 = high wing). It is continuous variable between 0 and 1 are acceptable.',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Wing.WETTED_AREA_SCALER
    Aircraft.Wing.WETTED_AREA,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['ACTWET.SWTWG', 'MISSA.SWET[1]'],
    },
    units='ft**2',
    desc='wing wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.WETTED_AREA_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.SWETW',  # ['&DEFINE.AERIN.SWETW', 'AWETO.SWETW', ],
    },
    units='unitless',
    desc='wing wetted area scaler',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.ZERO_LIFT_ANGLE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ALPHL0', 'FLOPS': None},
    units='deg',
    desc='zero lift angle of attack',
    default_value=0.0,
)

# ============================================================================================================================================
#  .----------------.  .----------------.  .-----------------. .----------------.  .----------------.  .----------------.  .----------------.
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
# | |  ________    | || |  ____  ____  | || | ____  _____  | || |      __      | || | ____    ____ | || |     _____    | || |     ______   | |
# | | |_   ___ `.  | || | |_  _||_  _| | || ||_   \|_   _| | || |     /  \     | || ||_   \  /   _|| || |    |_   _|   | || |   .' ___  |  | |
# | |   | |   `. \ | || |   \ \  / /   | || |  |   \ | |   | || |    / /\ \    | || |  |   \/   |  | || |      | |     | || |  / .'   \_|  | |
# | |   | |    | | | || |    \ \/ /    | || |  | |\ \| |   | || |   / ____ \   | || |  | |\  /| |  | || |      | |     | || |  | |         | |
# | |  _| |___.' / | || |    _|  |_    | || | _| |_\   |_  | || | _/ /    \ \_ | || | _| |_\/_| |_ | || |     _| |_    | || |  \ `.___.'\  | |
# | | |________.'  | || |   |______|   | || ||_____|\____| | || ||____|  |____|| || ||_____||_____|| || |    |_____|   | || |   `._____.'  | |
# | |              | || |              | || |              | || |              | || |              | || |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'
# ============================================================================================================================================

#              _                                       _
#      /\     | |                                     | |
#     /  \    | |_   _ __ ___     ___    ___   _ __   | |__     ___   _ __    ___
#    / /\ \   | __| | '_ ` _ \   / _ \  / __| | '_ \  | '_ \   / _ \ | '__|  / _ \
#   / ____ \  | |_  | | | | | | | (_) | \__ \ | |_) | | | | | |  __/ | |    |  __/
#  /_/    \_\  \__| |_| |_| |_|  \___/  |___/ | .__/  |_| |_|  \___| |_|     \___|
#                                             | |
#                                             |_|
# ================================================================================

add_meta_data(
    Dynamic.Atmosphere.DENSITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/ft**3',
    desc="Atmospheric density at the vehicle's current altitude",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.DYNAMIC_PRESSURE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf/ft**2',
    desc="Atmospheric dynamic pressure at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'XKV', 'FLOPS': None},
    units='lbf*s/ft**2',
    desc="Atmospheric dynamic viscosity at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)


add_meta_data(
    Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'XKV', 'FLOPS': None},
    units='ft**2/s',
    desc="Atmospheric kinematic viscosity at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.MACH,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Current Mach number of the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.MACH_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Current rate at which the Mach number of the vehicle is changing',
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.SPEED_OF_SOUND,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft/s',
    desc="Atmospheric speed of sound at vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.STATIC_PRESSURE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf/ft**2',
    desc="Atmospheric static pressure at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.TEMPERATURE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='degR',
    desc="Atmospheric temperature at vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)


#  __  __   _               _
# |  \/  | (_)             (_)
# | \  / |  _   ___   ___   _    ___    _ __
# | |\/| | | | / __| / __| | |  / _ \  | '_ \
# | |  | | | | \__ \ \__ \ | | | (_) | | | | |
# |_|  |_| |_| |___/ |___/ |_|  \___/  |_| |_|
# ============================================
add_meta_data(
    Dynamic.Mission.ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft',
    desc='Current geometric altitude of the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.ALTITUDE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft/s',
    desc='Current rate of altitude change (climb rate) of the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.ALTITUDE_RATE_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft/s',
    desc='Current maximum possible rate of altitude change (climb rate) of the vehicle '
    '(at hypothetical maximum thrust condition)',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.DISTANCE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'range'},
    units='NM',
    desc='The total distance the vehicle has traveled since brake release at the current time',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.DISTANCE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'range_rate'},
    units='NM/s',
    desc='The rate at which the distance traveled is changing at the current time',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.FLIGHT_PATH_ANGLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='rad',
    desc='Current flight path angle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='rad/s',
    desc='Current rate at which flight path angle is changing',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.SPECIFIC_ENERGY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='m/s',
    desc='Rate of change in specific energy (energy per unit weight) of the vehicle at '
    'current flight condition',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.SPECIFIC_ENERGY_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='m/s',
    desc='Rate of change in specific energy (specific power) of the vehicle at current '
    'flight condition',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='m/s',
    desc='Specific excess power of the vehicle at current flight condition and at '
    'hypothetical maximum thrust',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.VELOCITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft/s',
    desc='Current velocity of the vehicle along its body axis',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.VELOCITY_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft/s**2',
    desc='Current rate of change in velocity (acceleration) of the vehicle along its body axis',
    multivalue=True,
)

#  __      __         _       _          _
#  \ \    / /        | |     (_)        | |
#   \ \  / /    ___  | |__    _    ___  | |   ___
#    \ \/ /    / _ \ | '_ \  | |  / __| | |  / _ \
#     \  /    |  __/ | | | | | | | (__  | | |  __/
#      \/      \___| |_| |_| |_|  \___| |_|  \___|
# ================================================

add_meta_data(
    Dynamic.Vehicle.ANGLE_OF_ATTACK,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='deg',
    desc='Angle between aircraft wing cord and relative wind',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc="battery's current state of charge",
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='kJ',
    desc='Total amount of electric energy consumed by the vehicle up until this point '
    'in the mission',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.DRAG,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='Current total drag experienced by the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.LIFT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='Current total lift produced by the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Current total mass of the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.MASS_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/s',
    desc='Current rate at which the mass of the vehicle is changing',
    multivalue=True,
)

#   ___                             _        _
#  | _ \  _ _   ___   _ __   _  _  | |  ___ (_)  ___   _ _
#  |  _/ | '_| / _ \ | '_ \ | || | | | (_-< | | / _ \ | ' \
#  |_|   |_|   \___/ | .__/  \_,_| |_| /__/ |_| \___/ |_||_|
#                    |_|
# ==========================================================

add_meta_data(
    Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='kW',
    desc='The electric power consumption of each engine during the mission.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='kW',
    desc='Current total electric power consumption of the vehicle',
    multivalue=True,
)

# add_meta_data(
#     Dynamic.Vehicle.Propulsion.EXIT_AREA,
#     meta_data=_MetaData,
#     historical_name={'GASP': None,
#                     'FLOPS': None,
#                     },
#     units='kW',
#     desc='Current nozzle exit area of engines, per single instance of each '
#          'engine model'
# )

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/h',
    desc='Current rate of fuel consumption of the vehicle, per single instance of '
    'each engine model. Consumption (i.e. mass reduction) of fuel is defined as '
    'positive.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/h',
    desc='Current rate of fuel consumption of the vehicle, per single instance of each '
    'engine model. Consumption (i.e. mass reduction) of fuel is defined as negative.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/h',
    desc='Current rate of total fuel consumption of the vehicle. Consumption (i.e. '
    'mass reduction) of fuel is defined as negative.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/h',
    desc='Current rate of total fuel consumption of the vehicle. Consumption (i.e. '
    'mass reduction) of fuel is defined as positive.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.HYBRID_THROTTLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Current secondary throttle setting of each individual engine model on the '
    'vehicle, used as an additional degree of control for hybrid engines',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.NOX_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/h',
    desc='Current rate of nitrous oxide (NOx) production by the vehicle, per single '
    'instance of each engine model',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm/h',
    desc='Current total rate of nitrous oxide (NOx) production by the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft/s',
    desc='linear propeller tip speed due to rotation (not airspeed at propeller tip)',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.RPM,
    meta_data=_MetaData,
    historical_name={'GASP': ['RPM', 'RPMe'], 'FLOPS': None},
    units='rpm',
    desc='Rotational rate of shaft, per engine.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.SHAFT_POWER,
    meta_data=_MetaData,
    historical_name={'GASP': ['SHP, EHP'], 'FLOPS': None},
    units='hp',
    desc='current shaft power, per engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='hp',
    desc='The maximum possible shaft power currently producible, per engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='degR',
    desc='Current turbine exit temperature (T4) of turbine engines on vehicle, per '
    'single instance of each engine model',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THROTTLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='Current throttle setting for each individual engine model on the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='Current net thrust produced by engines, per single instance of each engine model',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='Hypothetical maximum possible net thrust that can be produced per single '
    "instance of each engine model at the vehicle's current flight condition",
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='Hypothetical maximum possible net thrust produced by the vehicle at its '
    'current flight condition',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbf',
    desc='Current total net thrust produced by the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.TORQUE,
    meta_data=_MetaData,
    historical_name={'GASP': 'TORQUE', 'FLOPS': None},
    units='N*m',
    desc='Current torque being produced, per engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.TORQUE_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='N*m',
    desc='Hypothetical maximum possible torque being produced at the current flight '
    'condition, per engine',
    multivalue=True,
)

# ============================================================================================================================================
#  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .-----------------.
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
# | | ____    ____ | || |     _____    | || |    _______   | || |    _______   | || |     _____    | || |     ____     | || | ____  _____  | |
# | ||_   \  /   _|| || |    |_   _|   | || |   /  ___  |  | || |   /  ___  |  | || |    |_   _|   | || |   .'    `.   | || ||_   \|_   _| | |
# | |  |   \/   |  | || |      | |     | || |  |  (__ \_|  | || |  |  (__ \_|  | || |      | |     | || |  /  .--.  \  | || |  |   \ | |   | |
# | |  | |\  /| |  | || |      | |     | || |   '.___`-.   | || |   '.___`-.   | || |      | |     | || |  | |    | |  | || |  | |\ \| |   | |
# | | _| |_\/_| |_ | || |     _| |_    | || |  |`\____) |  | || |  |`\____) |  | || |     _| |_    | || |  \  `--'  /  | || | _| |_\   |_  | |
# | ||_____||_____|| || |    |_____|   | || |  |_______.'  | || |  |_______.'  | || |    |_____|   | || |   `.____.'   | || ||_____|\____| | |
# | |              | || |              | || |              | || |              | || |              | || |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'
#  ============================================================================================================================================

add_meta_data(
    Mission.BLOCK_FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Fuel burned from taxi out of the gate through the regular missions to taxi into the gate.'
    'This does not include fuel burned in reserve phases. This works for energy-state EOM. Not used in 2DOF EOM',
)

add_meta_data(
    Mission.FINAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'None', 'FLOPS': None},  # TODO: Check on these
    units='lbm',
    desc='The final weight of the vehicle at the end of the last regular_phase (does not include reserve phases).',
)

add_meta_data(
    Mission.FINAL_TIME,
    meta_data=_MetaData,
    historical_name={'GASP': 'None', 'FLOPS': None},  # TODO: Check on these
    units='min',
    desc='Total mission time from the start of the first regular_phase'
    'to the end of the last regular_phase (does not include reserve phases).',
)

add_meta_data(
    Mission.FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Fuel burned from taxi-out through all regular phases of the mission (e.g. takeoff, climb, cruse, descent, landing).'
    'This does not include fuel burned in reserve phases or taxi-in.'
    'The only time taxi-in would be included in this is if the user'
    'specifies a taxi phase as part of the regular mission phases.',
)

add_meta_data(
    Mission.GROSS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Gross takeoff mass of aircraft for the mission being flown.'
    'May differ from Aircraft.Design.GROSS_MASS for off-design missions.',
)

add_meta_data(
    Mission.OPERATING_ITEMS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFUL', 'FLOPS': None},
    units='lbm',
    desc='Useful load group. Includes crew, unusable fuel, and oil mass.',
    default_value=0.0,
)

add_meta_data(
    Mission.OPERATING_MASS,
    meta_data=_MetaData,
    # TODO: check with Aviary and GASPy engineers to ensure these are indeed
    # defined the same way
    historical_name={
        'GASP': 'INGASP.OWE',
        # ['WTS.WSP(33, 2)', '~WEIGHT.WOWE', '~WTSTAT.WSP(33, 2)'],
        'FLOPS': 'MISSIN.DOWE',
    },
    units='lbm',
    desc='Operating mass of the aircraft. Includes structure mass, crew (and crew baggage), unusable '
    'fuel, oil, and operational items like cargo containers and passenger service mass.',
    default_value=0.0,
)

add_meta_data(
    Mission.RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='NM',
    desc='actual range that the aircraft flies on this mission. Equal to Aircraft.Design.RANGE value in the design case.',
)

add_meta_data(
    Mission.RESERVE_FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='fuel burned during reserve phases, this does not include fuel burned in regular phases',
    default_value=0.0,
)

add_meta_data(
    Mission.RESERVE_FUEL_ADDITIONAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FRESF', 'FLOPS': None},
    option=True,
    units='lbm',
    desc='required fuel reserves: directly in lbm',
    default_value=0,
)

add_meta_data(
    Mission.RESERVE_FUEL_MARGIN,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    option=True,
    units='unitless',
    desc='required fuel reserves: given as a precentage of mission fuel.'
    'Mission fuel only includes normal phases and excludes reserve phases.',
    default_value=0,
)

add_meta_data(
    Mission.TOTAL_FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFA', 'FLOPS': None},
    units='lbm',
    # Note: In GASP, WFA does not include fuel margin.
    desc='total fuel carried at the beginnning of a mission includes fuel burned in the mission, '
    'reserve fuel and fuel margin',
)

add_meta_data(
    Mission.TOTAL_RESERVE_FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='the total fuel reserves which is the sum of: '
    'Mission.RESERVE_FUEL, Mission.RESERVE_FUEL_ADDITIONAL, Mission.RESERVE_FUEL_MARGIN',
    default_value=0,
)

add_meta_data(
    Mission.ZERO_FUEL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(37,2)', '~WEIGHT.WZF', '~WTSTAT.WSP(37,2)'],
        'FLOPS': None,
    },
    units='lbm',
    desc='Aircraft zero fuel mass. Includes operating mass, passengers, baggage, and cargo.',
    default_value=0.0,
)

#   _____                         _                    _           _
#  / ____|                       | |                  (_)         | |
# | |        ___    _ __    ___  | |_   _ __    __ _   _   _ __   | |_   ___
# | |       / _ \  | '_ \  / __| | __| | '__|  / _` | | | | '_ \  | __| / __|
# | |____  | (_) | | | | | \__ \ | |_  | |    | (_| | | | | | | | | |_  \__ \
#  \_____|  \___/  |_| |_| |___/  \__| |_|     \__,_| |_| |_| |_|  \__| |___/
# ===========================================================================

add_meta_data(
    Mission.Constraints.EXCESS_FUEL_CAPACITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Difference between the usable fuel capacity on the aircraft and the total fuel (including reserve) required for the mission. '
    'Must be >= 0 to ensure that the aircraft has enough fuel to complete the mission',
)

add_meta_data(
    Mission.Constraints.GEARBOX_SHAFT_POWER_RESIDUAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='kW',
    desc='Must be zero or positive to ensure that the gearbox is sized large enough to handle the maximum shaft power the engine could output during any part of the mission',
)

add_meta_data(
    Mission.Constraints.MASS_RESIDUAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='residual to make sure aircraft mass closes on actual '
    'gross takeoff mass, value should be zero at convergence '
    '(within acceptable tolerance)',
)

add_meta_data(
    Mission.Constraints.MAX_MACH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.VMMO',
        #  [  # inputs
        #      '&DEFINE.WTIN.VMMO', 'VLIMIT.VMMO',
        #      # outputs
        #      'VLIMIT.VMAX',
        #  ],
    },
    units='unitless',
    desc='aircraft cruise Mach number',
    # TODO: derived default value: Aircraft.Design.CRUISE_MACH ???
    default_value=0.0,
    option=True,
)

add_meta_data(
    Mission.Constraints.RANGE_RESIDUAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='NM',
    desc='residual to make sure aircraft range is equal to the targeted '
    'range, value should be zero at convergence (within acceptable '
    'tolerance)',
)

add_meta_data(
    Mission.Constraints.RANGE_RESIDUAL_RESERVE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='NM',
    desc='residual to make sure aircraft reserve mission range is equal to the targeted '
    'range, value should be zero at convergence (within acceptable '
    'tolerance)',
)

#  _____                 _
# |  __ \               (_)
# | |  | |   ___   ___   _    __ _   _ __
# | |  | |  / _ \ / __| | |  / _` | | '_ \
# | |__| | |  __/ \__ \ | | | (_| | | | | |
# |_____/   \___| |___/ |_|  \__, | |_| |_|
#                             __/ |
#                            |___/
# =========================================

#  _                            _   _
# | |                          | | (_)
# | |        __ _   _ __     __| |  _   _ __     __ _
# | |       / _` | | '_ \   / _` | | | | '_ \   / _` |
# | |____  | (_| | | | | | | (_| | | | | | | | | (_| |
# |______|  \__,_| |_| |_|  \__,_| |_| |_| |_|  \__, |
#                                                __/ |
#                                               |___/
# ====================================================

add_meta_data(
    Mission.Landing.AIRPORT_ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ALTLND', 'FLOPS': None},
    units='ft',
    desc='altitude of airport where aircraft lands',
    default_value=0,
)

add_meta_data(
    Mission.Landing.BRAKING_DELAY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TDELAY', 'FLOPS': None},
    units='s',
    desc='time delay between touchdown and the application of brakes',
    default_value=1,
)

add_meta_data(
    Mission.Landing.BRAKING_FRICTION_COEFFICIENT,
    meta_data=_MetaData,
    # NOTE: FLOPS uses the same value for both takeoff and landing
    # historical_name={
    #     'FLOPS': ['&DEFTOL.TOLIN.BRAKMU', 'BALFLD.BRAKMU'],
    #     'GASP': None,
    historical_name={'FLOPS': None, 'GASP': None},
    default_value=0.3,
    units='unitless',
    desc='landing coefficient of friction, with brakes on',
)

add_meta_data(
    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCD', 'FLOPS': None},
    units='unitless',
    desc='drag coefficient increment at landing due to flaps',
)

add_meta_data(
    Mission.Landing.DRAG_COEFFICIENT_MIN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'TOLIN.CDMLD',  # ['&DEFINE.AERIN.CDMLD', 'LANDG.CDMLD'],
    },
    units='unitless',
    desc='Minimum drag coefficient for takeoff. Typically this is CD at zero lift.',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.FIELD_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~ANALYS.FARLDG',
    },
    units='ft',
    desc='FAR landing field length',
)

add_meta_data(
    Mission.Landing.FLARE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'TOLIN.VANGLD'},
    units='deg/s',
    desc='flare rate in detailed landing',
    default_value=2.0,
)

add_meta_data(
    Mission.Landing.GLIDE_TO_STALL_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VRATT', 'FLOPS': None},
    units='unitless',
    desc='ratio of glide (approach) speed to stall speed',
    default_value=1.3,
)

add_meta_data(
    Mission.Landing.GROUND_DISTANCE,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.DLT',  # Is DLT actual landing distance or field length?
        'FLOPS': None,
    },
    units='ft',
    desc='distance covered over the ground during landing',
)

add_meta_data(
    Mission.Landing.INITIAL_ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HIN', 'FLOPS': None},
    units='ft',
    desc='altitude where landing calculations begin',
)

add_meta_data(
    Mission.Landing.INITIAL_MACH,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='approach Mach number',
    default_value=0.1,
)

add_meta_data(
    Mission.Landing.INITIAL_VELOCITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.VGL',
        'FLOPS': 'AERIN.VAPPR',
    },
    units='ft/s',
    desc='approach velocity',
)

add_meta_data(
    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCL', 'FLOPS': None},
    units='unitless',
    desc='lift coefficient increment at landing due to flaps',
)

add_meta_data(
    # TODO: missing &DEFINE.AERIN.CLAPP ???
    #    - NOTE: there is a relationship in FLOPS between CLAPP and
    #      CLLDM (this variable)
    Mission.Landing.LIFT_COEFFICIENT_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CLMWLD',
        'FLOPS': 'AERIN.CLLDM',  # ['&DEFINE.AERIN.CLLDM', 'LANDG.CLLDM'],
    },
    units='unitless',
    desc='maximum lift coefficient for landing',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XLFMX', 'FLOPS': None},
    units='unitless',
    desc='maximum load factor during landing flare',
    default_value=1.15,
)

add_meta_data(
    Mission.Landing.MAXIMUM_SINK_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.RSMX', 'FLOPS': None},
    units='ft/min',
    desc='maximum rate of sink during glide',
    default_value=1000,
)

add_meta_data(
    Mission.Landing.OBSTACLE_HEIGHT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HAPP', 'FLOPS': None},
    units='ft',
    desc='landing obstacle height above the ground at airport altitude',
    default_value=50,
)

add_meta_data(
    Mission.Landing.ROLLING_FRICTION_COEFFICIENT,
    meta_data=_MetaData,
    # historical_name={"GASP": None,
    #                  "FLOPS": ['&DEFTOL.TOLIN.ROLLMU', 'BALFLD.ROLLMU'],
    #                  },
    historical_name={'FLOPS': None, 'GASP': None},
    units='unitless',
    desc='coefficient of rolling friction for groundroll portion of takeoff',
    default_value=0.025,
)

add_meta_data(
    Mission.Landing.SPOILER_DRAG_COEFFICIENT,
    meta_data=_MetaData,
    # historical_name={"GASP": None,
    #                  "FLOPS": '&DEFTOL.TOLIN.CDSPOL',
    #                  },
    historical_name={'FLOPS': None, 'GASP': None},
    units='unitless',
    desc='drag coefficient for spoilers during landing rollout',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.SPOILER_LIFT_COEFFICIENT,
    meta_data=_MetaData,
    # historical_name={"GASP": None,
    #                  "FLOPS": '&DEFTOL.TOLIN.CLSPOL',
    #                  },
    historical_name={'FLOPS': None, 'GASP': None},
    units='unitless',
    desc='lift coefficient for spoilers during landing rollout',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.STALL_VELOCITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VST', 'FLOPS': None},
    units='ft/s',
    desc='stall speed during approach',
)

add_meta_data(
    Mission.Landing.TOUCHDOWN_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['~ANALYS.WLDG', '~LNDING.GROSWT'],
    },
    units='lbm',
    desc='computed mass of aircraft for landing, is only '
    'required to be equal to Aircraft.Design.TOUCHDOWN_MASS_MAX '
    'when the design case is being run '
    'for ENERGY_STATE missions this is the mass at the end of the last regular phase (non-reserve phase)',
)

add_meta_data(
    Mission.Landing.TOUCHDOWN_SINK_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SINKTD', 'FLOPS': None},
    units='ft/s',
    desc='sink rate at touchdown',
    default_value=3,
)

#   ____    _         _                 _     _
#  / __ \  | |       (_)               | |   (_)
# | |  | | | |__      _    ___    ___  | |_   _  __   __   ___   ___
# | |  | | | '_ \    | |  / _ \  / __| | __| | | \ \ / /  / _ \ / __|
# | |__| | | |_) |   | | |  __/ | (__  | |_  | |  \ V /  |  __/ \__ \
#  \____/  |_.__/    | |  \___|  \___|  \__| |_|   \_/    \___| |___/
#                   _/ |
#                  |__/
# ===================================================================

add_meta_data(
    Mission.Objectives.FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='regularized objective that minimizes total fuel mass subject '
    'to other necessary additions',
)

add_meta_data(
    Mission.Objectives.RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='regularized objective that maximizes range subject to other necessary additions',
)

#  _______           _                      __    __
# |__   __|         | |                    / _|  / _|
#    | |      __ _  | | __   ___    ___   | |_  | |_
#    | |     / _` | | |/ /  / _ \  / _ \  |  _| |  _|
#    | |    | (_| | |   <  |  __/ | (_) | | |   | |
#    |_|     \__,_| |_|\_\  \___|  \___/  |_|   |_|
# ===================================================

add_meta_data(
    Mission.Takeoff.AIRPORT_ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft',
    desc='altitude of airport where aircraft takes off',
)

add_meta_data(
    Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY,
    meta_data=_MetaData,
    historical_name={
        # ['&DEFTOL.TOLIN.ALPRUN', 'BALFLD.ALPRUN', '~CLGRAD.ALPRUN'],
        'FLOPS': 'TOLIN.ALPRUN',
        'GASP': None,
    },
    option=True,
    default_value=0.0,
    units='deg',
    desc='angle of attack on ground',
)

add_meta_data(
    Mission.Takeoff.ASCENT_DURATION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='s',
    desc='duration of the ascent phase of takeoff',
)

add_meta_data(
    Mission.Takeoff.ASCENT_T_INITIAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='s',
    desc='time that the ascent phase of takeoff starts at',
    default_value=10,
)

add_meta_data(
    Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT,
    meta_data=_MetaData,
    # NOTE: FLOPS uses the same value for both takeoff and landing
    historical_name={
        'FLOPS': 'TOLIN.BRAKMU',  # ['&DEFTOL.TOLIN.BRAKMU', 'BALFLD.BRAKMU'],
        'GASP': None,
    },
    default_value=0.3,
    units='unitless',
    desc='takeoff coefficient of friction, with brakes on',
)

add_meta_data(
    Mission.Takeoff.DECISION_SPEED_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DV1', 'FLOPS': None},
    units='kn',
    desc='increment of engine failure decision speed above stall speed',
    default_value=5,
)

add_meta_data(
    Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCD', 'FLOPS': None},
    units='unitless',
    desc='drag coefficient increment at takeoff due to flaps',
)

add_meta_data(
    Mission.Takeoff.DRAG_COEFFICIENT_MIN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'TOLIN.CDMTO',  # ['&DEFINE.AERIN.CDMTO', 'LANDG.CDMTO'],
    },
    units='unitless',
    desc='Minimum drag coefficient for takeoff. Typically this is CD at zero lift.',
    default_value=0.0,
)

add_meta_data(
    Mission.Takeoff.FIELD_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~ANALYS.FAROFF',
    },
    units='ft',
    desc='FAR takeoff field length',
)

add_meta_data(
    Mission.Takeoff.FINAL_ALTITUDE,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFTOL.TOLIN.OBSTO', 'TOCOMM.OBSTO', 'TOCOMM.DUMC(8)'],
        'FLOPS': 'TOLIN.OBSTO',
    },
    units='ft',
    desc='altitude of aircraft at the end of takeoff',
    # Note default value is aircraft type dependent
    #    - transport: 35 ft
    # assume transport for now
    default_value=35.0,
)

add_meta_data(
    Mission.Takeoff.FINAL_MACH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
    },
    units='unitless',
    desc='Mach number of aircraft after taking off and clearing a 35 foot obstacle',
)

add_meta_data(
    Mission.Takeoff.FINAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='mass after aircraft has cleared 35 ft obstacle',
)

add_meta_data(
    # TODO FLOPS implementation is different from Aviary
    #    - correct variable reference?
    #    - correct Aviary equations?
    Mission.Takeoff.FINAL_VELOCITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~TOFF.V2',
    },
    units='m/s',
    desc='velocity of aircraft after taking off and clearing a 35 foot obstacle',
)

add_meta_data(
    # Note user override (no scaling)
    # Note FLOPS calculated as part of mission analysis, and not as
    # part of takeoff
    Mission.Takeoff.FUEL,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFMSS.MISSIN.FTKOFL', 'FFLALL.FTKOFL', '~MISSON.TAKOFL'],
        'FLOPS': 'MISSIN.FTKOFL',
    },
    units='lbm',
    desc='Fuel burned during takeoff for energy-state EOM. Not used in 2DOF EOM.',
    default_value=0.0,
)

add_meta_data(
    Mission.Takeoff.GROUND_DISTANCE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='ft',
    desc='ground distance covered by takeoff with all engines operating',
)

add_meta_data(
    Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCL', 'FLOPS': None},
    units='unitless',
    desc='lift coefficient increment at takeoff due to flaps',
)

add_meta_data(
    Mission.Takeoff.LIFT_COEFFICIENT_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CLMWTO',
        # ['&DEFINE.AERIN.CLTOM', 'LANDG.CLTOM', '~DEFTOL.CLTOA'],
        'FLOPS': ['AERIN.CLTOM', 'TOLIN.CLTOM'],
    },
    units='unitless',
    desc='maximum lift coefficient for takeoff',
    default_value=2.0,
)

add_meta_data(
    Mission.Takeoff.LIFT_OVER_DRAG,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~ANALYS.CLOD',
    },
    units='unitless',
    desc='ratio of lift to drag at takeoff',
)

add_meta_data(
    Mission.Takeoff.OBSTACLE_HEIGHT,
    meta_data=_MetaData,
    # historical_name={
    #     'GASP': None,
    #     'FLOPS': ['&DEFTOL.TOLIN.OBSTO', 'TOCOMM.OBSTO', 'TOCOMM.DUMC(8)'],
    historical_name={'GASP': None, 'FLOPS': None},
    option=True,
    # Note default value is aircraft type dependent
    #    - transport: 35 ft
    # assume transport for now
    default_value=35.0,
    units='ft',
    desc='takeoff obstacle height above the ground at airport altitude',
)

add_meta_data(
    Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFTOL.TOLIN.ROLLMU', 'BALFLD.ROLLMU'],
        'FLOPS': 'TOLIN.ROLLMU',
    },
    units='unitless',
    desc='coefficient of rolling friction for groundroll portion of takeoff',
    default_value=0.025,
)

add_meta_data(
    Mission.Takeoff.ROTATION_SPEED_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DVR', 'FLOPS': None},
    units='kn',
    desc='increment of takeoff rotation speed above engine failure decision speed',
    default_value=5,
)

add_meta_data(
    Mission.Takeoff.ROTATION_VELOCITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VR', 'FLOPS': None},
    units='kn',
    desc='rotation velocity',
)

add_meta_data(
    Mission.Takeoff.SPOILER_DRAG_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'TOLIN.CDSPOL',  # '&DEFTOL.TOLIN.CDSPOL',
    },
    units='unitless',
    desc='drag coefficient for spoilers during takeoff abort',
    default_value=0.0,
)

add_meta_data(
    Mission.Takeoff.SPOILER_LIFT_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'TOLIN.CLSPOL',  # '&DEFTOL.TOLIN.CLSPOL',
    },
    units='unitless',
    desc='lift coefficient for spoilers during takeoff abort',
    default_value=0.0,
)

add_meta_data(
    Mission.Takeoff.THRUST_INCIDENCE,
    meta_data=_MetaData,
    historical_name={
        'FLOPS': 'TOLIN.TINC',  # ['&DEFTOL.TOLIN.TINC', 'BALFLD.TINC', '~CLGRAD.TINC'],
        'GASP': None,
    },
    option=True,
    default_value=0.0,
    units='deg',
    desc='thrust incidence on ground',
)

#   _______                  _
#  |__   __|                (_)
#     | |      __ _  __  __  _
#     | |     / _` | \ \/ / | |
#     | |    | (_| |  >  <  | |
#     |_|     \__,_| /_/\_\ |_|
# =============================

add_meta_data(
    Mission.Taxi.DURATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELTT', 'FLOPS': None},
    units='h',
    desc='time spent taxiing before takeoff',
    option=True,
    default_value=0.167,
)

add_meta_data(
    Mission.Taxi.FUEL_TAXI_IN,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Fuel burned to taxi from the runway to the gate. Can be used with energy-stand and 2DOF EOM.',
    option=False,
    default_value=0.0,
)

add_meta_data(
    Mission.Taxi.FUEL_TAXI_OUT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='lbm',
    desc='Fuel burned to taxi from the gate to the runway. Only used in energy-state EOM. Not used in 2DOF EOM.',
    option=False,
    default_value=0.0,
)

add_meta_data(
    Mission.Taxi.MACH,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    desc='speed during taxi, must be nonzero if pycycle is enabled',
    option=True,
    default_value=0.0001,
)

#  .----------------.  .----------------.  .----------------.  .----------------.  .----------------.  .-----------------. .----------------.  .----------------.
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |
# | |    _______   | || |  _________   | || |  _________   | || |  _________   | || |     _____    | || | ____  _____  | || |    ______    | || |    _______   | |
# | |   /  ___  |  | || | |_   ___  |  | || | |  _   _  |  | || | |  _   _  |  | || |    |_   _|   | || ||_   \|_   _| | || |  .' ___  |   | || |   /  ___  |  | |
# | |  |  (__ \_|  | || |   | |_  \_|  | || | |_/ | | \_|  | || | |_/ | | \_|  | || |      | |     | || |  |   \ | |   | || | / .'   \_|   | || |  |  (__ \_|  | |
# | |   '.___`-.   | || |   |  _|  _   | || |     | |      | || |     | |      | || |      | |     | || |  | |\ \| |   | || | | |    ____  | || |   '.___`-.   | |
# | |  |`\____) |  | || |  _| |___/ |  | || |    _| |_     | || |    _| |_     | || |     _| |_    | || | _| |_\   |_  | || | \ `.___]  _| | || |  |`\____) |  | |
# | |  |_______.'  | || | |_________|  | || |   |_____|    | || |   |_____|    | || |    |_____|   | || ||_____|\____| | || |  `._____.'   | || |  |_______.'  | |
# | |              | || |              | || |              | || |              | || |              | || |              | || |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'

add_meta_data(
    Settings.AERODYNAMICS_METHOD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    desc="Sets which legacy code's methods will be used for aerodynamics estimation",
    option=True,
    types=LegacyCode,
    default_value=None,
)

add_meta_data(
    Settings.ATMOSPHERE_MODEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    desc='The atmospheric model used. Chose one of: standard, tropical, polar, hot, cold.',
    option=True,
    types=AtmosphereModel,
    default_value=AtmosphereModel.STANDARD,
)

add_meta_data(
    Settings.EQUATIONS_OF_MOTION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    desc='Sets which equations of motion Aviary will use in mission analysis',
    option=True,
    types=EquationsOfMotion,
    default_value=None,
)

add_meta_data(
    Settings.MASS_METHOD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    desc="Sets which legacy code's methods will be used for mass estimation",
    option=True,
    types=LegacyCode,
    default_value=None,
)

add_meta_data(
    Settings.PAYLOAD_RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='for SIZING missions only. If True, run a set of off-design missions to create a payload range diagram. Assumes SIZING mission describes the max payload + fuel point',
)

add_meta_data(
    Settings.PROBLEM_TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    desc="Select from Aviary's built in problem types: SIZING, OFF_DESIGN_MIN_FUEL, OFF_DESIGN_MAX_RANGE and MULTI_MISSION",
    option=True,
    types=ProblemType,
    default_value=None,
)

add_meta_data(
    Settings.VERBOSITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None},
    desc='Sets how much information Aviary outputs when run. Options include:'
    '0. QUIET: All output except errors are suppressed'
    '1. BRIEF: Only important information is output, in human-readable format'
    '2. VERBOSE: All user-relevant information is output, in human-readable format'
    '3. DEBUG: Any information can be outtputed, including warnings, intermediate calculations, etc., with no formatting requirement',
    option=True,
    types=Verbosity,
    default_value=Verbosity.BRIEF,
)

# here we create a copy of the Aviary-core metadata. The reason for this
# copy is that if we simply imported the Aviary _MetaData in all the
# external subsystem extensions, we would be modifying the original and
# the original _MetaData in the core of Aviary could get altered in
# undesirable ways. By importing this copy to the API the user modifies a
# new MetaData designed just for their purposes.
CoreMetaData = deepcopy(_MetaData)
