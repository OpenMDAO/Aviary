"""
Define meta data associated with variables in the Aviary data hierarchy.
"""

from copy import deepcopy
from pathlib import Path

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
        'LEAPS1': [
            '(WeightABC)self._air_conditioning_group_weight',
            'aircraft.outputs.L0_weights_summary.air_conditioning_group_weight',
        ],
    },
    units='lbm',
    desc='air conditioning system mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.AirConditioning.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(6)', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.air_conditioning_group_weight',
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
        # ['WTS.WSP(24, 2)', '~WEIGHT.WAI', '~WTSTAT.WSP(24, 2)'],
        'FLOPS': None,
        'LEAPS1': [
            '(WeightABC)self._aux_gear_weight',
            'aircraft.outputs.L0_weights_summary.aux_gear_weight',
        ],
    },
    units='lbm',
    desc='mass of anti-icing system (auxiliary gear)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.AntiIcing.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WAI', 'MISWT.WAI', 'MISWT.OAI'],
        'FLOPS': 'WTIN.WAI',
        'LEAPS1': 'aircraft.inputs.L0_overrides.aux_gear_weight',
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
        'LEAPS1': [
            '(WeightABC)self._aux_power_weight',
            'aircraft.outputs.L0_weights_summary.aux_power_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.aux_power_weight',
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
        'LEAPS1': [
            '(WeightABC)self._avionics_group_weight',
            'aircraft.outputs.L0_weights_summary.avionics_group_weight',
        ],
    },
    units='lbm',
    desc='avionics mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Avionics.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WAVONC', 'MISWT.WAVONC', 'MISWT.OAVONC'],
        'FLOPS': 'WTIN.WAVONC',
        'LEAPS1': 'aircraft.inputs.L0_overrides.avionics_group_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_battery.weight_offset',
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
        'LEAPS1': 'aircraft.inputs.L0_battery.depth_of_discharge',
    },
    units='unitless',
    desc='default constraint on how far the battery can discharge, as a proportion of '
    'total energy capacity',
    default_value=0.2,
)

add_meta_data(
    Aircraft.Battery.EFFICIENCY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.EFF_BAT', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    default_value=1.0,
    desc='battery pack efficiency',
)

add_meta_data(
    Aircraft.Battery.ENERGY_CAPACITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'EBATTAVL', 'FLOPS': None, 'LEAPS1': None},
    units='kJ',
    desc='total energy the battery can store',
)

add_meta_data(
    Aircraft.Battery.MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.WBATTIN',
        'FLOPS': None,
        'LEAPS1': 'aircraft.inputs.L0_battery.weight',
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
        'LEAPS1': 'aircraft.inputs.L0_battery.energy_density',
    },
    units='kW*h/kg',
    desc='specific energy density of the battery pack',
    default_value=1.0,
)


add_meta_data(
    Aircraft.Battery.PACK_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of the energy-storing components of the battery',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Battery.PACK_VOLUMETRIC_DENSITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='kW*h/L',
    desc='volumetric density of the battery pack',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Battery.VOLUME,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Flag if the detailed wing model is provided',
    option=True,
    types=bool,
    default_value=True,
)

add_meta_data(
    Aircraft.BWB.MAX_NUM_BAYS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'FUSEIN.NBAYMX',  # ['&DEFINE.FUSEIN.NBAYMX', 'FUSDTA.NBAYMX'],
        'LEAPS1': None,
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
        'LEAPS1': [
            'aircraft.inputs.L0_blended_wing_body_design.bay_count',
            'aircraft.cached.L0_blended_wing_body_design.bay_count',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_blended_wing_body_design.passenger_leading_edge_sweep',
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
        'LEAPS1': 'aircraft.inputs.L0_canard.area',
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
        'LEAPS1': 'aircraft.inputs.L0_canard.aspect_ratio',
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table[-1]',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table[-1]',
        ],
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table[-1]',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table[-1]',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.canard_percent_laminar_flow_lower_surface',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.canard_percent_laminar_flow_upper_surface',
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
        'LEAPS1': [
            '(WeightABC)self._canard_weight',
            'aircraft.outputs.L0_weights_summary.canard_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.canard_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_canard.taper_ratio',
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
        'LEAPS1': 'aircraft.inputs.L0_canard.thickness_to_chord_ratio',
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.canard_wetted_area',
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table[-1]',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table[-1]',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.canard_wetted_area',
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
    historical_name={'GASP': 'INGASP.CK15', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='technology factor on cockpit controls mass',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Controls.CONTROL_MASS_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELWFC', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='incremental flight controls mass',
    default_value=0,
)

add_meta_data(
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKSAS', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of stability augmentation system',
    default_value=0,
)

add_meta_data(
    Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK19', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='technology factor on stability augmentation system mass',
    default_value=1,
)

add_meta_data(
    Aircraft.Controls.TOTAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFC', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of cockpit controls, fixed wing controls, and SAS',
    default_value=0.0,
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
        'LEAPS1': [
            '(WeightABC)self._passenger_bag_weight',
            'aircraft.outputs.L0_weights_summary.passenger_bag_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.baggage_weight_per_passenger',
    },
    units='lbm',
    desc='baggage mass per passenger',
    option=True,
    default_value=0.0,
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
        'LEAPS1': [
            '(WeightABC)self._cargo_containers_weight',
            'aircraft.outputs.L0_weights_summary.cargo_containers_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.cargo_containers_weight',
    },
    units='unitless',
    desc='Scaler for mass of cargo containers',
    default_value=1.0,
)

add_meta_data(
    Aircraft.CrewPayload.CARGO_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of as-flown cargo',
)

add_meta_data(
    Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(12)', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of catering items per passenger',
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._cargo_weight',
            'aircraft.outputs.L0_weights_summary.cargo_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.business_class_count',
    },
    units='unitless',
    desc='number of business class passengers that the aircraft is designed to accommodate',
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
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.first_class_count',
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
        'LEAPS1': 'aircraft.outputs.L0_crew_and_payload.passenger_count',
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
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.NBABR', 'LEAPS1': None},
    units='unitless',
    desc='Number of business class passengers abreast',
    types=int,
    option=True,
    default_value=5,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.NFABR', 'LEAPS1': None},
    units='unitless',
    desc='Number of first class passengers abreast',
    types=int,
    option=True,
    default_value=4,
)

add_meta_data(
    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SAB', 'FLOPS': 'FUSEIN.NTABR', 'LEAPS1': None},
    units='unitless',
    desc='Number of tourist class passengers abreast',
    types=int,
    option=True,
    default_value=6,
)

# TODO rename to economy?
add_meta_data(
    Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NPT',  # ['&DEFINE.WTIN.NPT', 'WTS.NPT'],
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.tourist_class_count',
    },
    units='unitless',
    desc='number of tourist class passengers that the aircraft is designed to accommodate',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.BPITCH', 'LEAPS1': None},
    units='inch',
    desc='pitch of the business class seats',
    option=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'FUSEIN.FPITCH', 'LEAPS1': None},
    units='inch',
    desc='pitch of the first class seats',
    option=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.PS', 'FLOPS': 'FUSEIN.TPITCH', 'LEAPS1': None},
    units='inch',
    desc='pitch of the tourist class seats',
    option=True,
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
        'LEAPS1': [
            '(WeightABC)self._flight_crew_and_bag_weight',
            'aircraft.outputs.L0_weights_summary.flight_crew_and_bag_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.flight_crew_and_bag_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.weight_per_passenger',
    },
    units='lbm',
    desc='mass per passenger',
    option=True,
    default_value=165.0,
)

add_meta_data(
    Aircraft.CrewPayload.MISC_CARGO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.CARGOF',  # ['&DEFINE.WTIN.CARGOF', 'WTS.CARGOF'],
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.misc_cargo',
    },
    units='lbm',
    desc='cargo (other than passenger baggage) carried in fuselage',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER
    Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(28,2)', '~WEIGHT.WSTUAB', '~WTSTAT.WSP(28, 2)', '~INERT.WSTUAB'],
        'FLOPS': None,
        'LEAPS1': [
            '(WeightABC)self._cabin_crew_and_bag_weight',
            'aircraft.outputs.L0_weights_summary.cabin_crew_and_bag_weight',
        ],
    },
    units='lbm',
    desc='total mass of the non-flight crew and their baggage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WSTUAB', 'MISWT.WSTUAB', 'MISWT.OSTUAB'],
        'FLOPS': 'WTIN.WSTUAB',
        'LEAPS1': 'aircraft.inputs.L0_overrides.cabin_crew_and_bag_weight',
    },
    units='unitless',
    desc='scaler for total mass of the non-flight crew and their baggage',
    default_value=1.0,
)

add_meta_data(
    Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['&DEFINE.WTIN.NPB', 'WTS.NPB'],
        'LEAPS1': None,  # 'aircraft.inputs.L0_crew_and_payload.business_class_count',
    },
    units='unitless',
    desc='number of business class passengers',
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
        'LEAPS1': None,  # 'aircraft.inputs.L0_crew_and_payload.first_class_count',
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
        'LEAPS1': [
            'aircraft.inputs.L0_crew_and_payload.flight_attendants_count',
            'aircraft.cached.L0_crew_and_payload.flight_attendants_count',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_crew_and_payload.flight_crew_count',
            'aircraft.cached.L0_crew_and_payload.flight_crew_count',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_crew_and_payload.galley_crew_count',
            'aircraft.cached.L0_crew_and_payload.galley_crew_count',
        ],
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
        'LEAPS1': None,  # 'aircraft.outputs.L0_crew_and_payload.passenger_count',
    },
    units='unitless',
    desc='total number of passengers',
    option=True,
    default_value=0,
    types=int,
)

# TODO rename to economy?
add_meta_data(
    Aircraft.CrewPayload.NUM_TOURIST_CLASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['&DEFINE.WTIN.NPT', 'WTS.NPT'],
        'LEAPS1': None,  # 'aircraft.inputs.L0_crew_and_payload.tourist_class_count',
    },
    units='unitless',
    desc='number of tourist class passengers',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(34, 2)', '~WEIGHT.WPASS', '~WTSTAT.WSP(34, 2)', '~INERT.WPASS'],
        'FLOPS': None,
        'LEAPS1': [
            '(WeightABC)self._passenger_weight',
            'aircraft.outputs.L0_weights_summary.passenger_weight',
        ],
    },
    units='lbm',
    desc='TBD: total mass of all passengers without their baggage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.UWPAX', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of one passenger and their bags',
    option=True,
    default_value=200,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
    meta_data=_MetaData,
    # note: this GASP variable does not include cargo, but it does include
    # passenger baggage
    historical_name={'GASP': 'INGASP.WPL', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._passenger_service_weight',
            'aircraft.outputs.L0_weights_summary.passenger_service_weight',
        ],
    },
    units='lbm',
    desc='mass of passenger service equipment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(9)', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.passenger_service_weight',
    },
    units='unitless',
    desc='scaler for mass of passenger service equipment',
    default_value=1.0,
)

add_meta_data(
    Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of payload, including passengers, passenger baggage, and cargo',
)

add_meta_data(
    Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(14)', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='unit mass of ULD (unit load device) for cargo handling per passenger',
    default_value=0.0,
    types=float,
    option=True,
)

add_meta_data(
    Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(10)', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_crew_and_payload.wing_cargo',
    },
    units='lbm',
    desc='cargo carried in wing',
    default_value=0.0,
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
        'LEAPS1': [
            'aircraft.inputs.L0_aerodynamics.base_area',
            'aircraft.outputs.L0_aerodynamics.mission_base_area',
        ],
    },
    units='ft**2',
    desc='Aircraft base area (total exit cross-section area minus inlet '
    'capture areas for internally mounted engines)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.CG_DELTA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELCG', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table',
        ],
    },
    units='ft',
    desc='Reynolds characteristic length for each component',
)

add_meta_data(
    Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKCC', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='mass trend coefficient of cockpit controls',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='if true, use empirical tail volume coefficient equation. This is '
    'true if VBARVX is 0 in GASP.',
)

add_meta_data(
    Aircraft.Design.DRAG_COEFFICIENT_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELCD', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='increment to the profile drag coefficient',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.DRAG_DIVERGENCE_SHIFT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SCFAC', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='shift in drag divergence Mach number due to supercritical design',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.DRAG_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Drag polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.EMERGENCY_EQUIPMENT_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(11)', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of emergency equipment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.EMPTY_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFMSS.MISSIN.DOWE', '&FLOPS.RERUN.DOWE', 'ESB.DOWE'],
        'FLOPS': None,
        'LEAPS1': None,
    },
    units='lbm',
    desc='empty mass of the aircraft',
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
        'LEAPS1': '(WeightABC)self._weight_empty_margin',
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.weight_empty_margin',
    },
    units='unitless',
    desc='empty mass margin scaler',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
    historical_name={
        'GASP': None,
        'FLOPS': None,
        'LEAPS1': None,
    },
    meta_data=_MetaData,
    units='lbm',
    desc='total mass of all user-defined external subsystems',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR',
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table',
        ],
    },
    units='unitless',
    desc='table of component fineness ratios',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.FIXED_EQUIPMENT_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFE', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of fixed equipment: APU, Instruments, Hydraulics, Electrical, '
    'Avionics, AC, Anti-Icing, Auxiliary Equipment, and Furnishings',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.FIXED_USEFUL_LOAD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFUL', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of fixed useful load: crew, service items, trapped oil, etc',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.IJEFF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.IJEFF', 'FLOPS': None, 'LEAPS1': None},
    desc='A flag used by Jeff V. Bowles to debug GASP code during his 53 years supporting the development of GASP. '
    "This flag is planted here to thank him for his hard work and dedication, Aviary wouldn't be what it is today "
    'without his help.',
)

# TODO expected types and default value?
add_meta_data(
    Aircraft.Design.LAMINAR_FLOW_LOWER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.TRL',
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.mission_component_percent_laminar_flow_lower_surface_table',
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
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.mission_component_percent_laminar_flow_upper_surface_table',
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
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.landing_to_takeoff_weight_ratio',
    },
    units='unitless',
    desc='ratio of maximum landing mass to maximum takeoff mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.LIFT_CURVE_SLOPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLALPH', 'FLOPS': None, 'LEAPS1': None},
    units='1/rad',
    desc='lift curve slope at cruise Mach number',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'MISSIN.FCDI',  # '~DRGFCT.FCDI',
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.induced_drag_coeff_fact',
    },
    units='unitless',
    default_value=1.0,
    desc='Scaling factor for lift-dependent drag coefficient',
)

add_meta_data(
    Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Lift dependent drag polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Lift independent drag polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.LIFT_POLAR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Lift polar computed during Aviary pre-mission.',
    multivalue=True,
    types=float,
)

add_meta_data(
    Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.THEMAX', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='maximum fuselage pitch allowed',
    default_value=15,
)

add_meta_data(
    Aircraft.Design.MAX_STRUCTURAL_SPEED,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VMLFSL', 'FLOPS': None, 'LEAPS1': None},
    units='mi/h',
    desc='maximum structural design flight speed in miles per hour',
    default_value=0,
)

add_meta_data(
    Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CATD', 'FLOPS': None, 'LEAPS1': None},
    option=True,
    default_value=3,
    types=int,
    units='unitless',
    desc='part 25 structural category',
)

add_meta_data(
    Aircraft.Design.RESERVE_FUEL_ADDITIONAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FRESF', 'FLOPS': None, 'LEAPS1': None},
    option=True,
    units='lbm',
    desc='required fuel reserves: directly in lbm',
    default_value=0,
)

add_meta_data(
    Aircraft.Design.RESERVE_FUEL_FRACTION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    option=True,
    units='unitless',
    desc='required fuel reserves: given as a proportion of mission fuel. This value must be nonnegative. '
    'Mission fuel only includes normal phases and excludes reserve phases. '
    'If it is 0.5, the reserve fuel is half of the mission fuel (one third of the total fuel). Note '
    'it can be greater than 1. If it is 2, there would be twice as much reserve fuel as mission fuel '
    '(the total fuel carried would be 1/3 for the mission and 2/3 for the reserve)',
    default_value=0,
)

add_meta_data(
    Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    option=True,
    default_value=False,
    types=bool,
    units='unitless',
    desc='eliminates discontinuities in GASP-based mass estimation code if true',
)

add_meta_data(
    Aircraft.Design.STATIC_MARGIN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STATIC', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='aircraft static margin as a fraction of mean aerodynamic chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.STRUCTURAL_MASS_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELWST', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._total_structural_weight',
            'aircraft.outputs.L0_weights_summary.total_structural_weight',
        ],
    },
    units='lbm',
    desc='Total structural group mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'MISSIN.FCDSUB',  # '~DRGFCT.FCDSUB',
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.sub_drag_coeff_fact',
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
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.sup_drag_coeff_fact',
    },
    units='unitless',
    default_value=1.0,
    desc='Scaling factor for supersonic drag',
)

add_meta_data(
    Aircraft.Design.SYSTEMS_EQUIP_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(25, 2)', '~WEIGHT.WSYS', '~WTSTAT.WSP(25, 2)'],
        'FLOPS': None,
        'LEAPS1': [
            '(WeightABC)self._equipment_group_weight',
            'aircraft.outputs.L0_weights_summary.equipment_group_weight',
        ],
    },
    units='lbm',
    desc='Total systems & equipment group mass',
    default_value=0.0,
)

# TODO intermediate calculated values with no uses by other systems may not belong in the
#      variable hierarchy
add_meta_data(
    # Note in FLOPS/LEAPS1, this is the same variable as
    # Aircraft.Design.SYSTEMS_EQUIP_MASS, because FLOPS/LEAPS1 overwrite the
    # value during calculations; in Aviary, these must be separate variables
    Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='Total systems & equipment group mass without additional 1% of empty mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.THRUST_TO_WEIGHT_RATIO,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
        'LEAPS1': None,
        # NOTE TWR != THRUST_TO_WEIGHT_RATIO because Aviary\'s value is the actual T/W, while TWR is
        #      the desired T/W ratio
        # 'FLOPS': 'CONFIN.TWR',
        # 'LEAPS1': 'ipropulsion.req_thrust_weight_ratio',
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
        'LEAPS1': '~WeightABC._update_cycle.total_wetted_area',
    },
    units='ft**2',
    desc='total aircraft wetted area',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override (no scaling)
    Aircraft.Design.TOUCHDOWN_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.WLDG',
        #  [  # inputs
        #      '&DEFINE.WTIN.WLDG', 'WTS.WLDG',
        #      # outputs
        #      'CMODLW.WLDGO',
        #  ],
        'LEAPS1': [
            'aircraft.inputs.L0_landing_gear.design_landing_weight',
            'aircraft.outputs.L0_landing_gear.design_landing_weight',
        ],
    },
    units='lbm',
    desc='design landing mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Design.TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': ['INGASP.IHWB'], 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    types=AircraftTypes,
    option=True,
    default_value=AircraftTypes.TRANSPORT,
    desc='aircraft type: BWB for blended wing body, transport otherwise',
)

add_meta_data(
    Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER,
    meta_data=_MetaData,
    historical_name={'GASP': 'CATD', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_weights.use_alt_weights',
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table',
        ],
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
        'LEAPS1': None,
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
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.geom_drag_coeff_fact',
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='if true there is an augmented electrical system',
)

add_meta_data(
    Aircraft.Electrical.HYBRID_CABLE_LENGTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.LCABLE', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._electrical_group_weight',
            'aircraft.outputs.L0_weights_summary.electrical_group_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.electrical_group_weight',
    },
    units='unitless',
    desc='mass scaler for the electrical system',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(15)', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='additional propulsion system mass added to engine control and starter mass, or '
    'engine installation mass',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.ADDITIONAL_MASS_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SKPEI',
        'FLOPS': 'WTIN.WPMISC',  # ['&DEFINE.WTIN.WPMISC', 'FAWT.WPMISC'],
        'LEAPS1': 'aircraft.inputs.L0_propulsion.misc_weight',
    },
    units='unitless',
    option=True,
    desc='fraction of (scaled) engine mass used to calculate additional propulsion '
    'system mass added to engine control and starter mass, or used to calculate engine '
    'installation mass',
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
        'LEAPS1': ['iengine.fuel_leak', 'aircraft.inputs.L0_engine.fuel_leak'],
    },
    option=True,
    units='lbm/h',
    desc='Additional constant fuel flow. This value is not scaled with the engine',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.CONTROLS_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WEIGHT.WEC',
        'LEAPS1': '(WeightABC)self._engine_ctrl_weight',
    },
    units='lbm',
    desc='estimated mass of the engine controls',
    default_value=0.0,
    multivalue=True,
)

# TODO there should be a GASP name that pairs here
add_meta_data(
    Aircraft.Engine.DATA_FILE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'ENGDIN.EIFILE', 'LEAPS1': None},
    units='unitless',
    types=(Path, str),
    default_value=None,
    option=True,
    desc='filepath to data file containing engine performance tables',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FIXED_RPM,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='rpm',
    default_value=1.0,
    desc='RPM the engine is set to be running at. Overrides RPM provided by '
    'engine model or chosen by optimizer. Typically used when pairing a motor or '
    'turboshaft using a fixed operating RPM with a propeller.',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.FIDMAX',
        'LEAPS1': 'aircraft.L0_fuel_flow.idle_max_fract',
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
        'LEAPS1': 'aircraft.L0_fuel_flow.idle_min_fract',
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'ifuel_flow.scaling_const_term',
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
        'LEAPS1': 'ifuel_flow.scaling_linear_term',
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
        'LEAPS1': 'engine_model.imodel_info.flight_idle_index',
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
        'LEAPS1': 'imodel_info.geopotential_alt',
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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

# TODO dependency on NTYE? Does this var need preprocessing? Can this mention be removed?
add_meta_data(
    Aircraft.Engine.HAS_PROPELLERS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    option=True,
    units='unitless',
    default_value=False,
    types=bool,
    desc='if True, the aircraft has propellers, otherwise aircraft is assumed to have no '
    'propellers. In GASP this depended on NTYE',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.IGNORE_NEGATIVE_THRUST,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.NONEG',
        'LEAPS1': 'imodel_info.ignore_negative_thrust',
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
    Aircraft.Engine.INTERPOLATION_METHOD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    default_value='slinear',
    types=str,
    desc="method used for interpolation on an engine deck's data file, allowable values are "
    'table methods from openmdao.components.interp_util.interp',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.INTERPOLATION_SORT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.outputs.L0_weights_summary.Engine.WEIGHT',
    },
    units='lbm',
    desc='scaled mass of a single engine or bare engine if inlet and nozzle mass are supplied',
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
        'LEAPS1': 'aircraft.inputs.L0_propulsion.engine_weight_scale',
    },
    units='unitless',
    desc='scaler for engine mass',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.MASS_SPECIFIC,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SWSLS', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.outputs.L0_propulsion.total_engine_count',
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
        'LEAPS1': 'aircraft.inputs.L0_fuselage.engines_count',
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
        'LEAPS1': 'aircraft.inputs.L0_wing.engines_count',
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
        'LEAPS1': '(WeightABC)self._engine_pod_weight_list',
    },
    units='lbm',
    desc='engine pod mass including nacelles',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.POD_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK14', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='technology factor on mass of engine pods',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.POSITION_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKEPOS', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='engine position factor',
    default_value=0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.PYLON_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FPYL', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='factor for turbofan engine pylon mass',
    default_value=0.7,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.REFERENCE_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.DIAM_REF',
        'FLOPS': None,
        'LEAPS1': None,
    },  # no DIAM_REF in GASP
    units='ft',
    desc='engine reference diameter',
    default_value=0.0,
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
        'LEAPS1': '(WeightABC)self._Engine.WEIGHT',
    },
    units='lbm',
    desc='unscaled mass of a single engine or bare engine if inlet and nozzle mass are supplied',
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
        'LEAPS1': 'aircraft.inputs.L0_engine*.thrust',
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
        'LEAPS1': None,
    },
    units='rpm',
    desc='the designed output RPM from the engine for fixed-RPM shafts',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.SCALE_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Thrust-based scaling factor used to scale engine performance data during '
    'mission analysis',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.SCALE_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
        'LEAPS1': '(types)EngineScaleModes.WEIGHT',
    },
    desc='Toggle for enabling scaling of engine mass based on Aircraft.Engine.SCALE_FACTOR',
    option=True,
    types=bool,
    multivalue=True,
    default_value=True,
)

add_meta_data(
    Aircraft.Engine.SCALE_PERFORMANCE,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,
        'LEAPS1': [
            'iengine.scale_mode',
            '(types)EngineScaleModes.DEFAULT',
        ],
    },
    desc='Toggle for enabling scaling of engine performance including thrust, fuel flow, '
    'and electric power using Aircraft.Engine.SCALE_FACTOR',
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
        'LEAPS1': [
            'aircraft.outputs.L0_propulsion.max_rated_thrust',
            'aircraft.cached.L0_propulsion.max_rated_thrust',
        ],
    },
    units='lbf',
    desc='Maximum sea-level static thrust of an engine after scaling. Optional for '
    'EngineDecks if Aircraft.Engine.SCALE_FACTOR is provided, in which case this '
    'variable is computed.',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.STARTER_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WEIGHT.WSTART',
        'LEAPS1': '(WeightABC)self._starter_weight',
    },
    units='lbm',
    desc='mass of engine starter subsystem',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'ENGDIN.FFFSUB',
        'LEAPS1': 'aircraft.L0_fuel_flow.subsonic_factor',
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
        'GASP': None,
        'FLOPS': 'ENGDIN.FFFSUP',
        'LEAPS1': 'aircraft.L0_fuel_flow.supersonic_factor',
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
        'LEAPS1': [
            '(WeightABC)self._thrust_reversers_weight',
            'aircraft.outputs.L0_weights_summary.thrust_reversers_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.thrust_reversers_weight',
    },
    units='unitless',
    desc='scaler for mass of thrust reversers on engines. In FLOPS/LEAPS1 default to 0.0',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.NTYE', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_propulsion.wing_engine_locations',
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='The efficiency of the gearbox.',
    default_value=1.0,
    multivalue=True,
)
add_meta_data(
    Aircraft.Engine.Gearbox.GEAR_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},  # 1 / INPROP.GR
    units='unitless',
    desc='Reduction gear ratio, or the ratio of the RPM_in divided by the RPM_out for the gearbox.',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Gearbox.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': None,
    },
    units='hp',
    desc='A guess for the maximum power that will be transmitted through the gearbox during the mission (max shp input).',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Gearbox.SPECIFIC_TORQUE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    Aircraft.Engine.Motor.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'WMOTOR', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='Total motor mass (considers number of motors)',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Motor.TORQUE_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'INPROP.AF', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'INPROP.FT', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    types=(str, Path),
    default_value=None,
    option=True,
    desc='filepath to data file containing propeller data map',
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.DIAMETER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.DPROP', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='propeller diameter',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.CLI', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='propeller blade integrated design lift coefficient (Range: 0.3 to 0.8)',
    default_value=0.5,
    multivalue=True,
)

add_meta_data(
    Aircraft.Engine.Propeller.NUM_BLADES,
    meta_data=_MetaData,
    historical_name={'GASP': 'INPROP.BL', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': None,
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
        'LEAPS1': None,
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
        'LEAPS1': 'aircraft.inputs.L0_fins.area',
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
        'LEAPS1': [
            '(WeightABC)self._wing_vertical_fin_weight',
            'aircraft.outputs.L0_weights_summary.wing_vertical_fin_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.wing_vertical_fin_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_fins.fin_count',
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
        'LEAPS1': 'aircraft.inputs.L0_fins.taper_ratio',
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
        'LEAPS1': 'aircraft.inputs.L0_fuel.aux_capacity',
    },
    units='lbm',
    desc='fuel capacity of the auxiliary tank',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.BURN_PER_PASSENGER_MILE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/NM',
    desc='average fuel burn per passenger per mile flown',
)

add_meta_data(
    Aircraft.Fuel.DENSITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FUELD', 'FLOPS': 'WTIN.FULDEN', 'LEAPS1': None},
    units='lbm/galUS',
    desc='fuel density (jet fuel typical density of 6.7 lbm/galUS used in the calculation of wing_capacity'
    '(if wing_capacity is not input) and in the calculation of fuel system weight.',
    default_value=6.7,
)

add_meta_data(
    Aircraft.Fuel.FUEL_MARGIN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOL_MRG', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='percentage of excess fuel volume required, essentially the amount of fuel above '
    'the design point that there has to be volume to carry',
    default_value=0.0,
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
        'LEAPS1': [
            '(WeightABC)self._fuel_sys_weight',
            'aircraft.outputs.L0_weights_summary.fuel_sys_weight',
        ],
    },
    units='lbm',
    desc='fuel system mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKFS', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.fuel_sys_weight',
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
        'LEAPS1': [
            'aircraft.inputs.L0_fuel.fuselage_capacity',
            '(WeightABC)self._fuselage_fuel_capacity',
        ],
    },
    units='lbm',
    desc='fuel capacity of the fuselage',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'WTIN.IFUFU', 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_fuel.tank_count',
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
        'LEAPS1': [
            'aircraft.inputs.L0_fuel.total_capacity',
            'aircraft.cached.L0_fuel.total_capacity',
        ],
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
        'LEAPS1': [
            '(WeightABC)self._total_fuel_vol',
            '~WeightABC.calc_unusable_fuel.total_fuel_vol',
            '~WeightABC._pre_unusable_fuel.total_fuel_vol',
            '~BasicTransportWeight._pre_unusable_fuel.total_fuel_vol',
        ],
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
        'LEAPS1': [
            '(WeightABC)self._unusable_fuel_weight',
            'aircraft.outputs.L0_weights_summary.unusable_fuel_weight',
        ],
    },
    units='lbm',
    desc='unusable fuel mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(13)', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.unusable_fuel_weight',
    },
    units='unitless',
    desc='scaler for Unusable fuel mass',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Fuel.WING_FUEL_CAPACITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FULWMX',  # ['&DEFINE.WTIN.FULWMX', 'WTS.FULWMX'],
        'LEAPS1': 'aircraft.inputs.L0_fuel.wing_capacity',
    },
    units='lbm',
    desc='fuel capacity of the auxiliary tank',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_FUEL_FRACTION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKWF', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_fuel.wing_ref_capacity',
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
        'LEAPS1': 'aircraft.inputs.L0_fuel.wing_ref_capacity_area',
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
        'LEAPS1': 'aircraft.inputs.L0_fuel.wing_ref_capacity_1_5_term',
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
        'LEAPS1': 'aircraft.inputs.L0_fuel.wing_ref_capacity_linear_term',
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
#                      "LEAPS1": None
#                     },
#     FLOPS_name=None,
#     LEAPS1_name=None,
#     GASP_name='INGASP.FVOLW',
#     units='ft**3',
#     desc='wing tank fuel volume',
# )

add_meta_data(
    Aircraft.Fuel.WING_VOLUME_DESIGN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOLREQ', 'FLOPS': None, 'LEAPS1': None},
    units='ft**3',
    desc='wing tank fuel volume when carrying design fuel plus fuel margin',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOLW_GEOM', 'FLOPS': None, 'LEAPS1': None},
    units='ft**3',
    desc='wing tank fuel volume based on geometry',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FVOLW_MAX', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._furnishings_group_weight',
            'aircraft.outputs.L0_weights_summary.furnishings_group_weight',
        ],
    },
    units='lbm',
    desc='Total furnishings system mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Furnishings.MASS_BASE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='For FLOPS based, base furnishings system mass without additional 1% empty mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Furnishings.MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WFURN', 'MISWT.WFURN', 'MISWT.OFURN'],
        'FLOPS': 'WTIN.WFURN',
        'LEAPS1': 'aircraft.inputs.L0_overrides.furnishings_group_weight',
    },
    units='unitless',
    desc='Furnishings system mass scaler. In GASP based, it is applicale if gross mass '
    '> 10000 lbs and number of passengers >= 50. Set it to 0.0 if not use.',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Furnishings.USE_EMPIRICAL_EQUATION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'WGT_AB', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    default_value=0.0,
    desc='aftbody mass',
)

add_meta_data(
    Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.UWT_AFT', 'FLOPS': None, 'LEAPS1': None},
    units='lbm/ft**2',
    default_value=0.0,
    desc='aftbody structural areal unit weight',
)

add_meta_data(
    Aircraft.Fuselage.AISLE_WIDTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WAS', 'FLOPS': None, 'LEAPS1': None},
    units='inch',
    desc='width of the aisles in the passenger cabin',
    option=True,
    default_value=24,
)

# TODO FLOPS is not average diameter, but rather a reference diameter using max
#      height and length. New variable??
add_meta_data(
    Aircraft.Fuselage.AVG_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.WC', 'INGASP.SWF'],
        'FLOPS': None,  # 'EDETIN.XD',
        'LEAPS1': 'aircraft.outputs.L0_fuselage.avg_diam',
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
        'LEAPS1': [
            'aircraft.inputs.L0_blended_wing_body_design.cabin_area',
            'aircraft.cached.L0_blended_wing_body_design.cabin_area',
        ],
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table[3]',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table[3]',
        ],
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
        'LEAPS1': 'aircraft.outputs.L0_fuselage.mission_cross_sect_area',
    },
    units='ft**2',
    desc='fuselage cross sectional area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.DELTA_DIAMETER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HCK', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.outputs.L0_fuselage.mission_diam_to_wing_span_ratio',
    },
    units='unitless',
    desc='fuselage diameter to wing span ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[4]',
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table[3]',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table[3]',
        ],
    },
    units='unitless',
    desc='fuselage fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELFE', 'FLOPS': None, 'LEAPS1': None},
    units='ft**2',
    desc='increment to fuselage flat plate area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.FOREBODY_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'WGT_FB', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    default_value=0.0,
    desc='forebody mass',
)

add_meta_data(
    Aircraft.Fuselage.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKF', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='fuselage form factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HGTqWID', 'FLOPS': 'WTIN.TCF', 'LEAPS1': None},
    units='unitless',
    types=float,
    default_value=1.0,
    desc='fuselage height-to-width ratio',
)

add_meta_data(
    Aircraft.Fuselage.HYDRAULIC_DIAMETER,
    meta_data=_MetaData,
    historical_name={'GASP': 'DHYDRAL', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.fuselage_percent_laminar_flow_lower_surface',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.fuselage_percent_laminar_flow_upper_surface',
    },
    units='unitless',
    desc='define percent laminar flow for fuselage upper surface',
    default_value=0.0,
)

# TODO LEAPS variable description
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
        'LEAPS1': [
            'aircraft.inputs.L0_fuselage.total_length',
            'aircraft.outputs.L0_fuselage.total_length',
            # other
            'aircraft.cached.L0_fuselage.total_length',
        ],
    },
    units='ft',
    desc='Define the Fuselage total length. If total_length is not input for a '
    'passenger transport, LEAPS will calculate the fuselage length, width and '
    'depth and the length of the passenger compartment.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.LENGTH_TO_DIAMETER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['MISSA.BODYLD', '~CDCC.BODYLD'],
        'LEAPS1': 'aircraft.outputs.L0_fuselage.mission_len_to_diam_ratio',
    },
    units='unitless',
    desc='fuselage length to diameter ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.LIFT_COEFFICIENT_RATIO_BODY_TO_WING,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLBqCLW', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    types=float,
    default_value=0.0,
    desc='lift coefficient of body over lift coefficient of wing ratio',
)

add_meta_data(
    Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLALPH_B0', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._fuselage_weight',
            'aircraft.outputs.L0_weights_summary.fuselage_weight',
        ],
    },
    units='lbm',
    desc='mass of the fuselage structure',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKB', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.fuselage_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_fuselage.max_height',
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
        'LEAPS1': [
            'aircraft.inputs.L0_fuselage.max_width',
            'aircraft.outputs.L0_fuselage.max_width',
            # other
            'aircraft.cached.L0_fuselage.max_width',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_crew_and_payload.military_cargo',
            'aircraft.cached.L0_crew_and_payload.military_cargo',
        ],
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
    historical_name={'GASP': 'INGASP.ELODN', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='length to diameter ratio of nose cone',
    default_value=1,
)

add_meta_data(
    Aircraft.Fuselage.NUM_AISLES,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.AS', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_fuselage.count',
            # other
            'aircraft.cached.L0_fuselage.count',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_fuselage.passenger_compartment_length',
            'aircraft.cached.L0_fuselage.passenger_compartment_length',
        ],
    },
    units='ft',
    desc='length of passenger compartment',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELPC', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': '(WeightABC)self._fuselage_planform_area',
    },
    units='ft**2',
    desc='fuselage planform area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PRESSURE_DIFFERENTIAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELP', 'FLOPS': None, 'LEAPS1': None},
    units='psi',
    desc='fuselage pressure differential during cruise',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WPRFUS', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    default_value=0.0,
    desc='additional pressurized fuselage width for cargo bay',
)

add_meta_data(
    Aircraft.Fuselage.SEAT_WIDTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WS', 'FLOPS': None, 'LEAPS1': None},
    units='inch',
    desc='width of the economy class seats',
    option=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.SIMPLE_LAYOUT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='carry out simple or detailed layout of fuselage.',
    option=True,
    default_value=True,
)

add_meta_data(
    Aircraft.Fuselage.TAIL_FINENESS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELODT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.fuselage_wetted_area',
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table[3]',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table[3]',
        ],
    },
    units='ft**2',
    desc='fuselage wetted area',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SAFTqS', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.fuselage_wetted_area',
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
        'LEAPS1': [
            'aircraft.inputs.L0_horizontal_tail.area',
            'aircraft.cached.L0_horizontal_tail.area',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_horizontal_tail.aspect_ratio',
    },
    units='unitless',
    desc='horizontal tail theoretical aspect ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.AVERAGE_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CBARHT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table[1]',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table[1]',
        ],
    },
    units='ft',
    desc='Reynolds characteristic length for the horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[2]',
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table[1]',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table[1]',
        ],
    },
    units='unitless',
    desc='horizontal tail fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKHT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.horizontal_tail_percent_laminar_flow_lower_surface',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.horizontal_tail_percent_laminar_flow_upper_surface',
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
        'LEAPS1': [
            '(WeightABC)self._horizontal_tail_weight',
            'aircraft.outputs.L0_weights_summary.horizontal_tail_weight',
        ],
    },
    units='lbm',
    desc='mass of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKY', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.horizontal_tail_weight',
    },
    units='unitless',
    desc='mass scaler of the horizontal tail structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MOMENT_ARM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELTH', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='moment arm of horizontal tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.MOMENT_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.COELTH', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Ratio of wing chord to horizontal tail moment arm',
)

add_meta_data(
    Aircraft.HorizontalTail.ROOT_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRCLHT', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='horizontal tail root chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.HorizontalTail.SPAN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BHT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_horizontal_tail.sweep_at_quarter_chord',
            'aircraft.cached.L0_horizontal_tail.sweep_at_quarter_chord',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_horizontal_tail.taper_ratio',
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
        'LEAPS1': 'aircraft.inputs.L0_horizontal_tail.thickness_to_chord_ratio',
    },
    units='unitless',
    desc='horizontal tail thickness-chord ratio',
    default_value=0.0,
)

# TODO preprocessing for this variable on FLOPS side
add_meta_data(
    Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.SAH',
        'FLOPS': 'WTIN.HHT',  # ['&DEFINE.WTIN.HHT', 'EDETIN.HHT'],
        'LEAPS1': 'aircraft.inputs.L0_horizontal_tail.vertical_tail_fraction',
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
    historical_name={'GASP': 'INGASP.VBARHX', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.horizontal_tail_wetted_area',
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table[1]',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table[1]',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.horizontal_tail_wetted_area',
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
    historical_name={'GASP': 'INGASP.CW(3)', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='mass trend coefficient of hydraulics for flight control system',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(4)', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._hydraulics_group_weight',
            'aircraft.outputs.L0_weights_summary.hydraulics_group_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.hydraulics_group_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_weights.hydraulic_sys_press',
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
        'LEAPS1': [
            '(WeightABC)self._instrument_group_weight',
            'aircraft.outputs.L0_weights_summary.instrument_group_weight',
        ],
    },
    units='lbm',
    desc='instrument group mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Instruments.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CW(2)', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.instrument_group_weight',
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
        'LEAPS1': None,
    },
    option=True,
    default_value=0.0,
    units='unitless',
    desc='landing gear drag coefficient',
)

add_meta_data(
    Aircraft.LandingGear.FIXED_GEAR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.IGEAR', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'INGASP.YMG', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': '(WeightABC)self._landing_gear_main_weight',
    },
    units='lbm',
    desc='mass of main landing gear (WMG in GASP)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKMG', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='mass trend coefficient of main gear, fraction of total landing gear',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.FRLGM',  # ['&DEFINE.WTIN.FRLGM', 'WTS.FRLGM'],
        'LEAPS1': 'aircraft.inputs.L0_overrides.landing_gear_main_weight',
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
        'LEAPS1': [
            'aircraft.inputs.L0_landing_gear.extend_main_gear_oleo_len',
            'aircraft.outputs.L0_landing_gear.extend_main_gear_oleo_len',
            'aircraft.cached.L0_landing_gear.extend_main_gear_oleo_len',
        ],
    },
    units='inch',
    desc='length of extended main landing gear oleo',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKLG', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': '(WeightABC)self._landing_gear_nose_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.landing_gear_nose_weight',
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
        'LEAPS1': [
            'aircraft.inputs.L0_landing_gear.extend_nose_gear_oleo_len',
            'aircraft.outputs.L0_landing_gear.extend_nose_gear_oleo_len',
            'aircraft.cached.L0_landing_gear.extend_nose_gear_oleo_len',
        ],
    },
    units='inch',
    desc='length of extended nose landing gear oleo',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKTL', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='factor on tail mass for arresting hook',
    default_value=1,
)

add_meta_data(
    Aircraft.LandingGear.TOTAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WLG', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of landing gear',
    default_value=0.0,
)

add_meta_data(
    Aircraft.LandingGear.TOTAL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CK12', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_engine.nacelle_avg_diam',
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
        'LEAPS1': 'aircraft.inputs.L0_engine.nacelle_avg_length',
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table[4]',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table[4]',
        ],
    },
    units='ft',
    desc='Reynolds characteristic length for nacelle for each engine model',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.CLEARANCE_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLEARqDN', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='the minimum number of nacelle diameters above the ground that the bottom of the nacelle must be',
    default_value=0.0,  # should be at least 0.2
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.CORE_DIAMETER_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DNQDE', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='ratio of nacelle diameter to engine core diameter',
    default_value=1.25,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.XLQDE',
        'FLOPS': None,  # 'MISSA.FR[5]',
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table[4]',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table[4]',
        ],
    },
    units='unitless',
    desc='nacelle fineness ratio',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKN', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.nacelle_percent_laminar_flow_lower_surface',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.nacelle_percent_laminar_flow_upper_surface',
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
        'LEAPS1': [
            '(WeightABC)self._nacelle_weight',
            'aircraft.outputs.L0_weights_summary.nacelle_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.nacelle_weight',
    },
    units='unitless',
    desc='mass scaler of the nacelle structure for each engine model',
    default_value=1.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.MASS_SPECIFIC,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.UWNAC', 'FLOPS': None, 'LEAPS1': None},
    units='lbm/ft**2',
    desc='nacelle mass/nacelle surface area; lbm per sq ft.',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HEBQDN', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='percentage of nacelle diameter buried in fuselage over nacelle diameter',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Aircraft.Nacelle.SURFACE_AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SN', 'FLOPS': None, 'LEAPS1': None},  # SN is wetted area
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
        'LEAPS1': 'aircraft.outputs.L0_aerodynamics.nacelle_wetted_area',
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table[4]',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table[4]',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.nacelle_wetted_area',
    },
    units='unitless',
    desc='nacelle wetted area scaler for each engine model',
    default_value=1.0,
    multivalue=True,
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
        'LEAPS1': [
            '(WeightABC)self._total_paint_weight',
            'aircraft.outputs.L0_weights_summary.total_paint_weight',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_weights.paint_per_unit_area',
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
    Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WOIL', 'MISWT.WOIL', 'MISWT.OOIL'],
        'FLOPS': 'WTIN.WOIL',
        'LEAPS1': 'aircraft.inputs.L0_overrides.engine_oil_weight',
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
        'LEAPS1': [
            '(WeightABC)self._prop_sys_weight',
            'aircraft.outputs.L0_weights_summary.prop_sys_weight',
        ],
    },
    units='lbm',
    desc='Total propulsion group mass',
    default_value=0.0,
)

# TODO clash with per-engine scaling, need to resolve w/ heterogeneous engine
add_meta_data(
    Aircraft.Propulsion.MISC_MASS_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFINE.WTIN.WPMSC', 'MISWT.WPMSC', 'MISWT.OPMSC'],
        'FLOPS': 'WTIN.WPMSC',
        'LEAPS1': ['aircraft.inputs.L0_overrides.misc_propulsion_weight'],
    },
    units='unitless',
    desc='scaler applied to miscellaneous engine mass (sum of engine control, starter, '
    'and additional mass)',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total estimated mass of the engine controls for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_ENGINE_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WEP', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._engine_oil_weight',
            'aircraft.outputs.L0_weights_summary.engine_oil_weight',
        ],
    },
    units='lbm',
    desc='engine oil mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total engine pod mass for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Propulsion.MISC_WEIGHT_SCALER
    Aircraft.Propulsion.TOTAL_MISC_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='sum of engine control, starter, and additional mass for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_NUM_ENGINES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='total number of engines for the aircraft (fuselage, wing, or otherwise)',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='total number of fuselage-mounted engines for the aircraft',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='total number of wing-mounted engines for the aircraft',
    types=int,
    option=True,
    default_value=0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_REFERENCE_SLS_THRUST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='total maximum thrust of all unscalsed engines on aircraft, sea-level static',
    option=True,
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='total maximum thrust of all scaled engines on aircraft, sea-level static',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Propulsion.TOTAL_STARTER_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='total mass of starters for all engines on aircraft',
    default_value=0.0,
)

add_meta_data(
    # Note user override
    #    - see also: Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER
    Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'INGASP.STRTWS', 'FLOPS': None, 'LEAPS1': None},
    units='ft**2',
    desc='strut area',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.AREA_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SSTQSW', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': None,
    },
    units='ft',
    desc='attachment location of strut the full attachment-to-attachment span',
    default_value=0.0,
)

# related to Aircraft.Strut.ATTACHMENT_LOCATION
add_meta_data(
    Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='attachment location of strut as fraction of the half-span',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STRTCHD', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='chord of the strut',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    default_value=True,
    types=bool,
    desc='if true the location of the strut is given dimensionally, otherwise '
    'it is given non-dimensionally. In GASP this depended on STRUT',
)

add_meta_data(
    Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKSTRT', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='strut/fuselage interference factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Strut.LENGTH,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.STRTLNG', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='length of the strut',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WSTRUT', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of the strut',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKSTRUT', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='mass trend coefficient of the strut',
    default_value=0,
)

add_meta_data(
    Aircraft.Strut.THICKNESS_TO_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TCSTRT', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'INGASP.ELFFC', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_vertical_tails.area',
            'aircraft.cached.L0_vertical_tails.area',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_vertical_tails.aspect_ratio',
            # ??? where is this assigned; potential error???
            'aircraft.cached.L0_vertical_tails.aspect_ratio',
        ],
    },
    units='unitless',
    desc='vertical tail theoretical aspect ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.AVERAGE_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CBARVT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table[2]',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table[2]',
        ],
    },
    units='ft',
    desc='Reynolds characteristic length for the vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.FINENESS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.FR[3]',
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table[2]',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table[2]',
        ],
    },
    units='unitless',
    desc='vertical tail fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKVT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.vertical_tail_percent_laminar_flow_lower_surface',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.vertical_tail_percent_laminar_flow_upper_surface',
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
        'LEAPS1': [
            '(WeightABC)self._vertical_tail_weight',
            'aircraft.outputs.L0_weights_summary.vertical_tail_weight',
        ],
    },
    units='lbm',
    desc='mass of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKZ', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.vertical_tail_weight',
    },
    units='unitless',
    desc='mass scaler of the vertical tail structure',
    default_value=1.0,
)

add_meta_data(
    Aircraft.VerticalTail.MOMENT_ARM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ELTV', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='moment arm of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.MOMENT_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BOELTV', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='ratio of wing span to vertical tail moment arm',
)

add_meta_data(
    Aircraft.VerticalTail.NUM_TAILS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.NVERT',  # ['&DEFINE.WTIN.NVERT', 'EDETIN.NVERT'],
        'LEAPS1': 'aircraft.inputs.L0_vertical_tails.count',
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
    historical_name={'GASP': 'INGASP.CRCLVT', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='root chord of vertical tail',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.SPAN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BVT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_vertical_tail.sweep_at_quarter_chord',
            'aircraft.cached.L0_vertical_tail.sweep_at_quarter_chord',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_vertical_tails.taper_ratio',
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
        'LEAPS1': [
            'aircraft.inputs.L0_vertical_tails.thickness_to_chord_ratio',
            'aircraft.cached.L0_vertical_tails.thickness_to_chord_ratio',
        ],
    },
    units='unitless',
    desc='vertical tail thickness-chord ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.VerticalTail.VOLUME_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VBARVX', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.vertical_tail_wetted_area',
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table[2]',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table[2]',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.vertical_tail_wetted_area',
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
        'LEAPS1': 'aircraft.inputs.L0_wing.aeroelastic_fraction',
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
        'LEAPS1': [
            'aircraft.inputs.L0_aerodynamics.airfoil',
            'aircraft.outputs.L0_aerodynamics.mission_airfoil',
            'aircraft.cached.L0_aerodynamics.mission_airfoil',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.wing_ref_area',
            'aircraft.outputs.L0_design_variables.wing_ref_area',
            'aircraft.outputs.L0_design_variables.mission_wing_ref_area',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.wing_aspect_ratio',
            'aircraft.outputs.L0_design_variables.wing_aspect_ratio',
            'aircraft.outputs.L0_design_variables.mission_wing_aspect_ratio',
        ],
    },
    units='unitless',
    desc='ratio of the wing span to its mean chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.ASPECT_RATIO_REF,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.ARREF',  # ['&DEFINE.WTIN.ARREF'],
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.ref_aspect_ratio',
    },
    units='unitless',
    desc='Reference aspect ratio, used for detailed wing mass estimation.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.AVERAGE_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CBARW', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.outputs.L0_wing.bending_material_factor',
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
        'LEAPS1': 'aircraft.outputs.L0_wing.bending_mat_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.wing_bending_mat_weight',
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
        'LEAPS1': 'aircraft.outputs.L0_wing.bwb_aft_body_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.bwb_aft_body_weight',
    },
    units='unitless',
    desc='mass scaler of the blended-wing-body aft-body wing mass term',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.CENTER_CHORD,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRCLW', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='wing chord at fuselage centerline, usually called root chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.CENTER_DISTANCE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XWQLF', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='distance (percent fuselage length) from nose to the wing aerodynamic center',
)

add_meta_data(
    Aircraft.Wing.CHARACTERISTIC_LENGTH,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # 'MISSA.EL[1]',
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_component_char_len_table[0]',
            'aircraft.cached.L0_aerodynamics.mission_component_char_len_table[0]',
        ],
    },
    units='ft',
    desc='Reynolds characteristic length for the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.CHOOSE_FOLD_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    default_value=True,
    types=bool,
    option=True,
    desc='if true, fold location is based on your chosen value, otherwise it is '
    'based on strut location. In GASP this depended on STRUT or YWFOLD',
)

add_meta_data(
    # see also: station_chord_lengths
    Aircraft.Wing.CHORD_PER_SEMISPAN_DIST,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.CHD',  # ['&DEFINE.WTIN.CHD', 'WDEF.CHD'],
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.wing_station_chord_lengths',
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
        'LEAPS1': 'aircraft.inputs.L0_wing.composite_fraction',
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
        'LEAPS1': [
            # TODO ~WingWeight.__call__.flap_ratio: see ~WWGHT.SFLAP
            '~WeightABC._pre_surface_ctrls.surface_flap_area',
            '~WeightABC.calc_surface_ctrls.surface_flap_area',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_wing.flap_ratio',
    },
    units='unitless',
    desc='Defines the ratio of total moveable wing control surface areas '
    '(flaps, elevators, spoilers, etc.) to reference wing area.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.DETAILED_WING,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_wing.dihedral',
            # unit converted value for reporting
            'aircraft.cached.L0_wing.dihedral',
        ],
    },
    units='deg',
    desc='wing dihedral (positive) or anhedral (negative) angle',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~WWGHT.CAYE',
        'LEAPS1': 'aircraft.outputs.L0_wing.engine_inertia_relief_factor',
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
        'LEAPS1': None,
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.mission_fineness_ratio_table[0]',
            'aircraft.cached.L0_aerodynamics.mission_fineness_ratio_table[0]',
        ],
    },
    units='unitless',
    desc='wing fineness ratio',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FLAP_CHORD_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CFOC', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='ratio of flap chord to wing chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FLAP_DEFLECTION_LANDING,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DFLPLD', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='Deflection of flaps for landing',
)

add_meta_data(
    Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DFLPTO', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='Deflection of flaps for takeoff',
)

add_meta_data(
    Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCDOTE', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='drag coefficient increment due to optimally deflected trailing edge flaps (default depends on flap type)',
)

add_meta_data(
    Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCLMTE', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='lift coefficient increment due to optimally deflected trailing edge flaps (default depends on flap type)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FLAP_SPAN_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BTEOB', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='fraction of wing trailing edge with flaps',
    default_value=0.65,
)

add_meta_data(
    Aircraft.Wing.FLAP_TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.JFLTYP', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': 'INGASP.WWFOLD', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of the folding area of the wing',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.FOLD_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKWFOLD', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='mass trend coefficient of the wing fold',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.FOLDED_SPAN,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.YWFOLD', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='folded wingspan',
    default_value=0,
)

add_meta_data(
    Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='folded wingspan',
    default_value=1,
)

add_meta_data(
    Aircraft.Wing.FOLDING_AREA,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SWFOLD', 'FLOPS': None, 'LEAPS1': None},
    units='ft**2',
    desc='wing area of folding part of wings',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FORM_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKW', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='wing form factor',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CKI', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_wing.glove_and_bat',
    },
    units='ft**2',
    desc='total glove and bat area beyond theoretical wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.HAS_FOLD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    desc='if true a fold will be included in the wing',
    default_value=False,
    types=bool,
)

add_meta_data(
    Aircraft.Wing.HAS_STRUT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    option=True,
    units='unitless',
    default_value=False,
    types=bool,
    desc='if true then aircraft has a strut. In GASP this depended on STRUT',
)

add_meta_data(
    Aircraft.Wing.HEIGHT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HTG', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='wing height above ground during ground run, measured at roughly '
    'location of mean aerodynamic chord at the mid plane of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.HIGH_LIFT_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WHLDEV', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of the high lift devices',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WCFLAP', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='mass trend coefficient of high lift devices (default depends on flap type)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.INCIDENCE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.EYEW', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='incidence angle of the wings with respect to the fuselage',
    default_value=0.0,
)

add_meta_data(
    # see also: station_locations
    # NOTE required for blended-wing-body type aircraft
    Aircraft.Wing.INPUT_STATION_DIST,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.ETAW',  # ['&DEFINE.WTIN.ETAW', 'WDEF.ETAW'],
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.wing_station_locations',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.wing_percent_laminar_flow_lower_surface',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.wing_percent_laminar_flow_upper_surface',
    },
    units='unitless',
    desc='define percent laminar flow for wing upper surface',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.LEADING_EDGE_SWEEP,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SWPLE', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.pressure_dist',
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
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.carried_load_fraction',
    },
    units='unitless',
    desc='fraction of load carried by defined wing',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.SWL',  # ['&DEFINE.WTIN.SWL', 'WDEF.SWL'],
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.wing_station_load_path_sweeps',
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            '(WeightABC)self._total_wing_weight',
            'aircraft.outputs.L0_weights_summary.total_wing_weight',
        ],
    },
    units='lbm',
    desc='wing total mass',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKWW', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.total_wing_weight',
    },
    units='unitless',
    desc='mass scaler of the overall wing',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.MATERIAL_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKNO', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_aerodynamics.max_camber_at_70_semispan',
            'aircraft.outputs.L0_aerodynamics.mission_max_camber_at_70_semispan',
        ],
    },
    units='unitless',
    desc='Maximum camber at 70 percent semispan, percent of local chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MAX_LIFT_REF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.RCLMAX', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='input reference maximum lift coefficient for basic wing',
)

add_meta_data(
    Aircraft.Wing.MAX_SLAT_DEFLECTION_LANDING,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELLED', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='leading edge slat deflection during landing',
    default_value=10,
)

add_meta_data(
    Aircraft.Wing.MAX_SLAT_DEFLECTION_TAKEOFF,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELLED', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='leading edge slat deflection during takeoff',
    default_value=10,
)

add_meta_data(
    Aircraft.Wing.MAX_THICKNESS_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XCTCMX', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='location (percent chord) of max wing thickness',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.MIN_PRESSURE_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XCPS', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.outputs.L0_wing.misc_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.wing_misc_weight',
    },
    units='unitless',
    desc='mass scaler of the miscellaneous wing mass term',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.NUM_FLAP_SEGMENTS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.FLAPN', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.integration_station_count',
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
    historical_name={'GASP': 'INGASP.DELTEO', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='optimum flap deflection angle (default depends on flap type)',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DELLEO', 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='optimum slat deflection angle',
    default_value=20,
)

add_meta_data(
    Aircraft.Wing.ROOT_CHORD,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CROOTW',
        'FLOPS': 'WTIN.XLW',
        'LEAPS1': None,
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
        'LEAPS1': 'aircraft.outputs.L0_wing.struct_weight',
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.wing_struct_weights',
    },
    units='unitless',
    desc='mass scaler of the shear and control term',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.SLAT_CHORD_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CLEOC', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='ratio of slat chord to wing chord',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCLMLE', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='lift coefficient increment due to optimally deflected LE slats',
)

add_meta_data(
    Aircraft.Wing.SLAT_SPAN_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.BLEOB', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_wing.span',
            'aircraft.outputs.L0_wing.span',
            'BasicTransportWeight.wing_span',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.wing_span_efficiency_factor',
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.wing_span_efficiency_reduction',
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
        'LEAPS1': 'aircraft.inputs.L0_wing.struct_bracing_factor',
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
        'LEAPS1': [
            '(WeightABC)self._surface_ctrls_weight',
            'aircraft.outputs.L0_weights_summary.surface_ctrls_weight',
        ],
    },
    units='lbm',
    desc='mass of surface controls',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SKFW', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_overrides.surface_ctrls_weight',
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
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.wing_sweep_at_quarter_chord',
            'aircraft.outputs.L0_design_variables.wing_sweep_at_quarter_chord',
            'aircraft.outputs.L0_design_variables.mission_wing_sweep_at_quarter_chord',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.wing_taper_ratio',
            'aircraft.outputs.L0_design_variables.wing_taper_ratio',
            'aircraft.outputs.L0_design_variables.mission_wing_taper_ratio',
        ],
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
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.wing_thickness_to_chord_ratio',
            'aircraft.outputs.L0_design_variables.wing_thickness_to_chord_ratio',
            'aircraft.outputs.L0_design_variables.mission_wing_thickness_to_chord_ratio',
        ],
    },
    units='unitless',
    desc='wing thickness-chord ratio (weighted average)',
    default_value=0.0,  # TODO required
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TOC',  # ['&DEFINE.WTIN.TOC', 'WDEF.TOC'],
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.wing_station_thickness_to_chord_ratios',
    },
    units='unitless',
    desc='the thickeness-chord ratios at station locations',
    default_value=[0.0],
    types=float,
    multivalue=True,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_REF,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'WTIN.TCREF',  # ['&DEFINE.WTIN.TCREF'],
        'LEAPS1': 'aircraft.inputs.L0_detailed_wing.ref_thickness_to_chord_ratio',
    },
    units='unitless',
    desc='Reference thickness-to-chord ratio, used for detailed wing mass estimation.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TCR', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='thickness-to-chord ratio at the root of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TCT', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='thickness-to-chord ratio at the tip of the wing',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TC', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_weights.struct_ult_load_factor',
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
        'LEAPS1': 'aircraft.inputs.L0_wing.var_sweep_weight_penalty',
    },
    units='unitless',
    desc='Define the fraction of wing variable sweep mass penalty where: '
    '0.0 == fixed-geometry wing; 1.0 == full variable-sweep wing.',
    default_value=0.0,
)

add_meta_data(
    Aircraft.Wing.VERTICAL_MOUNT_LOCATION,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HWING', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.outputs.L0_aerodynamics.wing_wetted_area',
            'aircraft.outputs.L0_aerodynamics.mission_component_wetted_area_table[0]',
            'aircraft.cached.L0_aerodynamics.mission_component_wetted_area_table[0]',
        ],
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
        'LEAPS1': 'aircraft.inputs.L0_aerodynamics.wing_wetted_area',
    },
    units='unitless',
    desc='wing wetted area scaler',
    default_value=1.0,
)

add_meta_data(
    Aircraft.Wing.ZERO_LIFT_ANGLE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ALPHL0', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/ft**3',
    desc="Atmospheric density at the vehicle's current altitude",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.DYNAMIC_PRESSURE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf/ft**2',
    desc="Atmospheric dynamic pressure at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'XKV', 'FLOPS': None, 'LEAPS1': None},
    units='ft**2/s',
    desc="Atmospheric kinematic viscosity at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.MACH,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Current Mach number of the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.MACH_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Current rate at which the Mach number of the vehicle is changing',
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.SPEED_OF_SOUND,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft/s',
    desc="Atmospheric speed of sound at vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.STATIC_PRESSURE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf/ft**2',
    desc="Atmospheric static pressure at the vehicle's current flight condition",
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Atmosphere.TEMPERATURE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='Current altitude of the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.ALTITUDE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft/s',
    desc='Current rate of altitude change (climb rate) of the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.ALTITUDE_RATE_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft/s',
    desc='Current maximum possible rate of altitude change (climb rate) of the vehicle '
    '(at hypothetical maximum thrust condition)',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.DISTANCE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'range', 'LEAPS1': None},
    units='NM',
    desc='The total distance the vehicle has traveled since brake release at the current time',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.DISTANCE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'range_rate', 'LEAPS1': None},
    units='NM/s',
    desc='The rate at which the distance traveled is changing at the current time',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.FLIGHT_PATH_ANGLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='rad',
    desc='Current flight path angle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='rad/s',
    desc='Current rate at which flight path angle is changing',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.SPECIFIC_ENERGY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='m/s',
    desc='Rate of change in specific energy (energy per unit weight) of the vehicle at '
    'current flight condition',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.SPECIFIC_ENERGY_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='m/s',
    desc='Rate of change in specific energy (specific power) of the vehicle at current '
    'flight condition',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='m/s',
    desc='Specific excess power of the vehicle at current flight condition and at '
    'hypothetical maximum thrust',
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.VELOCITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft/s',
    desc='Current velocity of the vehicle along its body axis',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Mission.VELOCITY_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='deg',
    desc='Angle between aircraft wing cord and relative wind',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.BATTERY_STATE_OF_CHARGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc="battery's current state of charge",
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='kJ',
    desc='Total amount of electric energy consumed by the vehicle up until this point '
    'in the mission',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.DRAG,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='Current total drag experienced by the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.LIFT,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='Current total lift produced by the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='Current total mass of the vehicle',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.MASS_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='kW',
    desc='Current electric power consumption of each engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='kW',
    desc='Current total electric power consumption of the vehicle',
    multivalue=True,
)

# add_meta_data(
#     Dynamic.Vehicle.Propulsion.EXIT_AREA,
#     meta_data=_MetaData,
#     historical_name={'GASP': None,
#                     'FLOPS': None,
#                     'LEAPS1': None
#                     },
#     units='kW',
#     desc='Current nozzle exit area of engines, per single instance of each '
#          'engine model'
# )

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/h',
    desc='Current rate of fuel consumption of the vehicle, per single instance of '
    'each engine model. Consumption (i.e. mass reduction) of fuel is defined as '
    'positive.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/h',
    desc='Current rate of fuel consumption of the vehicle, per single instance of each '
    'engine model. Consumption (i.e. mass reduction) of fuel is defined as negative.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/h',
    desc='Current rate of total fuel consumption of the vehicle. Consumption (i.e. '
    'mass reduction) of fuel is defined as negative.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/h',
    desc='Current rate of total fuel consumption of the vehicle. Consumption (i.e. '
    'mass reduction) of fuel is defined as positive.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.HYBRID_THROTTLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Current secondary throttle setting of each individual engine model on the '
    'vehicle, used as an additional degree of control for hybrid engines',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.NOX_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/h',
    desc='Current rate of nitrous oxide (NOx) production by the vehicle, per single '
    'instance of each engine model',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm/h',
    desc='Current total rate of nitrous oxide (NOx) production by the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft/s',
    desc='linear propeller tip speed due to rotation (not airspeed at propeller tip)',
    default_value=0.0,
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.RPM,
    meta_data=_MetaData,
    historical_name={'GASP': ['RPM', 'RPMe'], 'FLOPS': None, 'LEAPS1': None},
    units='rpm',
    desc='Rotational rate of shaft, per engine.',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.SHAFT_POWER,
    meta_data=_MetaData,
    historical_name={'GASP': ['SHP, EHP'], 'FLOPS': None, 'LEAPS1': None},
    units='hp',
    desc='current shaft power, per engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='hp',
    desc='The maximum possible shaft power currently producible, per engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='degR',
    desc='Current turbine exit temperature (T4) of turbine engines on vehicle, per '
    'single instance of each engine model',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THROTTLE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='Current throttle setting for each individual engine model on the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='Current net thrust produced by engines, per single instance of each engine model',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='Hypothetical maximum possible net thrust that can be produced per single '
    "instance of each engine model at the vehicle's current flight condition",
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='Hypothetical maximum possible net thrust produced by the vehicle at its '
    'current flight condition',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbf',
    desc='Current total net thrust produced by the vehicle',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.TORQUE,
    meta_data=_MetaData,
    historical_name={'GASP': 'TORQUE', 'FLOPS': None, 'LEAPS1': None},
    units='N*m',
    desc='Current torque being produced, per engine',
    multivalue=True,
)

add_meta_data(
    Dynamic.Vehicle.Propulsion.TORQUE_MAX,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='Difference between the usable fuel capacity on the aircraft and the total fuel (including reserve) required for the mission. '
    'Must be >= 0 to ensure that the aircraft has enough fuel to complete the mission',
)

add_meta_data(
    Mission.Constraints.GEARBOX_SHAFT_POWER_RESIDUAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='kW',
    desc='Must be zero or positive to ensure that the gearbox is sized large enough to handle the maximum shaft power the engine could output during any part of the mission',
)

add_meta_data(
    Mission.Constraints.MASS_RESIDUAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_weights.max_mach',
            'aircraft.outputs.L0_weights.max_mach',
        ],
    },
    units='unitless',
    desc='aircraft cruise Mach number',
    # TODO: derived default value: Mission.Summary.CRUISE_MACH ???
    default_value=0.0,
    option=True,
)

add_meta_data(
    Mission.Constraints.RANGE_RESIDUAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='NM',
    desc='residual to make sure aircraft range is equal to the targeted '
    'range, value should be zero at convergence (within acceptable '
    'tolerance)',
)

add_meta_data(
    Mission.Constraints.RANGE_RESIDUAL_RESERVE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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

add_meta_data(
    Mission.Design.CRUISE_ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.CRALT', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    option=True,
    default_value=25000.0,
    desc='design mission cruise altitude',
)

add_meta_data(
    Mission.Design.CRUISE_RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='NM',
    desc='the distance flown by the aircraft during cruise',
    default_value=0.0,
)

add_meta_data(
    Mission.Summary.FUEL_MASS_REQUIRED,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFAREQ', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='fuel carried by the aircraft when it is on the ramp at the beginning of the design '
    'mission',
    default_value=0.0,
)

add_meta_data(
    Mission.Design.GROSS_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.WG',
        # ['&DEFINE.WTIN.DGW', 'WTS.DGW', '~WEIGHT.DG', '~WWGHT.DG'],
        'FLOPS': 'WTIN.DGW',
        'LEAPS1': [  # TODO: 'aircraft.inputs.L0_weights.design_ramp_weight_fraction' ???
            #    - design_ramp_weight_fraction has a default: 1.0
            #    - design_ramp_weight does not have an explicit default
            #        - design_ramp_weight has an implicit default, by way of
            #          design_ramp_weight_fraction:
            #          [L0_design_variables] ramp_weight
            'aircraft.inputs.L0_weights.design_ramp_weight',
            '(weightABC)self._design_gross_weight',
        ],
    },
    units='lbm',
    desc='design gross mass of the aircraft',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override (no scaling)
    Mission.Design.LIFT_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.FCLDES',
        #  [  # inputs
        #      '&DEFINE.AERIN.FCLDES', 'OSWALD.FCLDES',
        #      # outputs
        #      '~EDET.CLDES', '~CLDESN.CLDES', '~MDESN.CLDES'
        #  ],
        'LEAPS1': [
            'aircraft.inputs.L0_aerodynamics.design_lift_coeff',
            'aircraft.outputs.L0_aerodynamics.design_lift_coeff',
        ],
    },
    units='unitless',
    desc='Fixed design lift coefficient. If input, overrides design lift '
    'coefficient computed by EDET.',
    default_value=0.0,
)

add_meta_data(
    Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
    meta_data=_MetaData,
    historical_name={
        'GASP': ['INGASP.CLMWFU', 'INGASP.CLMAX'],
        'FLOPS': None,
        'LEAPS1': None,
    },
    units='unitless',
    desc='maximum lift coefficient from flaps model when flaps are up (not deployed)',
    default_value=0.0,
)

add_meta_data(
    # NOTE: user override (no scaling)
    Mission.Design.MACH,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CRMACH',
        'FLOPS': 'AERIN.FMDES',
        #  [  # inputs
        #      '&DEFINE.AERIN.FMDES', 'OSWALD.FMDES'
        #      # outputs
        #      '~EDET.DESM', '~MDESN.DESM'
        #  ],
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.design_mach',
            'aircraft.outputs.L0_design_variables.design_mach',
        ],
    },
    units='unitless',
    desc='aircraft design Mach number',
    default_value=0.0,
)

add_meta_data(
    Mission.Design.RANGE,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.ARNGE',
        # ['&DEFINE.CONFIN.DESRNG', 'CONFIG.DESRNG'],
        'FLOPS': 'CONFIN.DESRNG',
        'LEAPS1': 'aircraft.inputs.L0_configuration.design_range',
    },
    units='NM',
    desc='the aircraft target distance',
    default_value=0.0,
)

add_meta_data(
    Mission.Design.RATE_OF_CLIMB_AT_TOP_OF_CLIMB,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.ROCTOC', 'FLOPS': None, 'LEAPS1': None},
    option=True,
    units='ft/min',
    desc='The required rate of climb at top of climb',
    default_value=0.0,
)

add_meta_data(
    Mission.Design.RESERVE_FUEL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='the total fuel reserves which is the sum of: '
    'RESERVE_FUEL_BURNED, RESERVE_FUEL_ADDITIONAL, RESERVE_FUEL_FRACTION',
    default_value=0,
)

add_meta_data(
    # TODO move to Engine?
    # TODO this isn't actually tied to the engines in any way - user provided value is
    #      arbitrary and will not update as engines resize
    Mission.Design.THRUST_TAKEOFF_PER_ENG,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # FLOPS may scale the input value as it resizes the engine if requested by
        # the user
        # ['&DEFINE.AERIN.THROFF', 'LANDG.THROF', 'LANDG.THROFF'],
        'FLOPS': 'AERIN.THROFF',
        # LEAPS1 uses the average thrust_takeoff of all operational engines
        # actually on the airplane, possibly after resizing (as with FLOPS)
        'LEAPS1': [
            'aircraft.inputs.L0_engine.thrust_takeoff',
            '(SimpleTakeoff)self.thrust',
        ],
    },
    units='lbf',
    # need better description of what state. rolling takeoff condition? alt? mach?
    desc='thrust on the aircraft for takeoff',
    default_value=0.0,
)

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
    historical_name={'GASP': 'INGASP.ALTLND', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='altitude of airport where aircraft lands',
    default_value=0,
)

add_meta_data(
    Mission.Landing.BRAKING_DELAY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.TDELAY', 'FLOPS': None, 'LEAPS1': None},
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
    #     'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.braking_mu'},
    historical_name={'FLOPS': None, 'GASP': None, 'LEAPS1': None},
    default_value=0.3,
    units='unitless',
    desc='landing coefficient of friction, with brakes on',
)

add_meta_data(
    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCD', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='drag coefficient increment at landing due to flaps',
)

add_meta_data(
    Mission.Landing.DRAG_COEFFICIENT_MIN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.CDMLD',  # ['&DEFINE.AERIN.CDMLD', 'LANDG.CDMLD'],
        'LEAPS1': None,
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
        'LEAPS1': '(SimpleLanding)self.landing_distance',
    },
    units='ft',
    desc='FAR landing field length',
)

add_meta_data(
    Mission.Landing.FLARE_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': 'TOLIN.VANGLD', 'LEAPS1': None},
    units='deg/s',
    desc='flare rate in detailed landing',
    default_value=2.0,
)

add_meta_data(
    Mission.Landing.GLIDE_TO_STALL_RATIO,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VRATT', 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': None,
    },
    units='ft',
    desc='distance covered over the ground during landing',
)

add_meta_data(
    Mission.Landing.INITIAL_ALTITUDE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HIN', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='altitude where landing calculations begin',
)

add_meta_data(
    Mission.Landing.INITIAL_MACH,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': '(SimpleLanding)self.vapp',
    },
    units='ft/s',
    desc='approach velocity',
)

add_meta_data(
    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCL', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='lift coefficient increment at landing due to flaps',
)

add_meta_data(
    # TODO: missing &DEFINE.AERIN.CLAPP ???
    #    - NOTE: there is a relationship in FLOPS/LEAPS1 between CLAPP and
    #      CLLDM (this variable)
    Mission.Landing.LIFT_COEFFICIENT_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CLMWLD',
        'FLOPS': 'AERIN.CLLDM',  # ['&DEFINE.AERIN.CLLDM', 'LANDG.CLLDM'],
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.max_landing_lift_coeff',
    },
    units='unitless',
    desc='maximum lift coefficient for landing',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.XLFMX', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='maximum load factor during landing flare',
    default_value=1.15,
)

add_meta_data(
    Mission.Landing.MAXIMUM_SINK_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.RSMX', 'FLOPS': None, 'LEAPS1': None},
    units='ft/min',
    desc='maximum rate of sink during glide',
    default_value=1000,
)

add_meta_data(
    Mission.Landing.OBSTACLE_HEIGHT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.HAPP', 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='landing obstacle height above the ground at airport altitude',
    default_value=50,
)

add_meta_data(
    Mission.Landing.ROLLING_FRICTION_COEFFICIENT,
    meta_data=_MetaData,
    # historical_name={"GASP": None,
    #                  "FLOPS": ['&DEFTOL.TOLIN.ROLLMU', 'BALFLD.ROLLMU'],
    #                  "LEAPS1": ['aircraft.inputs.L0_takeoff_and_landing.rolling_mu',
    #                             '(GroundRoll)self.mu',
    #                             '(Rotate)self.mu',
    #                             '(GroundBrake)self.rolling_mu',
    #                             ]
    #                  },
    historical_name={'FLOPS': None, 'GASP': None, 'LEAPS1': None},
    units='unitless',
    desc='coefficient of rolling friction for groundroll portion of takeoff',
    default_value=0.025,
)

add_meta_data(
    Mission.Landing.SPOILER_DRAG_COEFFICIENT,
    meta_data=_MetaData,
    # historical_name={"GASP": None,
    #                  "FLOPS": '&DEFTOL.TOLIN.CDSPOL',
    #                  "LEAPS1": None
    #                  },
    historical_name={'FLOPS': None, 'GASP': None, 'LEAPS1': None},
    units='unitless',
    desc='drag coefficient for spoilers during landing rollout',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.SPOILER_LIFT_COEFFICIENT,
    meta_data=_MetaData,
    # historical_name={"GASP": None,
    #                  "FLOPS": '&DEFTOL.TOLIN.CLSPOL',
    #                  "LEAPS1": None
    #                  },
    historical_name={'FLOPS': None, 'GASP': None, 'LEAPS1': None},
    units='unitless',
    desc='lift coefficient for spoilers during landing rollout',
    default_value=0.0,
)

add_meta_data(
    Mission.Landing.STALL_VELOCITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VST', 'FLOPS': None, 'LEAPS1': None},
    units='ft/s',
    desc='stall speed during approach',
)

add_meta_data(
    Mission.Landing.TOUCHDOWN_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # ['~ANALYS.WLDG', '~LNDING.GROSWT'],
        'LEAPS1': '(SimpleLanding)self.weight',
    },
    units='lbm',
    desc='computed mass of aircraft for landing, is only '
    'required to be equal to Aircraft.Design.TOUCHDOWN_MASS '
    'when the design case is being run '
    'for HEIGHT_ENERGY missions this is the mass at the end of the last regular phase (non-reserve phase)',
)

add_meta_data(
    Mission.Landing.TOUCHDOWN_SINK_RATE,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.SINKTD', 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='regularized objective that minimizes total fuel mass subject '
    'to other necessary additions',
)

add_meta_data(
    Mission.Objectives.RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='regularized objective that maximizes range subject to other necessary additions',
)

#   _____
#  / ____|
# | (___    _   _   _ __ ___    _ __ ___     __ _   _ __   _   _
#  \___ \  | | | | | '_ ` _ \  | '_ ` _ \   / _` | | '__| | | | |
#  ____) | | |_| | | | | | | | | | | | | | | (_| | | |    | |_| |
# |_____/   \__,_| |_| |_| |_| |_| |_| |_|  \__,_| |_|     \__, |
#                                                           __/ |
#                                                          |___/
# ===============================================================

add_meta_data(
    Mission.Summary.CRUISE_MACH,
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
        'LEAPS1': [
            'aircraft.inputs.L0_design_variables.cruise_mach',
            'aircraft.outputs.L0_design_variables.cruise_mach',
            'aircraft.outputs.L0_design_variables.mission_cruise_mach',
        ],
    },
    units='unitless',
    desc='aircraft cruise Mach number',
    default_value=0.0,  # TODO: required
)

add_meta_data(
    Mission.Summary.CRUISE_MASS_FINAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass of the aircraft at the end of cruise',
    default_value=0.0,
)

add_meta_data(
    Mission.Summary.FINAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'None', 'FLOPS': None, 'LEAPS1': None},  # TODO: Check on these
    units='lbm',
    desc='The final weight of the vehicle at the end of the last regular_phase (does not include reserve phases).',
)

add_meta_data(
    Mission.Summary.FINAL_TIME,
    meta_data=_MetaData,
    historical_name={'GASP': 'None', 'FLOPS': None, 'LEAPS1': None},  # TODO: Check on these
    units='min',
    desc='Total mission time from the start of the first regular_phase'
    'to the end of the last regular_phase (does not include reserve phases).',
)

add_meta_data(
    Mission.Summary.FUEL_BURNED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='fuel burned during regular phases, this does not include fuel burned in reserve phases',
)

# NOTE if per-mission level scaling is not best mapping for GASP's 'CKFF', map
#      to FFFSUB/FFFSUP
# CKFF is consistent for one aircraft over all missions, once the vehicle is sized
# can we map it to both FFFSUB and FFFSUP?
add_meta_data(
    Mission.Summary.FUEL_FLOW_SCALER,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CKFF',
        'FLOPS': 'MISSIN.FACT',  # ['&DEFMSS.MISSIN.FACT', 'TRNSF.FACT'],
        'LEAPS1': ['aircraft.inputs.L0_fuel_flow.overall_factor'],
    },
    units='unitless',
    desc='scale factor on overall fuel flow',
    default_value=1.0,
    option=True,
)

add_meta_data(
    Mission.Summary.FUEL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.WFADES',
        'FLOPS': None,  # ['WSP(38, 2)', '~WEIGHT.FUELM', '~INERT.FUELM'],
        'LEAPS1': [
            '(WeightABC)self._fuel_weight',
            'aircraft.outputs.L0_weights_summary.fuel_weight',
        ],
    },
    units='lbm',
    desc='fuel carried by the aircraft when it is on the ramp at the beginning of the mission',
    default_value=0.0,
)

add_meta_data(
    Mission.Summary.GROSS_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='gross takeoff mass of aircraft for that specific mission, not '
    'necessarily the value for the aircraft`s design mission',
)

add_meta_data(
    Mission.Summary.OPERATING_MASS,
    meta_data=_MetaData,
    # TODO: check with Aviary and GASPy engineers to ensure these are indeed
    # defined the same way
    historical_name={
        'GASP': 'INGASP.OWE',
        # ['WTS.WSP(33, 2)', '~WEIGHT.WOWE', '~WTSTAT.WSP(33, 2)'],
        'FLOPS': 'MISSIN.DOWE',
        'LEAPS1': [
            '(WeightABC)self._operating_weight_empty',
            'aircraft.outputs.L0_weights_summary.operating_weight_empty',
        ],
    },
    units='lbm',
    desc='operating mass of the aircraft, or aircraft mass without mission fuel, or passengers.'
    'Includes crew, unusable fuel, oil, and operational items like cargo containers and passenger '
    'service mass.',
    default_value=0.0,
)

add_meta_data(
    Mission.Summary.RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='NM',
    desc='actual range that the aircraft flies, whether '
    'it is a design case or an off design case. Equal '
    'to Mission.Design.RANGE value in the design case.',
)

add_meta_data(
    Mission.Summary.RESERVE_FUEL_BURNED,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='fuel burned during reserve phases, this does not include fuel burned in regular phases',
    default_value=0.0,
)

add_meta_data(
    Mission.Summary.TOTAL_FUEL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.WFA', 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    # Note: In GASP, WFA does not include fuel margin.
    desc='total fuel carried at the beginnning of a mission includes fuel burned in the mission, '
    'reserve fuel and fuel margin',
)

add_meta_data(
    Mission.Summary.ZERO_FUEL_MASS,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['WTS.WSP(37,2)', '~WEIGHT.WZF', '~WTSTAT.WSP(37,2)'],
        'FLOPS': None,
        'LEAPS1': [
            '(WeightABC)self._zero_fuel_weight',
            'aircraft.outputs.L0_weights.zero_fuel_weight',
            'aircraft.outputs.L0_weights_summary.zero_fuel_weight',
        ],
    },
    units='lbm',
    desc='Aircraft zero fuel mass, which includes structural mass (empty weight) and payload mass '
    '(passengers, baggage, and cargo)',
    default_value=0.0,
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.alpha_runway',
    },
    option=True,
    default_value=0.0,
    units='deg',
    desc='angle of attack on ground',
)

add_meta_data(
    Mission.Takeoff.ASCENT_DURATION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='s',
    desc='duration of the ascent phase of takeoff',
)

add_meta_data(
    Mission.Takeoff.ASCENT_T_INITIAL,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.braking_mu',
    },
    default_value=0.3,
    units='unitless',
    desc='takeoff coefficient of friction, with brakes on',
)

add_meta_data(
    Mission.Takeoff.DECISION_SPEED_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DV1', 'FLOPS': None, 'LEAPS1': None},
    units='kn',
    desc='increment of engine failure decision speed above stall speed',
    default_value=5,
)

add_meta_data(
    Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCD', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='drag coefficient increment at takeoff due to flaps',
)

add_meta_data(
    Mission.Takeoff.DRAG_COEFFICIENT_MIN,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'AERIN.CDMTO',  # ['&DEFINE.AERIN.CDMTO', 'LANDG.CDMTO'],
        'LEAPS1': None,
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
        'LEAPS1': '(SimpleTakeoff)self.takeoff_distance',
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
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.obstacle_height',
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
        'LEAPS1': None,
    },
    units='unitless',
    desc='Mach number of aircraft after taking off and clearing a 35 foot obstacle',
)

add_meta_data(
    Mission.Takeoff.FINAL_MASS,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='lbm',
    desc='mass after aircraft has cleared 35 ft obstacle',
)

add_meta_data(
    # TODO FLOPS/LEAPS1 implementation is different from Aviary
    #    - correct variable reference?
    #    - correct Aviary equations?
    Mission.Takeoff.FINAL_VELOCITY,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': None,  # '~TOFF.V2',
        'LEAPS1': '(ClimbToObstacle)self.V2',
    },
    units='m/s',
    desc='velocity of aircraft after taking off and clearing a 35 foot obstacle',
)

add_meta_data(
    # Note user override (no scaling)
    # Note FLOPS/LEAPS1 calculated as part of mission analysis, and not as
    # part of takeoff
    Mission.Takeoff.FUEL_SIMPLE,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        # ['&DEFMSS.MISSIN.FTKOFL', 'FFLALL.FTKOFL', '~MISSON.TAKOFL'],
        'FLOPS': 'MISSIN.FTKOFL',
        'LEAPS1': [
            'aircraft.inputs.L0_mission.fixed_takeoff_fuel',
            'aircraft.outputs.L0_takeoff_and_landing.takeoff_fuel',
        ],
    },
    units='lbm',
    desc='fuel burned during simple takeoff calculation',
    default_value=0.0,
)

add_meta_data(
    Mission.Takeoff.GROUND_DISTANCE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='ft',
    desc='ground distance covered by takeoff with all engines operating',
)

add_meta_data(
    Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DCL', 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    desc='lift coefficient increment at takeoff due to flaps',
)

add_meta_data(
    Mission.Takeoff.LIFT_COEFFICIENT_MAX,
    meta_data=_MetaData,
    historical_name={
        'GASP': 'INGASP.CLMWTO',
        # ['&DEFINE.AERIN.CLTOM', 'LANDG.CLTOM', '~DEFTOL.CLTOA'],
        'FLOPS': 'AERIN.CLTOM',
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.max_takeoff_lift_coeff',
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
        'LEAPS1': '(SimpleTakeoff)self.lift_over_drag_ratio',
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
    #     'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.obstacle_height'},
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
        'LEAPS1': [
            'aircraft.inputs.L0_takeoff_and_landing.rolling_mu',
            '(GroundRoll)self.mu',
            '(Rotate)self.mu',
            '(GroundBrake)self.rolling_mu',
        ],
    },
    units='unitless',
    desc='coefficient of rolling friction for groundroll portion of takeoff',
    default_value=0.025,
)

add_meta_data(
    Mission.Takeoff.ROTATION_SPEED_INCREMENT,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.DVR', 'FLOPS': None, 'LEAPS1': None},
    units='kn',
    desc='increment of takeoff rotation speed above engine failure decision speed',
    default_value=5,
)

add_meta_data(
    Mission.Takeoff.ROTATION_VELOCITY,
    meta_data=_MetaData,
    historical_name={'GASP': 'INGASP.VR', 'FLOPS': None, 'LEAPS1': None},
    units='kn',
    desc='rotation velocity',
)

add_meta_data(
    Mission.Takeoff.SPOILER_DRAG_COEFFICIENT,
    meta_data=_MetaData,
    historical_name={
        'GASP': None,
        'FLOPS': 'TOLIN.CDSPOL',  # '&DEFTOL.TOLIN.CDSPOL',
        'LEAPS1': None,
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
        'LEAPS1': None,
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
        'LEAPS1': 'aircraft.inputs.L0_takeoff_and_landing.thrust_incidence_angle',
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
    historical_name={'GASP': 'INGASP.DELTT', 'FLOPS': None, 'LEAPS1': None},
    units='h',
    desc='time spent taxiing before takeoff',
    option=True,
    default_value=0.167,
)

add_meta_data(
    Mission.Taxi.MACH,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    desc="Sets which legacy code's methods will be used for aerodynamics estimation",
    option=True,
    types=LegacyCode,
    default_value=None,
)

add_meta_data(
    Settings.EQUATIONS_OF_MOTION,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    desc='Sets which equations of motion Aviary will use in mission analysis',
    option=True,
    types=EquationsOfMotion,
    default_value=None,
)

add_meta_data(
    Settings.MASS_METHOD,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    desc="Sets which legacy code's methods will be used for mass estimation",
    option=True,
    types=LegacyCode,
    default_value=None,
)

add_meta_data(
    Settings.PAYLOAD_RANGE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    units='unitless',
    option=True,
    default_value=False,
    types=bool,
    desc='if True, run a set of off-design missions to create a payload range diagram.',
)

add_meta_data(
    Settings.PROBLEM_TYPE,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
    desc="Select from Aviary's built in problem types: Sizing, Alternate, and Fallout",
    option=True,
    types=ProblemType,
    default_value=None,
)

add_meta_data(
    Settings.VERBOSITY,
    meta_data=_MetaData,
    historical_name={'GASP': None, 'FLOPS': None, 'LEAPS1': None},
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
