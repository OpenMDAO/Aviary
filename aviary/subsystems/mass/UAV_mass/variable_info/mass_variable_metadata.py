"""
Metadata for UAV variables defined in mass_variables.py
"""
from copy import deepcopy
import aviary.api as av
from aviary.subsystems.mass.UAV_mass.variable_info.enums import WingType
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

ExtendedMetaData = deepcopy(av.CoreMetaData)

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

#  ______                        _
# |  ____|                      | |
# | |__     _   _   ___    ___  | |   __ _    __ _    ___
# |  __|   | | | | / __|  / _ \ | |  / _` |  / _` |  / _ \
# | |      | |_| | \__ \ |  __/ | | | (_| | | (_| | |  __/
# |_|       \__,_| |___/  \___| |_|  \__,_|  \__, |  \___|
#                                             __/ |
#                                            |___/
# ========================================================

av.add_meta_data(
    Aircraft.Fuselage.AREAL_SKIN_DENSITY,
    units='kg/m**2',
    desc='Areal density of fuselage skin',
    default_value=0.08,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.AVG_HEIGHT,
    units='m',
    desc='UAV Height of fuselage (assumed rectangular prism shape)',
    default_value=0.3,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.AVG_WIDTH,
    units='m',
    desc='Width of fuselage (assumed rectangular prism shape)',
    default_value=0.3,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.BULKHEAD_DENSITY,
    units='kg/m**3',
    types=float,
    desc='Material density of the rib',
    default_value=[0.0],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.BULKHEAD_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the rib area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.BULKHEAD_MATERIALS,
    units='unitless',
    types=str,
    desc='Material density of the bulkhead',
    default_value=[
        'Balsa', 'Balsa', 'Balsa', 'Balsa', 'Balsa',
        'Balsa', 'Balsa', 'Balsa', 'Balsa', 'Balsa',
        'Balsa', 'Balsa', 'Balsa', 'Balsa', 'Ply',
        'Ply', 'Ply', 'Ply', 'Ply', 'Ply',
    ],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.BULKHEAD_THICKNESS,
    units='m',
    types=float,
    desc='Thickness of a single rib',
    default_value=[
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
        0.003175, 0.003175, 0.003175, 0.003175, 0.00635,
    ],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.FLOOR_DENSITY,
    units='kg/m**3',
    desc='Density of fuselage floor',
    default_value=0.02,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.FLOOR_LENGTH,
    units='m',
    desc='length of fuselage floor',
    default_value=0.7,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.FLOOR_THICKNESS,
    units='m',
    desc='Thickness of fuselage floor',
    default_value=0.003,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.GLUE_FACTOR,
    units='unitless',
    desc='Added margin for glue. Only added to ribs, spars, and stringers',
    default_value=0.15,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.MISC_MASS,
    units='kg',
    desc='Mass made up of smaller, non structural components. Can be used for higher fidelity options as well',
    default_value=0.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.NUM_BULKHEADS,
    units='unitless',
    desc='Number of fuselage ribs',
    default_value=10,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.NUM_SPARS,
    units='unitless',
    desc='Number of fuselage spars',
    default_value=2,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SHEETING_COVERAGE,
    units='unitless',
    desc='Fraction of the wetted area covered by sheeting',
    default_value=1.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SHEETING_DENSITY,
    units='kg/m**3',
    desc='Material density of the sheeting',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SHEETING_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the sheeting area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SHEETING_THICKNESS,
    units='m',
    desc='Thickness of sheeting',
    default_value=0.003,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SPAR_DENSITY,
    units='kg/m**3',
    desc='Material density of the spar',
    default_value=1500,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SPAR_OUTER_DIAMETER,
    units='m',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.SPAR_WALL_THICKNESS,
    units='m',
    desc='Thickness of spar wall',
    default_value=0.002,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.STRINGER_DENSITY,
    units='kg/m**3',
    desc='Material density of the stringer',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Fuselage.STRINGER_THICKNESS,
    units='m',
    desc='Thickness of stringers',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

#  _    _                  _                         _             _   _______           _   _
# | |  | |                (_)                       | |           | | |__   __|         (_) | |
# | |__| |   ___    _ __   _   ____   ___    _ __   | |_    __ _  | |    | |      __ _   _  | |
# |  __  |  / _ \  | '__| | | |_  /  / _ \  | '_ \  | __|  / _` | | |    | |     / _` | | | | |
# | |  | | | (_) | | |    | |  / /  | (_) | | | | | | |_  | (_| | | |    | |    | (_| | | | | |
# |_|  |_|  \___/  |_|    |_| /___|  \___/  |_| |_|  \__|  \__,_| |_|    |_|     \__,_| |_| |_|
# =============================================================================================

av.add_meta_data(
    Aircraft.HorizontalTail.AREAL_SKIN_DENSITY,
    units='kg/m**2',
    desc='Areal density of horizontal tail skin',
    default_value=0.08,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.AIRFOIL_PATH,
    units='unitless',
    types=str,
    desc='Path to csv file containing airfoil data',
    default_value='subsystems/mass/UAV_mass/utils/n0012-il.csv',
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.GLUE_FACTOR,
    units='unitless',
    desc='Added margin for glue. Only added to ribs, spars, and stringers',
    default_value=0.15,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.MISC_MASS,
    units='kg',
    desc='Mass made up of smaller, non structural components. Can be used for higher fidelity options as well',
    default_value=0.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.NUM_SPARS,
    units='unitless',
    desc='Number of wing spars',
    default_value=2,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.NUM_STRINGERS,
    units='unitless',
    desc='Number of stringers(assumed length=span)',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.RIB_DENSITY,
    units='kg/m**3',
    types=float,
    desc='Material density of the rib',
    default_value=[0.0],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.RIB_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the rib area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.RIB_MATERIALS,
    units='unitless',
    types=str,
    desc='Material of the rib',
    default_value=['Balsa'],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.RIB_THICKNESS,
    units='m',
    types=float,
    desc='Thickness of a single rib',
    default_value=[0.00635],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)



av.add_meta_data(
    Aircraft.HorizontalTail.SHEETING_COVERAGE,
    units='unitless',
    desc='Fraction of the wetted area covered by sheeting',
    default_value=1.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SHEETING_DENSITY,
    units='kg/m**3',
    desc='Material density of the sheeting',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SHEETING_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the sheeting area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SHEETING_THICKNESS,
    units='m',
    desc='Thickness of sheeting',
    default_value=0.003,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SPAR_DENSITY,
    units='kg/m**3',
    desc='Material density of the spar',
    default_value=1500,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SPAR_OUTER_DIAMETER,
    units='m',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SPAR_WALL_THICKNESS,
    units='m',
    desc='Thickness of spar wall',
    default_value=0.002,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.STRINGER_DENSITY,
    units='kg/m**3',
    desc='Material density of the stringer',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.HorizontalTail.STRINGER_THICKNESS,
    units='m',
    desc='Thickness of stringers',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

# __      __                _     _                  _   _______           _   _
# \ \    / /               | |   (_)                | | |__   __|         (_) | |
#  \ \  / /    ___   _ __  | |_   _    ___    __ _  | |    | |      __ _   _  | |
#   \ \/ /    / _ \ | '__| | __| | |  / __|  / _` | | |    | |     / _` | | | | |
#    \  /    |  __/ | |    | |_  | | | (__  | (_| | | |    | |    | (_| | | | | |
#     \/      \___| |_|     \__| |_|  \___|  \__,_| |_|    |_|     \__,_| |_| |_|
# ===============================================================================

av.add_meta_data(
    Aircraft.VerticalTail.AREAL_SKIN_DENSITY,
    units='kg/m**2',
    desc='Areal density of vertical tail skin',
    default_value=0.08,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.AIRFOIL_PATH,
    units='unitless',
    types=str,
    desc='Path to csv file containing airfoil data',
    default_value='subsystems/mass/UAV_mass/utils/n0012-il.csv',
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.GLUE_FACTOR,
    units='unitless',
    desc='Added margin for glue. Only added to ribs, spars, and stringers',
    default_value=0.15,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.MISC_MASS,
    units='kg',
    desc='Mass made up of smaller, non structural components. Can be used for higher fidelity options as well',
    default_value=0.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.NUM_SPARS,
    units='unitless',
    desc='Number of wing spars',
    default_value=2,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.NUM_STRINGERS,
    units='unitless',
    desc='Number of stringers(assumed length=span)',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.RIB_DENSITY,
    units='kg/m**3',
    types=float,
    desc='Material density of the rib',
    default_value=[0.0],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.RIB_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the rib area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.RIB_MATERIALS,
    units='unitless',
    types=str,
    desc='Material of the rib',
    default_value=['Balsa'],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.RIB_THICKNESS,
    units='m',
    types=float,
    desc='Thickness of a single rib',
    default_value=[0.00635],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SHEETING_COVERAGE,
    units='unitless',
    desc='Fraction of the wetted area covered by sheeting',
    default_value=1.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SHEETING_DENSITY,
    units='kg/m**3',
    desc='Material density of the sheeting',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SHEETING_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the sheeting area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SHEETING_THICKNESS,
    units='m',
    desc='Thickness of sheeting',
    default_value=0.003,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SPAR_DENSITY,
    units='kg/m**3',
    desc='Material density of the spar',
    default_value=1500,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SPAR_OUTER_DIAMETER,
    units='m',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.SPAR_WALL_THICKNESS,
    units='m',
    desc='Thickness of spar wall',
    default_value=0.002,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.STRINGER_DENSITY,
    units='kg/m**3',
    desc='Material density of the stringer',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.VerticalTail.STRINGER_THICKNESS,
    units='m',
    desc='Thickness of stringers',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
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

av.add_meta_data(
    Aircraft.Wing.AREAL_SKIN_DENSITY,
    units='kg/m**2',
    desc='Areal density of wing skin',
    default_value=0.08,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.AIRFOIL_PATH,
    units='unitless',
    types=str,
    desc='Path to csv file containing airfoil data',
    default_value='subsystems/mass/UAV_mass/utils/mh84-il.csv',
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.FOAM_DENSITY,
    units='kg/m**3',
    desc='density of foam used for simple wing design',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True
)

av.add_meta_data(
    Aircraft.Wing.GLUE_FACTOR,
    units='unitless',
    desc='Added margin for glue. Only added to ribs, spars, and stringers',
    default_value=0.15,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.NUM_SPARS,
    units='unitless',
    desc='Number of wing spars',
    default_value=2,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.NUM_STRINGERS,
    units='unitless',
    desc='Number of stringers(assumed length=span)',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.RIB_DENSITY,
    units='kg/m**3',
    types=float,
    desc='Material density of the rib',
    default_value=[0.0],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.RIB_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the rib area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.RIB_MATERIALS,
    units='unitless',
    types=str,
    desc='Material of the rib',
    default_value=[
        'Balsa', 'Balsa', 'Balsa', 'Balsa', 'Balsa',
        'Balsa', 'Balsa', 'Balsa', 'Balsa', 'Balsa',
        'Balsa', 'Balsa', 'Balsa', 'Balsa', 'Balsa',
        'Ply', 'Ply', 'Ply', 'Ply', 'Ply',
    ],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.RIB_THICKNESS,
    units='m',
    types=float,
    desc='Thickness of a single rib',
    default_value=[
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
        0.003175, 0.003175, 0.003175, 0.003175, 0.003175,
    ],
    meta_data = ExtendedMetaData,
    multivalue=True,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.ROD_DENSITY,
    units='kg/m**3',
    desc='density of rod used for simple wing design',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True
)

av.add_meta_data(
    Aircraft.Wing.ROD_RADIUS,
    units='m',
    desc='radius of rod for simple wing',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True
)

av.add_meta_data(
    Aircraft.Wing.ROD_THICKNESS,
    units='m',
    desc='thickness of the rod in simple wing design',
    default_value=2.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SHEETING_COVERAGE,
    units='unitless',
    desc='Fraction of the wetted area covered by sheeting',
    default_value=1.0,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SHEETING_DENSITY,
    units='kg/m**3',
    desc='Material density of the sheeting',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SHEETING_LIGHTENING_FACTOR,
    units='unitless',
    desc='Fraction of the sheeting area that remains after lightening cuts',
    default_value=0.5,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SHEETING_THICKNESS,
    units='m',
    desc='Thickness of sheeting',
    default_value=0.003,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SPAR_DENSITY,
    units='kg/m**3',
    desc='Material density of the spar',
    default_value=1500,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SPAR_OUTER_DIAMETER,
    units='m',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.SPAR_WALL_THICKNESS,
    units='m',
    desc='Thickness of spar wall',
    default_value=0.002,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.STRINGER_DENSITY,
    units='kg/m**3',
    desc='Material density of the stringer',
    default_value=250,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.STRINGER_THICKNESS,
    units='m',
    desc='Thickness of stringers',
    default_value=0.005,
    meta_data = ExtendedMetaData,
    option=True,
)

av.add_meta_data(
    Aircraft.Wing.TYPE,
    meta_data = ExtendedMetaData,
    units = 'unitless',
    desc = 'Specifies "simple" or "medium" wing design',
    default_value = WingType.SIMPLE,
    option = True,
    types = WingType,
)