'''
Define meta data associated with variables in the disciplinary data hierarchy.
'''
from copy import deepcopy
import numpy as np

import aviary.api as av
from aviary.examples.variables_extension import Aircraft, Mission

# ---------------------------
# Meta data associated with variables in the aircraft data hierarchy.
# Please add variables in alphabetical order to match the order in the
# aircraft data hierarchy.
#
# ASCII art from http://patorjk.com/software/taag/#p=display&h=0&f=Big&t=
# Super categories such as aircraft and mission are in 'Blocks' font
# Sub categories such as AntiIcing and Wing are in 'Big' font
# ---------------------------

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

av.add_meta_data(
    Aircraft.CG,
    units='ft',
    desc='Center of gravity',
    default_value=np.zeros(3),
    meta_data=ExtendedMetaData,
    # note that VSP_example is not a real code and center_of_gravity is not a real variable name. These are here to show how the historical_name argument can be used.
    historical_name={'VSP_example': 'center_of_gravity'},
)

av.add_meta_data(
    Aircraft.MASS,
    units='lbm',
    desc='Total aircraft mass.',
    default_value=1.,
    meta_data=ExtendedMetaData
)

#  _    _                  _                         _             _   _______           _   _
# | |  | |                (_)                       | |           | | |__   __|         (_) | |
# | |__| |   ___    _ __   _   ____   ___    _ __   | |_    __ _  | |    | |      __ _   _  | |
# |  __  |  / _ \  | '__| | | |_  /  / _ \  | '_ \  | __|  / _` | | |    | |     / _` | | | | |
# | |  | | | (_) | | |    | |  / /  | (_) | | | | | | |_  | (_| | | |    | |    | (_| | | | | |
# |_|  |_|  \___/  |_|    |_| /___|  \___/  |_| |_|  \__|  \__,_| |_|    |_|     \__,_| |_| |_|
# =============================================================================================

av.add_meta_data(
    Aircraft.HorizontalTail.MEAN_AERO_CHORD,
    units='ft',
    desc='Mean aerodynamic chord.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.HorizontalTail.Elevator.AREA,
    units='ft**2',
    desc='Area of each elevator element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.HorizontalTail.Elevator.ROOT_CHORD,
    units='ft',
    desc='Root chord of each elevator element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.HorizontalTail.Elevator.SPAN,
    units='ft',
    desc='Span of each elevator element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

#       _
#      | |
#      | |  _   _   _ __   _   _
#  _   | | | | | | | '__| | | | |
# | |__| | | |_| | | |    | |_| |
#  \____/   \__,_| |_|     \__, |
#                           __/ |
#                          |___/
# ===============================

av.add_meta_data(
    Aircraft.Jury.MASS,
    units='kg',
    desc='mass of the aircraft`s jury',
    default_value=0,
    meta_data=ExtendedMetaData
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

av.add_meta_data(
    Aircraft.LandingGear.MAIN_GEAR_OLEO_DIAMETER,
    units='ft',
    desc='Main gear oleo diameter',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

# __      __                _     _                  _   _______           _   _
# \ \    / /               | |   (_)                | | |__   __|         (_) | |
#  \ \  / /    ___   _ __  | |_   _    ___    __ _  | |    | |      __ _   _  | |
#   \ \/ /    / _ \ | '__| | __| | |  / __|  / _` | | |    | |     / _` | | | | |
#    \  /    |  __/ | |    | |_  | | | (__  | (_| | | |    | |    | (_| | | | | |
#     \/      \___| |_|     \__| |_|  \___|  \__,_| |_|    |_|     \__,_| |_| |_|
# ===============================================================================

av.add_meta_data(
    Aircraft.VerticalTail.MEAN_AERO_CHORD,
    units='ft',
    desc='Mean aerodynamic chord.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.VerticalTail.Rudder.AREA,
    units='ft**2',
    desc='Area of each rudder element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.VerticalTail.Rudder.ROOT_CHORD,
    units='ft',
    desc='Root chord of each rudder element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.VerticalTail.Rudder.SPAN,
    units='ft',
    desc='Span of each rudder element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
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
    Aircraft.Wing.AERO_CENTER,
    units='ft',
    desc='aerodynamic center.',
    default_value=np.zeros(3),
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.CHORD,
    units='ft',
    desc='Reference chord.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.Flap.AREA,
    units='ft**2',
    desc='Area of each flap element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.Flap.ROOT_CHORD,
    units='ft',
    desc='Root chord of each flap element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.Flap.SPAN,
    units='ft',
    desc='Span of each flap element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.Krueger.AREA,
    units='ft**2',
    desc='Area of each Krueger element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.Krueger.ROOT_CHORD,
    units='ft',
    desc='Root chord of each krueger element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Wing.Krueger.SPAN,
    units='ft',
    desc='Span of each krueger element.',
    default_value=0.0,
    meta_data=ExtendedMetaData
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

#   _____                  _
#  / ____|                (_)
# | |       _ __   _   _   _   ___    ___
# | |      | '__| | | | | | | / __|  / _ \
# | |____  | |    | |_| | | | \__ \ |  __/
#  \_____| |_|     \__,_| |_| |___/  \___|
# ========================================

av.add_meta_data(
    Mission.Cruise.FUEL_MASS,
    units='lbm',
    desc='Fuel mass states along cruise phase',
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Mission.Cruise.MACH,
    units='unitless',
    desc='Mach number states along cruise phase',
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Mission.Cruise.MASS,
    units='lbm',
    desc='Gross mass states along cruise phase',
    default_value=None,
    meta_data=ExtendedMetaData
)

#  _______                  _
# |__   __|                (_)
#    | |      __ _  __  __  _
#    | |     / _` | \ \/ / | |
#    | |    | (_| |  >  <  | |
#    |_|     \__,_| /_/\_\ |_|
# ============================

av.update_meta_data(
    Mission.Taxi.DURATION,
    units='h',
    desc='I am changing the description of this variable to demonstrate the update_meta_data function',
    default_value=0.167,
    meta_data=ExtendedMetaData
    # We use the ExtendedMetaData because this is the metadata we want to edit.
    # Despite the fact that we never added Mission.Taxi.DURATION to the ExtendedMetaData,
    # the variable is already there because it exists in the av.CoreMetaData
)
