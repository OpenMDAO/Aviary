from aviary.variable_info.enums import SpeedType
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.enums import LegacyCode

GASP = LegacyCode.GASP


throttle_max = 1.0
throttle_climb = 0.956
throttle_cruise = 0.930
throttle_idle = 0.0

prop = CorePropulsionBuilder('core_propulsion', BaseMetaData)
mass = CoreMassBuilder('core_mass', BaseMetaData, GASP)
aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, GASP)
geom = CoreGeometryBuilder('core_geometry', BaseMetaData, GASP)

default_premission_subsystems = [prop, geom, aero, mass]
default_mission_subsystems = [aero, prop]

phase_info = {
    'groundroll': {
        'num_segments': 3,
        'throttle_setting': throttle_max,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': True,
        'clean': False,
        'initial_ref': (100., 'kn'),
        'initial_bounds': ((0., 500.), 'kn'),
        'duration_ref': (100., 'kn'),
        'duration_bounds': ((50., 1000.), 'kn'),
        'control_order': 1,
        'order': 3,
        'opt': True,
    },
    'rotation': {
        'num_segments': 2,
        'throttle_setting': throttle_max,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': True,
        'clean': False,
        'initial_ref': (1.e3, 'm'),
        'initial_bounds': ((50., 5000.), 'm'),
        'duration_ref': (1.e3, 'm'),
        'duration_bounds': ((50., 2000.), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': True,
    },
    'ascent_to_gear_retract': {
        'num_segments': 2,
        'throttle_setting': throttle_max,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': False,
        'clean': False,
        'initial_ref': (1.e3, 'm'),
        'initial_bounds': ((10., 2.e3), 'm'),
        'duration_ref': (1.e3, 'm'),
        'duration_bounds': ((500., 1.e4), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': False,
    },
    'ascent_to_flap_retract': {
        'num_segments': 2,
        'throttle_setting': throttle_max,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': False,
        'clean': False,
        'initial_ref': (1.e3, 'm'),
        'initial_bounds': ((10., 2.e5), 'm'),
        'duration_ref': (1.e3, 'm'),
        'duration_bounds': ((500., 1.e4), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': False,
    },
    'ascent': {
        'num_segments': 2,
        'throttle_setting': throttle_max,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e3, 'm'),
        'initial_bounds': ((10., 2.e5), 'm'),
        'duration_ref': (1.e3, 'm'),
        'duration_bounds': ((500., 1.e4), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': False,
    },
    'climb_at_constant_TAS': {
        'num_segments': 2,
        'throttle_setting': throttle_max,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e3, 'm'),
        'initial_bounds': ((10., 2.e5), 'm'),
        'duration_ref': (1.e3, 'm'),
        'duration_bounds': ((500., 1.e4), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': False,
    },
    'climb_at_constant_EAS': {
        'num_segments': 2,
        'throttle_setting': throttle_climb,
        'input_speed_type': SpeedType.EAS,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e5, 'm'),
        'initial_bounds': ((100., 1.e7), 'm'),
        'duration_ref': (1.e5, 'm'),
        'duration_bounds': ((1.e4, 5.e4), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': False,
        'fixed_EAS': 250.,
    },
    'climb_at_constant_EAS_to_mach': {
        'num_segments': 2,
        'throttle_setting': throttle_climb,
        'input_speed_type': SpeedType.EAS,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e5, 'm'),
        'initial_bounds': ((100., 1.e7), 'm'),
        'duration_ref': (1.e5, 'm'),
        'duration_bounds': ((10.e3, 1.e6), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': True,
        'fixed_EAS': 270.,
    },
    'climb_at_constant_mach': {
        'num_segments': 2,
        'throttle_setting': throttle_climb,
        'input_speed_type': SpeedType.MACH,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e5, 'm'),
        'initial_bounds': ((100., 1.e7), 'm'),
        'duration_ref': (1.e5, 'm'),
        'duration_bounds': ((10.e3, 1.e6), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': True,
    },
    'cruise': {
        'num_segments': 5,
        'throttle_setting': throttle_cruise,
        'input_speed_type': SpeedType.MACH,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e6, 'm'),
        'initial_bounds': ((1.e4, 2.e7), 'm'),
        'duration_ref': (5.e6, 'm'),
        'duration_bounds': ((0.1e7, 3.e7), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': False,
    },
    'descent': {
        'num_segments': 3,
        'throttle_setting': throttle_idle,
        'input_speed_type': SpeedType.TAS,
        'ground_roll': False,
        'clean': True,
        'initial_ref': (1.e6, 'm'),
        'initial_bounds': ((100., 1.e7), 'm'),
        'duration_ref': (1.e5, 'm'),
        'duration_bounds': ((1.5e5, 2.e5), 'm'),
        'control_order': 1,
        'order': 3,
        'opt': True,
    },
}
