import numpy as np

from aviary.variable_info.enums import SpeedType
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData, Mission
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

mission_distance = 3675

phase_info = {
    'groundroll': {
        'num_segments': 1,
        'order': 3,
        'connect_initial_mass': False,
        'fix_initial': True,
        'fix_initial_mass': False,
        'duration_ref': (50., 's'),
        'duration_bounds': ((1., 100.), 's'),
        'TAS_lower': (0, 'kn'),
        'TAS_upper': (1000, 'kn'),
        'TAS_ref': (150, 'kn'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (0, 'ft'),
        'distance_upper': (10.e3, 'ft'),
        'distance_ref': (3000, 'ft'),
        'distance_defect_ref': (3000, 'ft'),
        'throttle_setting': throttle_max,
        'initial_guesses': {
            'times': ([0.0, 40.0], 's'),
            'TAS': ([0.066, 143.1], 'kn'),
            'distance': ([0.0, 1000.], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'rotation': {
        'num_segments': 1,
        'order': 3,
        'fix_initial': False,
        'duration_bounds': ((1, 100), 's'),
        'duration_ref': (50.0, 's'),
        'TAS_lower': (0, 'kn'),
        'TAS_upper': (1000, 'kn'),
        'TAS_ref': (100, 'kn'),
        'TAS_ref0': (0, 'kn'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (0, 'ft'),
        'distance_upper': (10_000, 'ft'),
        'distance_ref': (5000, 'ft'),
        'distance_defect_ref': (5000, 'ft'),
        'angle_lower': (0., 'deg'),
        'angle_upper': (5., 'deg'),
        'angle_ref': (5., 'deg'),
        'angle_defect_ref': (5., 'deg'),
        'normal_ref': (10000, 'lbf'),
        'throttle_setting': throttle_max,
        'initial_guesses': {
            'times': ([40.0, 5.0], 's'),
            'alpha': ([0.0, 2.5], 'deg'),
            'TAS': ([143, 150.], 'kn'),
            'distance': ([3680.37217765, 4000], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'ascent': {
        'num_segments': 4,
        'order': 3,
        'fix_initial': False,
        'TAS_lower': (0, 'kn'),
        'TAS_upper': (700, 'kn'),
        'TAS_ref': (200, 'kn'),
        'TAS_ref0': (0, 'kn'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (0, 'ft'),
        'distance_upper': (15_000, 'ft'),
        'distance_ref': (1e4, 'ft'),
        'distance_defect_ref': (1e4, 'ft'),
        'alt_lower': (0, 'ft'),
        'alt_upper': (700, 'ft'),
        'alt_ref': (1000, 'ft'),
        'alt_defect_ref': (1000, 'ft'),
        'alt_constraint_eq': (500, 'ft'),
        'alt_constraint_ref': (500, 'ft'),
        'alt_constraint_ref0': (0, 'ft'),
        'angle_lower': (-10, 'deg'),
        'angle_upper': (20, 'deg'),
        'angle_ref': (1, 'deg'),
        'angle_defect_ref': (1.0, 'deg'),
        'pitch_constraint_lower': (0., 'deg'),
        'pitch_constraint_upper': (15., 'deg'),
        'pitch_constraint_ref': (1., 'deg'),
        'throttle_setting': throttle_max,
        'initial_guesses': {
            'times': ([45., 25.], 's'),
            'flight_path_angle': ([0.0, 8.], 'deg'),
            'alpha': ([2.5, 1.5], 'deg'),
            'TAS': ([150., 185.], 'kn'),
            'distance': ([4.e3, 10.e3], 'ft'),
            'altitude': ([0.0, 500.], 'ft'),
            'tau_gear': (0.2, None),
            'tau_flaps': (0.9, None),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'accel': {
        'num_segments': 1,
        'order': 3,
        'fix_initial': False,
        'alt': (500, 'ft'),
        'EAS_constraint_eq': (250, 'kn'),
        'time_initial_bounds': ((5, 1000), 's'),
        'duration_bounds': ((1, 200), 's'),
        'duration_ref': (1000, 's'),
        'TAS_lower': (150, 'kn'),
        'TAS_upper': (270, 'kn'),
        'TAS_ref': (250, 'kn'),
        'TAS_ref0': (150, 'kn'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (0, 'NM'),
        'distance_upper': (150, 'NM'),
        'distance_ref': (5, 'NM'),
        'distance_defect_ref': (5, 'NM'),
        'throttle_setting': throttle_max,
        'initial_guesses': {
            'times': ([70., 13.], 's'),
            'TAS': ([185., 250.], 'kn'),
            'distance': ([10.e3, 20.e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'climb1': {
        'num_segments': 1,
        'order': 3,
        'fix_initial': False,
        'EAS_target': (250, 'kn'),
        'mach_cruise': 0.8,
        'target_mach': False,
        'final_alt': (10.e3, 'ft'),
        'time_initial_bounds': ((20, 200), 's'),
        'duration_bounds': ((30, 300), 's'),
        'duration_ref': (1000, 's'),
        'alt_lower': (400, 'ft'),
        'alt_upper': (11_000, 'ft'),
        'alt_ref': (10.e3, 'ft'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (0, 'NM'),
        'distance_upper': (500, 'NM'),
        'distance_ref': (10, 'NM'),
        'distance_ref0': (0, 'NM'),
        'throttle_setting': throttle_climb,
        'initial_guesses': {
            'times': ([1., 2.], 'min'),
            'distance': ([20.e3, 100.e3], 'ft'),
            'altitude': ([500., 10.e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'climb2': {
        'num_segments': 3,
        'order': 3,
        'fix_initial': False,
        'EAS_target': (270, 'kn'),
        'mach_cruise': 0.8,
        'target_mach': True,
        'final_alt': (37.5e3, 'ft'),
        'required_available_climb_rate': (0.1, 'ft/min'),
        'time_initial_bounds': ((10, 700), 's'),
        'duration_bounds': ((200, 17_000), 's'),
        'duration_ref': (5000, 's'),
        'alt_lower': (9000, 'ft'),
        'alt_upper': (40000, 'ft'),
        'alt_ref': (30000, 'ft'),
        'alt_ref0': (0, 'ft'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (10, 'NM'),
        'distance_upper': (1000, 'NM'),
        'distance_ref': (500, 'NM'),
        'distance_ref0': (0, 'NM'),
        'distance_defect_ref': (500, 'NM'),
        'throttle_setting': throttle_climb,
        'initial_guesses': {
            'times': ([216., 1300.], 's'),
            'distance': ([100.e3, 200.e3], 'ft'),
            'altitude': ([10.e3, 37.5e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'cruise': {
        'initial_guesses': {
            # [Initial mass, delta mass] for special cruise phase.
            'mass': ([171481., -35000], 'lbm'),
            'initial_distance': (200.e3, 'ft'),
            'initial_time': (1516., 's'),
            'altitude': (37.5e3, 'ft'),
            'mach': (0.8, 'unitless'),
        }
    },
    'desc1': {
        'num_segments': 3,
        'order': 3,
        'fix_initial': False,
        'input_initial': False,
        'EAS_limit': (350, 'kn'),
        'mach_cruise': 0.8,
        'input_speed_type': SpeedType.MACH,
        'final_alt': (10.e3, 'ft'),
        'time_initial_bounds': ((1000, 35_000), 's'),
        'time_initial_ref': (10_000, 's'),
        'duration_bounds': ((300., 900.), 's'),
        'duration_ref': (1000, 's'),
        'alt_lower': (1000, 'ft'),
        'alt_upper': (40_000, 'ft'),
        'alt_ref': (30_000, 'ft'),
        'alt_ref0': (0, 'ft'),
        'alt_constraint_ref': (10000, 'ft'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (140_000, 'lbm'),
        'mass_ref0': (0, 'lbm'),
        'mass_defect_ref': (140_000, 'lbm'),
        'distance_lower': (3000., 'NM'),
        'distance_upper': (5000., 'NM'),
        'distance_ref': (mission_distance, 'NM'),
        'distance_ref0': (0, 'NM'),
        'distance_defect_ref': (100, 'NM'),
        'throttle_setting': throttle_idle,
        'initial_guesses': {
            'mass': (136000., 'lbm'),
            'altitude': ([37.5e3, 10.e3], 'ft'),
            'throttle': ([0.0, 0.0], 'unitless'),
            'distance': ([.92*mission_distance, .96*mission_distance], 'NM'),
            'times': ([28000., 500.], 's'),
        }
    },
    'desc2': {
        'num_segments': 1,
        'order': 7,
        'fix_initial': False,
        'input_initial': False,
        'EAS_limit': (250, 'kn'),
        'mach_cruise': 0.80,
        'input_speed_type': SpeedType.EAS,
        'final_alt': (1000, 'ft'),
        'time_initial_bounds': ((2000, 50_000), 's'),
        'time_initial_ref': (15000, 's'),
        'duration_bounds': ((100., 5000), 's'),
        'duration_ref': (500, 's'),
        'alt_lower': (500, 'ft'),
        'alt_upper': (11_000, 'ft'),
        'alt_ref': (10.e3, 'ft'),
        'alt_ref0': (1000, 'ft'),
        'alt_constraint_ref': (1000, 'ft'),
        'mass_lower': (0, 'lbm'),
        'mass_upper': (None, 'lbm'),
        'mass_ref': (150_000, 'lbm'),
        'mass_defect_ref': (150_000, 'lbm'),
        'distance_lower': (0, 'NM'),
        'distance_upper': (5000, 'NM'),
        'distance_ref': (3500, 'NM'),
        'distance_defect_ref': (100, 'NM'),
        'throttle_setting': throttle_idle,
        'initial_guesses': {
            'mass': (136000., 'lbm'),
            'altitude': ([10.e3, 1.e3], 'ft'),
            'throttle': ([0., 0.], 'unitless'),
            'distance': ([.96*mission_distance, mission_distance], 'NM'),
            'times': ([28500., 500.], 's'),
        }
    },
}


def phase_info_parameterization(phase_info, aviary_inputs):
    """
    Modify the values in the phase_info dictionary to accomodate different values
    for the following mission design inputs: cruise altitude, cruise mach number,
    cruise range, design gross mass.

    Parameters
    ----------
    phase_info : dict
        Dictionary of phase settings for a mission profile
    aviary_inputs : <AviaryValues>
        Object containing values and units for all aviary inputs and options

    Returns
    -------
    dict
        Modified phase_info that has been changed to match the new mission
        parameters
    """

    range_cruise = aviary_inputs.get_val(Mission.Design.RANGE, units='NM')
    alt_cruise = aviary_inputs.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft')
    gross_mass = aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')
    mach_cruise = aviary_inputs.get_val(Mission.Design.MACH)

    # Range
    old_range_cruise = phase_info['desc2']['initial_guesses']['distance'][0][1]
    range_scale = 1.0
    if range_cruise != old_range_cruise:

        phase_info['desc1']['initial_guesses']['distance'] = \
            ([.92*range_cruise, .96*range_cruise], 'NM')
        phase_info['desc2']['initial_guesses']['distance'] = \
            ([.96*range_cruise, range_cruise], 'NM')
        range_scale = range_cruise / old_range_cruise

    # Altitude
    old_alt_cruise = phase_info['climb2']['final_alt'][0]
    if alt_cruise != old_alt_cruise:

        phase_info['climb2']['final_alt'] = (alt_cruise, 'ft')
        phase_info['climb2']['initial_guesses']['altitude'] = ([10.e3, alt_cruise], 'ft')
        phase_info['cruise']['initial_guesses']['altitude'] = (alt_cruise, 'ft')
        phase_info['desc1']['initial_guesses']['altitude'] = ([alt_cruise, 10.e3], 'ft')

        # TODO - Could adjust time guesses/bounds in climb2 and desc2.

    # Mass
    old_gross_mass = 175400.0
    if gross_mass != old_gross_mass:

        # Note, this requires that the guess for gross mass is pretty close to the
        # compute mass.

        fuel_used = 35000 * range_scale
        phase_info['groundroll']['initial_guesses']['mass'] = \
            ([gross_mass, gross_mass], 'lbm')
        phase_info['rotation']['initial_guesses']['mass'] = \
            ([gross_mass, gross_mass], 'lbm')
        phase_info['accel']['initial_guesses']['mass'] = \
            ([gross_mass, gross_mass], 'lbm')
        phase_info['ascent']['initial_guesses']['mass'] = \
            ([gross_mass, gross_mass], 'lbm')

        phase_info['cruise']['initial_guesses']['mass'] = \
            ([gross_mass, -fuel_used], 'lbm')

        end_mass = gross_mass - fuel_used
        phase_info['desc1']['initial_guesses']['mass'] = (end_mass, 'lbm')
        phase_info['desc2']['initial_guesses']['mass'] = (end_mass, 'lbm')

    # Mach
    old_mach_cruise = phase_info['cruise']['initial_guesses']['mach'][0]
    if mach_cruise != old_mach_cruise:

        phase_info['cruise']['initial_guesses']['mach'] = (mach_cruise, 'unitless')

    return phase_info
