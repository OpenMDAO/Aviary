from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Mission

# defaults for 2DOF based phases
mission_distance = 3675

phase_info = {
    'groundroll': {
        'subsystem_options': {'core_aerodynamics': {'method': 'low_speed'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'connect_initial_mass': False,
            'fix_initial': True,
            'fix_initial_mass': False,
            'time_duration_ref': (50.0, 's'),
            'time_duration_bounds': ((1.0, 100.0), 's'),
            'velocity_lower': (0, 'kn'),
            'velocity_upper': (1000, 'kn'),
            'velocity_ref': (150, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'ft'),
            'distance_upper': (10.0e3, 'ft'),
            'distance_ref': (3000, 'ft'),
            'distance_defect_ref': (3000, 'ft'),
        },
        'initial_guesses': {
            'time': ([0.0, 40.0], 's'),
            'velocity': ([0.066, 143.1], 'kn'),
            'distance': ([0.0, 1000.0], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'rotation': {
        'subsystem_options': {'core_aerodynamics': {'method': 'low_speed'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'time_duration_bounds': ((1, 100), 's'),
            'time_duration_ref': (50.0, 's'),
            'velocity_lower': (0, 'kn'),
            'velocity_upper': (1000, 'kn'),
            'velocity_ref': (100, 'kn'),
            'velocity_ref0': (0, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'ft'),
            'distance_upper': (10_000, 'ft'),
            'distance_ref': (5000, 'ft'),
            'distance_defect_ref': (5000, 'ft'),
            'angle_lower': (0.0, 'rad'),
            'angle_upper': (5.0, 'rad'),
            'angle_ref': (5.0, 'rad'),
            'angle_defect_ref': (5.0, 'rad'),
            'normal_ref': (10000, 'lbf'),
        },
        'initial_guesses': {
            'time': ([40.0, 5.0], 's'),
            'angle_of_attack': ([0.0, 2.5], 'deg'),
            'velocity': ([143, 150.0], 'kn'),
            'distance': ([3680.37217765, 4000], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'ascent': {
        'subsystem_options': {'core_aerodynamics': {'method': 'low_speed'}},
        'user_options': {
            'num_segments': 4,
            'order': 3,
            'fix_initial': False,
            'velocity_lower': (0, 'kn'),
            'velocity_upper': (700, 'kn'),
            'velocity_ref': (200, 'kn'),
            'velocity_ref0': (0, 'kn'),
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
            'altitude_final': (500, 'ft'),
            'alt_constraint_ref': (500, 'ft'),
            'angle_lower': (-10, 'rad'),
            'angle_upper': (20, 'rad'),
            'angle_ref': (57.2958, 'deg'),
            'angle_defect_ref': (57.2958, 'deg'),
            'pitch_constraint_lower': (0.0, 'deg'),
            'pitch_constraint_upper': (15.0, 'deg'),
            'pitch_constraint_ref': (1.0, 'deg'),
        },
        'initial_guesses': {
            'time': ([45.0, 25.0], 's'),
            'flight_path_angle': ([0.0, 8.0], 'deg'),
            'angle_of_attack': ([2.5, 1.5], 'deg'),
            'velocity': ([150.0, 185.0], 'kn'),
            'distance': ([4.0e3, 10.0e3], 'ft'),
            'altitude': ([0.0, 500.0], 'ft'),
            'tau_gear': (0.2, 'unitless'),
            'tau_flaps': (0.9, 'unitless'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'accel': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'alt': (500, 'ft'),
            'EAS_constraint_eq': (250, 'kn'),
            'time_duration_bounds': ((1, 200), 's'),
            'time_duration_ref': (1000, 's'),
            'velocity_lower': (150, 'kn'),
            'velocity_upper': (270, 'kn'),
            'velocity_ref': (250, 'kn'),
            'velocity_ref0': (150, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'NM'),
            'distance_upper': (150, 'NM'),
            'distance_ref': (5, 'NM'),
            'distance_defect_ref': (5, 'NM'),
        },
        'initial_guesses': {
            'time': ([70.0, 13.0], 's'),
            'velocity': ([185.0, 250.0], 'kn'),
            'distance': ([10.0e3, 20.0e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'climb1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'EAS_target': (250, 'kn'),
            'mach_cruise': 0.8,
            'target_mach': False,
            'altitude_final': (10.0e3, 'ft'),
            'time_duration_bounds': ((30, 300), 's'),
            'time_duration_ref': (1000, 's'),
            'alt_lower': (400, 'ft'),
            'alt_upper': (11_000, 'ft'),
            'alt_ref': (10.0e3, 'ft'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'NM'),
            'distance_upper': (500, 'NM'),
            'distance_ref': (10, 'NM'),
            'distance_ref0': (0, 'NM'),
        },
        'initial_guesses': {
            'time': ([1.0, 2.0], 'min'),
            'distance': ([20.0e3, 100.0e3], 'ft'),
            'altitude': ([500.0, 10.0e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'climb2': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise'}},
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'fix_initial': False,
            'EAS_target': (270, 'kn'),
            'mach_cruise': 0.8,
            'target_mach': True,
            'altitude_final': (37.5e3, 'ft'),
            'required_available_climb_rate': (0.1, 'ft/min'),
            'time_duration_bounds': ((200, 17_000), 's'),
            'time_duration_ref': (5000, 's'),
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
        },
        'initial_guesses': {
            'time': ([216.0, 1300.0], 's'),
            'distance': ([100.0e3, 200.0e3], 'ft'),
            'altitude': ([10.0e3, 37.5e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise'}},
        'user_options': {
            'alt_cruise': (37.5e3, 'ft'),
            'mach_cruise': 0.8,
        },
        'initial_guesses': {
            # [Initial mass, delta mass] for special cruise phase.
            'mass': ([171481.0, -35000], 'lbm'),
            'initial_distance': (200.0e3, 'ft'),
            'initial_time': (1516.0, 's'),
            'altitude': (37.5e3, 'ft'),
            'mach': (0.8, 'unitless'),
        },
    },
    'desc1': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise'}},
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'fix_initial': False,
            'input_initial': False,
            'EAS_limit': (350, 'kn'),
            'mach_cruise': 0.8,
            'input_speed_type': SpeedType.MACH,
            'altitude_final': (10.0e3, 'ft'),
            'time_duration_bounds': ((300.0, 900.0), 's'),
            'time_duration_ref': (1000, 's'),
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
            'distance_lower': (3000.0, 'NM'),
            'distance_upper': (5000.0, 'NM'),
            'distance_ref': (mission_distance, 'NM'),
            'distance_ref0': (0, 'NM'),
            'distance_defect_ref': (100, 'NM'),
        },
        'initial_guesses': {
            'mass': (136000.0, 'lbm'),
            'altitude': ([37.5e3, 10.0e3], 'ft'),
            'throttle': ([0.0, 0.0], 'unitless'),
            'distance': ([0.92 * mission_distance, 0.96 * mission_distance], 'NM'),
            'time': ([28000.0, 500.0], 's'),
        },
    },
    'desc2': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise'}},
        'user_options': {
            'num_segments': 1,
            'order': 7,
            'fix_initial': False,
            'input_initial': False,
            'EAS_limit': (250, 'kn'),
            'mach_cruise': 0.80,
            'input_speed_type': SpeedType.EAS,
            'altitude_final': (1000, 'ft'),
            'time_duration_bounds': ((100.0, 5000), 's'),
            'time_duration_ref': (500, 's'),
            'alt_lower': (500, 'ft'),
            'alt_upper': (11_000, 'ft'),
            'alt_ref': (10.0e3, 'ft'),
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
        },
        'initial_guesses': {
            'mass': (136000.0, 'lbm'),
            'altitude': ([10.0e3, 1.0e3], 'ft'),
            'throttle': ([0.0, 0.0], 'unitless'),
            'distance': ([0.96 * mission_distance, mission_distance], 'NM'),
            'time': ([28500.0, 500.0], 's'),
        },
    },
}


def phase_info_parameterization(phase_info, post_mission_info, aviary_inputs):
    """
    Modify the values in the phase_info dictionary to accommodate different values
    for the following mission design inputs: cruise altitude, cruise Mach number,
    cruise range, design gross mass.

    Parameters
    ----------
    phase_info : dict
        Dictionary of phase settings for a mission profile
    post_mission_info : dict
        Dictionary of phase settings for a post mission profile
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
        phase_info['desc1']['initial_guesses']['distance'] = (
            [0.92 * range_cruise, 0.96 * range_cruise],
            'NM',
        )
        phase_info['desc2']['initial_guesses']['distance'] = (
            [0.96 * range_cruise, range_cruise],
            'NM',
        )
        range_scale = range_cruise / old_range_cruise

    # Altitude
    old_alt_cruise = phase_info['climb2']['user_options']['altitude_final'][0]
    if alt_cruise != old_alt_cruise:
        phase_info['climb2']['user_options']['altitude_final'] = (alt_cruise, 'ft')
        phase_info['climb2']['initial_guesses']['altitude'] = ([10.0e3, alt_cruise], 'ft')
        phase_info['cruise']['initial_guesses']['altitude'] = (alt_cruise, 'ft')
        phase_info['desc1']['initial_guesses']['altitude'] = ([alt_cruise, 10.0e3], 'ft')

        # TODO - Could adjust time guesses/bounds in climb2 and desc2.

    # Mass
    old_gross_mass = 175400.0
    if gross_mass != old_gross_mass:
        # Note, this requires that the guess for gross mass is pretty close to the
        # compute mass.

        fuel_used = 35000 * range_scale
        phase_info['groundroll']['initial_guesses']['mass'] = ([gross_mass, gross_mass], 'lbm')
        phase_info['rotation']['initial_guesses']['mass'] = ([gross_mass, gross_mass], 'lbm')
        phase_info['accel']['initial_guesses']['mass'] = ([gross_mass, gross_mass], 'lbm')
        phase_info['ascent']['initial_guesses']['mass'] = ([gross_mass, gross_mass], 'lbm')

        phase_info['cruise']['initial_guesses']['mass'] = ([gross_mass, -fuel_used], 'lbm')

        end_mass = gross_mass - fuel_used
        phase_info['desc1']['initial_guesses']['mass'] = (end_mass, 'lbm')
        phase_info['desc2']['initial_guesses']['mass'] = (end_mass, 'lbm')

    # Mach
    old_mach_cruise = phase_info['cruise']['initial_guesses']['mach'][0]
    if mach_cruise != old_mach_cruise:
        phase_info['cruise']['initial_guesses']['mach'] = (mach_cruise, 'unitless')

    return phase_info, post_mission_info
