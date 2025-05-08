from aviary.variable_info.variables import Mission

# defaults for height energy based phases

phase_info = {
    'pre_mission': {'include_takeoff': True, 'optimize_mass': True},
    'climb': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'optimize_mach': True,
            'optimize_altitude': True,
            'num_segments': 6,
            'order': 3,
            'solve_for_distance': False,
            'initial_mach': (0.2, 'unitless'),
            'final_mach': (0.79, 'unitless'),
            'mach_bounds': ((0.18, 0.81), 'unitless'),
            'initial_altitude': (0.0, 'ft'),
            'final_altitude': (35000.0, 'ft'),
            'altitude_bounds': ((0.0, 37000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'fix_initial': False,
            'input_initial': True,
            'constrain_final': False,
            'fix_duration': False,
            'initial_bounds': ((0.0, 0.0), 'min'),
            'duration_bounds': ((12.1, 60.0), 'min'),
            'add_initial_mass_constraint': False,
            'no_descent': True,
        },
        'initial_guesses': {'time': ([0, 40.0], 'min')},
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'optimize_mach': False,
            'optimize_altitude': False,
            'num_segments': 1,
            'order': 3,
            'solve_for_distance': False,
            'initial_mach': (0.79, 'unitless'),
            'final_mach': (0.79, 'unitless'),
            'mach_bounds': ((0.77, 0.81), 'unitless'),
            'initial_altitude': (35000.0, 'ft'),
            'final_altitude': (35000.0, 'ft'),
            'altitude_bounds': ((34000.0, 36000.0), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'fix_initial': False,
            'constrain_final': False,
            'fix_duration': False,
            'initial_bounds': ((12.1, 60.0), 'min'),
            'duration_bounds': ((203.1, 812.4), 'min'),
        },
    },
    'descent': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'optimize_mach': True,
            'optimize_altitude': True,
            'num_segments': 5,
            'order': 3,
            'solve_for_distance': False,
            'initial_mach': (0.79, 'unitless'),
            'final_mach': (0.3, 'unitless'),
            'mach_bounds': ((0.3, 0.79), 'unitless'),
            'initial_altitude': (35000.0, 'ft'),
            'final_altitude': (35.0, 'ft'),
            'altitude_bounds': ((0.0, 38000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'fix_initial': False,
            'constrain_final': True,
            'fix_duration': False,
            'initial_bounds': ((215.1, 872.4), 'min'),
            'duration_bounds': ((14.6, 58.5), 'min'),
            'no_climb': True,
        },
    },
    'post_mission': {
        'include_landing': True,
        'constrain_range': True,
        'target_range': (3380.0, 'nmi'),
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
        Modified phase_info and post_mission_info that have been changed to match
        the new mission parameters
    """

    alt_cruise = aviary_inputs.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft')
    mach_cruise = aviary_inputs.get_val(Mission.Summary.CRUISE_MACH)

    # Range
    old_range_cruise, range_units = post_mission_info['target_range']
    range_cruise = aviary_inputs.get_val(Mission.Design.RANGE, units=range_units)
    if range_cruise != old_range_cruise:
        new_val = post_mission_info['target_range'][0] * range_cruise / old_range_cruise
        post_mission_info['target_range'] = (new_val, range_units)

    # Altitude
    old_alt_cruise = 35000.0
    if alt_cruise != old_alt_cruise:
        phase_info['climb']['user_options']['final_altitude'] = (alt_cruise, 'ft')
        phase_info['cruise']['user_options']['initial_altitude'] = (alt_cruise, 'ft')
        phase_info['cruise']['user_options']['final_altitude'] = (alt_cruise, 'ft')
        phase_info['descent']['user_options']['initial_altitude'] = (alt_cruise, 'ft')

    # Mach
    old_mach_cruise = 0.79
    if mach_cruise != old_mach_cruise:
        phase_info['climb']['user_options']['final_mach'] = (mach_cruise, 'unitless')
        phase_info['cruise']['user_options']['initial_mach'] = (mach_cruise, 'unitless')
        phase_info['cruise']['user_options']['final_mach'] = (mach_cruise, 'unitless')
        phase_info['descent']['user_options']['initial_mach'] = (mach_cruise, 'unitless')

    return phase_info, post_mission_info
