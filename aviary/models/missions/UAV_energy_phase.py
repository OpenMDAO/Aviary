from copy import deepcopy


phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': False},
    'climb': {
        'subsystem_options': {'aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'mach_optimize': True,
            'mach_polynomial_order': 3,
            'mach_initial': (0.05, 'unitless'),
            'mach_final': (0.1, 'unitless'),
            'mass_ref': (1, 'kg'),
            # 'distance_initial': (0, 'ft'), # Do not hard-fix climb initial distance.
            'distance_ref': (1.0e2, 'ft'),
            'altitude_optimize': True,
            'altitude_polynomial_order': 3,
            'altitude_initial': (0.0, 'ft'),
            'altitude_final': (200.0, 'ft'),
            'throttle_enforcement': 'control',
            'throttle_polynomial_order': 1,
            'time_initial': (0.0, 'min'),
            'time_duration_bounds': ((0.1, 25), 's'),
            'constraints': {
                'mach': {
                    'upper': 0.145773,
                    'units': 'unitless',
                    'type': 'path',
                }
            },
        },
        'initial_guesses': {'time': ([0, 6], 's'), 'mach': ([0.05, 0.1], 'unitless')},
    },
    'cruise': {
        'subsystem_options': {'aerodynamics': {'method': 'external'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': True,
            'mach_initial': (0.0538, 'unitless'),
            'mach_bounds': ((0.05, 0.3), 'unitless'),
            # 'mach_ref': (0.05, 'unitless'),
            'mass_ref': (4.0, 'kg'),
            # 'alt_ref': (100, 'ft'),
            # 'mach_final': (0.05, 'unitless'),
            'altitude_optimize': True,
            'altitude_initial': (200.0, 'ft'),
            'altitude_bounds': ((100.0, 300.0), 'ft'),
            'altitude_final': (200.0, 'ft'),
            'distance_initial': (0.0, 'm'),
            'distance_ref': (1000.0, 'm'),
            'target_distance': (1000.0, 'm'),
            'throttle_enforcement': 'control',
            # 'throttle_polynomial_order': 1,
            # Time
            'time_initial': (0.0, 's'),
            'time_duration_bounds': ((0, 180.0), 's'),
        },
        'initial_guesses': {
            'distance': ([0, 1000], 'm'),
            'time': ([0, 54.7], 's'),
        },
    },
    'descent': {
        'subsystem_options': {'aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.36, 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (34000.0, 'ft'),
            'altitude_final': (500.0, 'ft'),
            'throttle_enforcement': 'path_constraint',
            'time_initial_bounds': ((120.5, 361.5), 'min'),
            'time_duration_bounds': ((29.0, 87.0), 'min'),
        },
        'initial_guesses': {'time': ([241, 58], 'min')},
    },
    'post_mission': {
        'include_landing': False,
        # 'target_range': (200, 'ft'),
        # 'constraint_range':True,
    },
}
