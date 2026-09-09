"""
Phase info for a simple UAV mission.
"""

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': False},
    'climb': {
        'subsystem_options': {'aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'mach_optimize': True,
            'mach_polynomial_order': 3,
            'mach_initial': (0.0538, 'unitless'),
            'mach_final': (0.0538, 'unitless'),
            'mach_bounds' : ((0.05, 0.1), 'unitless'),
            'mass_ref': (1, 'kg'),
            'distance_initial': (0, 'ft'),
            'distance_ref': (1.0e2, 'ft'),
            'altitude_optimize': True,
            'altitude_initial': (0.0, 'ft'),
            'altitude_final': (200.0, 'ft'),
            'altitude_bounds': ((0, 200), 'ft'),
            'throttle_enforcement': 'control',
            'time_initial': (0.0, 'min'),
            'time_duration_bounds': ((0.1, 100), 's'),
        },
        'initial_guesses': {'time': ([0, 20], 's'), 'mach': ([0.05, 0.0538], 'unitless')},
    },
    'cruise': {
        'subsystem_options': {'aerodynamics': {'method': 'external'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'mach_optimize': True,
            'mach_initial': (0.05, 'unitless'),
            'mach_bounds': ((0.01, 0.3), 'unitless'),
            'mach_ref': (0.05, 'unitless'),
            'mass_ref': (4.0, 'kg'),
            'altitude_optimize': True,
            'altitude_initial': (200.0, 'ft'),
            'altitude_bounds': ((100.0, 300.0), 'ft'),
            'altitude_ref': (200, 'ft'),
            'distance_ref': (1000.0, 'm'),
            'throttle_enforcement': 'control',
            'time_duration_bounds': ((5, 240.0), 's'),
        },
        'initial_guesses': {
            'distance': ([0, 1000], 'm'),
            'time': ([20, 120], 's'),
        },
    },
    'descent': {
        'subsystem_options': {'aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'mach_optimize': True,
            'mach_polynomial_order': 1,
            'mach_initial': (0.0538, 'unitless'),
            'mach_final': (0.05, 'unitless'),
            'altitude_optimize': True,
            'altitude_initial': (200.0, 'ft'),
            'altitude_final': (0.0, 'ft'),
            'throttle_enforcement': 'control',
            'time_duration_bounds': ((0.1, 100), 's'),
        },
        'initial_guesses': {'time': ([140, 20], 's')},
    },
    'post_mission': {
        'include_landing': False,
    },
}
