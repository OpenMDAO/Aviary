from aviary.variable_info.enums import SpeedType, ThrottleAllocation
from aviary.subsystems.energy.battery_builder import BatteryBuilder

# Energy method
energy_phase_info = {
    "pre_mission": {'external_subsystems': [BatteryBuilder()], "include_takeoff": False, "optimize_mass": True},
    "climb": {
        'external_subsystems': [BatteryBuilder()],
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.2, "unitless"),
            "final_mach": (0.475, "unitless"),
            "mach_bounds": ((0.08, 0.478), "unitless"),
            "initial_altitude": (0.0, "ft"),
            "final_altitude": (21_000.0, "ft"),
            "altitude_bounds": ((0.0, 22_000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": True,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 0.0), "min"),
            "duration_bounds": ((24.0, 192.0), "min"),
            "add_initial_mass_constraint": False,
        },
    },
    "cruise": {
        'external_subsystems': [BatteryBuilder()],
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.475, "unitless"),
            "final_mach": (0.475, "unitless"),
            "mach_bounds": ((0.47, 0.48), "unitless"),
            "initial_altitude": (21_000.0, "ft"),
            "final_altitude": (21_000.0, "ft"),
            "altitude_bounds": ((20_000.0, 22_000.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((64.0, 192.0), "min"),
            "duration_bounds": ((56.5, 169.5), "min"),
        },
    },
    "descent": {
        'external_subsystems': [BatteryBuilder()],
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.475, "unitless"),
            "final_mach": (0.1, "unitless"),
            "mach_bounds": ((0.08, 0.48), "unitless"),
            "initial_altitude": (21_000.0, "ft"),
            "final_altitude": (500.0, "ft"),
            "altitude_bounds": ((0.0, 22_000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((100, 361.5), "min"),
            "duration_bounds": ((29.0, 87.0), "min"),
        },
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": True,
        "target_range": (2_020.0, "nmi"),
    },
}

# 2DOF
two_dof_phase_info = {
    'pre_mission': {'external_subsystems': [BatteryBuilder()]},
    'groundroll': {
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'connect_initial_mass': False,
            'fix_initial': True,
            'fix_initial_mass': False,
            'duration_ref': (50.0, 's'),
            'duration_bounds': ((1.0, 100.0), 's'),
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
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'duration_bounds': ((1, 100), 's'),
            'duration_ref': (50.0, 's'),
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
        'external_subsystems': [BatteryBuilder()],
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
            'final_altitude': (500, 'ft'),
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
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'alt': (500, 'ft'),
            'EAS_constraint_eq': (250, 'kn'),
            'duration_bounds': ((1, 200), 's'),
            'duration_ref': (1000, 's'),
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
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'EAS_target': (250, 'kn'),
            'mach_cruise': 0.475,
            'target_mach': False,
            'final_altitude': (10.0e3, 'ft'),
            'duration_bounds': ((30, 300), 's'),
            'duration_ref': (1000, 's'),
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
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'fix_initial': False,
            'EAS_target': (250, 'kn'),
            'mach_cruise': 0.475,
            'target_mach': True,
            'final_altitude': (21_000, 'ft'),
            'required_available_climb_rate': (0.1, 'ft/min'),
            'duration_bounds': ((200, 17_000), 's'),
            'duration_ref': (5000, 's'),
            'alt_lower': (9000, 'ft'),
            'alt_upper': (22_000, 'ft'),
            'alt_ref': (20_000, 'ft'),
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
            'altitude': ([10_000, 21_000], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        },
    },
    'cruise': {
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'alt_cruise': (21_000, 'ft'),
            'mach_cruise': 0.475,
        },
        'initial_guesses': {
            # [Initial mass, delta mass] for special cruise phase.
            'mass': ([150_000.0, -35_000], 'lbm'),
            'initial_distance': (100.0e3, 'ft'),
            'initial_time': (1_000.0, 's'),
            'altitude': (21_000, 'ft'),
            'mach': (0.475, 'unitless'),
        },
    },
    'desc1': {
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'fix_initial': False,
            'input_initial': False,
            'EAS_limit': (350, 'kn'),
            'mach_cruise': 0.475,
            'input_speed_type': SpeedType.MACH,
            'final_altitude': (10_000, 'ft'),
            'duration_bounds': ((300.0, 900.0), 's'),
            'duration_ref': (1000, 's'),
            'alt_lower': (1000, 'ft'),
            'alt_upper': (22_000, 'ft'),
            'alt_ref': (20_000, 'ft'),
            'alt_ref0': (0, 'ft'),
            'alt_constraint_ref': (10000, 'ft'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (140_000, 'lbm'),
            'mass_ref0': (0, 'lbm'),
            'mass_defect_ref': (140_000, 'lbm'),
            'distance_lower': (1_000.0, 'NM'),
            'distance_upper': (3_000.0, 'NM'),
            'distance_ref': (2_020, 'NM'),
            'distance_ref0': (0, 'NM'),
            'distance_defect_ref': (100, 'NM'),
        },
        'initial_guesses': {
            'mass': (136000.0, 'lbm'),
            'altitude': ([20_000, 10_000], 'ft'),
            'throttle': ([0.0, 0.0], 'unitless'),
            'distance': ([0.92 * 2_020, 0.96 * 2_020], 'NM'),
            'time': ([28000.0, 500.0], 's'),
        },
    },
    'desc2': {
        'external_subsystems': [BatteryBuilder()],
        'user_options': {
            'num_segments': 1,
            'order': 7,
            'fix_initial': False,
            'input_initial': False,
            'EAS_limit': (250, 'kn'),
            'mach_cruise': 0.80,
            'input_speed_type': SpeedType.EAS,
            'final_altitude': (1000, 'ft'),
            'duration_bounds': ((100.0, 5000), 's'),
            'duration_ref': (500, 's'),
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
            'distance': ([0.96 * 2_020, 2_020], 'NM'),
            'time': ([28500.0, 500.0], 's'),
        },
    },
}

phase_info = two_dof_phase_info
