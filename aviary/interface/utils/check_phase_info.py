from openmdao.utils.units import valid_units
from aviary.variable_info.enums import SpeedType, EquationsOfMotion

HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY
TWO_DEGREES_OF_FREEDOM = EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
SIMPLE = EquationsOfMotion.SIMPLE
SOLVED = EquationsOfMotion.SOLVED


def check_phase_info(phase_info, mission_method):
    # Define common keys for all phases
    common_keys = {
        'num_segments': int,
        'order': int,
        'fix_initial': (bool, dict),
    }

    # Common key-values for climb, cruise, and descent
    common_entries = {
        'initial_ref': tuple,
        'initial_bounds': tuple,
        'duration_ref': tuple,
        'duration_bounds': tuple,
    }

    # Phase-specific entries
    climb_specific = {
        'input_initial': bool,
        'no_descent': bool,
        'initial_mach': float,
        'initial_altitude': tuple,
        'final_altitude': tuple,
        'final_mach': float,
        'fix_range': bool,
        'fix_initial_time': bool,
    }

    cruise_specific = {
        'min_mach': float,
        'max_mach': float,
        'required_available_climb_rate': tuple,
        'mass_f_cruise': tuple,
        'range_f_cruise': tuple,
        'fix_final': bool,
    }

    descent_specific = {
        'initial_altitude': tuple,
        'final_altitude': tuple,
        'initial_mach': float,
        'final_mach': float,
        'no_climb': bool,
        'fix_range': bool,
    }

    # Combine common and phase-specific entries
    phase_keys_flops = {
        'pre_mission': {'include_takeoff': bool, 'optimize_mass': bool},
        'climb': {**common_entries, **climb_specific},
        'cruise': {**common_entries, **cruise_specific},
        'descent': {**common_entries, **descent_specific},
        'post_mission': {'include_landing': bool}
    }

    common_TAS = {
        'TAS_lower': tuple,
        'TAS_upper': tuple,
        'TAS_ref': tuple,
    }
    common_mass = {
        'mass_lower': tuple,
        'mass_upper': tuple,
        'mass_ref': tuple,
        'mass_defect_ref': tuple,
    }
    common_distance = {
        'distance_lower': tuple,
        'distance_upper': tuple,
        'distance_ref': tuple,
    }
    common_angle = {
        'angle_lower': tuple,
        'angle_upper': tuple,
        'angle_ref': tuple,
        'angle_defect_ref': tuple,
    }
    common_duration = {
        'duration_bounds': tuple,
        'duration_ref': tuple,
    }
    common_alt = {
        'alt_lower': tuple,
        'alt_upper': tuple,
        'alt_ref': tuple,
    }
    common_descent = {
        'input_initial': bool,
        'EAS_limit': tuple,
        'mach_cruise': float,
        'input_speed_type': SpeedType,
        'final_alt': tuple,
        'time_initial_bounds': tuple,
        'time_initial_ref': tuple,
        'alt_constraint_ref': tuple,
        'throttle_setting': float,
    }

    phase_keys_gasp = {
        'groundroll': {
            'connect_initial_mass': bool,
            'fix_initial_mass': bool,
            **common_duration,
            **common_TAS,
            **common_mass,
            **common_distance,
            'throttle_setting': float,
            'distance_defect_ref': tuple,
        },
        'rotation': {
            **common_duration,
            **common_TAS,
            **common_mass,
            **common_distance,
            **common_angle,
            'normal_ref': tuple,
            'throttle_setting': float,
            'TAS_ref0': tuple,
            'distance_defect_ref': tuple,
        },
        'ascent': {
            **common_TAS,
            **common_mass,
            **common_distance,
            **common_alt,
            'alt_constraint_eq': tuple,
            'alt_constraint_ref': tuple,
            'alt_constraint_ref0': tuple,
            'alt_defect_ref': tuple,
            **common_angle,
            'pitch_constraint_lower': tuple,
            'pitch_constraint_upper': tuple,
            'pitch_constraint_ref': tuple,
            'throttle_setting': float,
            'TAS_ref0': tuple,
            'distance_defect_ref': tuple,
        },
        'accel': {
            'alt': tuple,
            'EAS_constraint_eq': tuple,
            'time_initial_bounds': tuple,
            **common_duration,
            'duration_ref': tuple,
            **common_TAS,
            **common_mass,
            **common_distance,
            'throttle_setting': float,
            'TAS_ref0': tuple,
            'distance_defect_ref': tuple,
        },
        'climb1': {
            'EAS_target': tuple,
            'mach_cruise': float,
            'target_mach': bool,
            'final_alt': tuple,
            'time_initial_bounds': tuple,
            **common_duration,
            **common_alt,
            **common_mass,
            **common_distance,
            'throttle_setting': float,
            'distance_ref0': tuple,
        },
        'climb2': {
            'EAS_target': tuple,
            'mach_cruise': float,
            'target_mach': bool,
            'final_alt': tuple,
            'required_available_climb_rate': tuple,
            'time_initial_bounds': tuple,
            **common_duration,
            **common_alt,
            **common_mass,
            **common_distance,
            'throttle_setting': float,
            'alt_ref0': tuple,
            'distance_ref0': tuple,
            'distance_defect_ref': tuple,
        },
        'cruise': {'initial_guesses': dict, },
        'desc1': {
            **common_descent,
            **common_duration,
            **common_alt,
            **common_mass,
            **common_distance,
            'alt_ref0': tuple,
            'mass_ref0': tuple,
            'distance_ref0': tuple,
        },
        'desc2': {
            **common_descent,
            **common_duration,
            **common_alt,
            **common_mass,
            **common_distance,
            'alt_ref0': tuple,
            'distance_defect_ref': tuple,
        },
    }

    phase_keys = {}
    if mission_method is HEIGHT_ENERGY:
        for phase in phase_info:
            if phase != 'pre_mission' and phase != 'post_mission':
                phase_keys[phase] = {**common_keys, **phase_keys_flops[phase]}
            else:
                phase_keys[phase] = phase_keys_flops[phase]
    elif mission_method is TWO_DEGREES_OF_FREEDOM:
        for phase in phase_info:
            if phase != 'pre_mission' and phase != 'post_mission' and phase != 'cruise':
                phase_keys[phase] = {**common_keys, **phase_keys_gasp[phase]}
            else:
                phase_keys[phase] = phase_keys_gasp[phase]
    elif mission_method is SOLVED:
        return
    elif mission_method is SIMPLE:
        return
    else:
        raise ValueError(
            "Invalid mission_method. Please choose either 'FLOPS', 'GASP', 'simple', or 'solved'.")

    # Check if all phases exist in phase_info
    for phase in phase_info:
        if 'user_options' in phase_info[phase]:
            phase_options = phase_info[phase]['user_options']
        else:
            phase_options = phase_info[phase]

        # Check if all required keys exist, if there are no extra keys, and if they are of the correct type
        for key, expected_type in phase_keys[phase].items():
            # Check tuples for (val, units) structure
            if expected_type is tuple:
                if key not in phase_options:
                    raise ValueError(
                        f"Key {key} not found in phase {phase} of the provided dictionary.")
                value = phase_options[key]
                if not isinstance(value, tuple) or len(value) != 2:
                    raise ValueError(f"Key {key} in phase {phase} should be a tuple of size 2 (value, unit), "
                                     f"but got {value}.")
                unit = value[1]
                if not valid_units(unit):
                    raise ValueError(
                        f"Key {key} in phase {phase} has an invalid unit {unit}.")

            if not isinstance(phase_options[key], expected_type):
                raise TypeError(f"Key {key} in phase {phase} should be of type {expected_type.__name__}, "
                                f"but got type {type(phase_options[key]).__name__}")

    return True
