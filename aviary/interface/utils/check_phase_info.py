from openmdao.utils.units import valid_units
from aviary.variable_info.enums import SpeedType, EquationsOfMotion

TWO_DEGREES_OF_FREEDOM = EquationsOfMotion.TWO_DEGREES_OF_FREEDOM
HEIGHT_ENERGY = EquationsOfMotion.HEIGHT_ENERGY
SOLVED_2DOF = EquationsOfMotion.SOLVED_2DOF


# Define common keys for all phases
common_keys = {
    'num_segments': int,
    'order': int,
    'fix_initial': (bool, dict),
}

# Common key-values for climb, cruise, and descent
common_entries = {
    'optimize_mach': bool,
    'optimize_altitude': bool,
    'solve_for_distance': bool,
    'initial_mach': tuple,
    'final_mach': tuple,
    'mach_bounds': tuple,
    'initial_altitude': tuple,
    'final_altitude': tuple,
    'altitude_bounds': tuple,
    'throttle_enforcement': str,
    'constrain_final': bool,
    'fix_duration': bool,
    'initial_bounds': tuple,
    'duration_bounds': tuple,
}

# Combine common and phase-specific entries
phase_keys_height_energy = {
    'pre_mission': {'include_takeoff': bool, 'optimize_mass': bool},
    'post_mission': {'include_landing': bool}
}

common_TAS = {
    'velocity_lower': tuple,
    'velocity_upper': tuple,
    'velocity_ref': tuple,
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
    'input_initial': (bool, dict),
    'EAS_limit': tuple,
    'mach_cruise': float,
    'input_speed_type': SpeedType,
    'final_altitude': tuple,
    'alt_constraint_ref': tuple,
}

phase_keys_gasp = {
    'groundroll': {
        'connect_initial_mass': bool,
        'fix_initial_mass': bool,
        **common_duration,
        **common_TAS,
        **common_mass,
        **common_distance,
        'distance_defect_ref': tuple,
    },
    'rotation': {
        **common_duration,
        **common_TAS,
        **common_mass,
        **common_distance,
        **common_angle,
        'normal_ref': tuple,
        'velocity_ref0': tuple,
        'distance_defect_ref': tuple,
    },
    'ascent': {
        **common_TAS,
        **common_mass,
        **common_distance,
        **common_alt,
        'final_altitude': tuple,
        'alt_constraint_ref': tuple,
        'alt_defect_ref': tuple,
        **common_angle,
        'pitch_constraint_lower': tuple,
        'pitch_constraint_upper': tuple,
        'pitch_constraint_ref': tuple,
        'velocity_ref0': tuple,
        'distance_defect_ref': tuple,
    },
    'accel': {
        'alt': tuple,
        'EAS_constraint_eq': tuple,
        **common_duration,
        'duration_ref': tuple,
        **common_TAS,
        **common_mass,
        **common_distance,
        'velocity_ref0': tuple,
        'distance_defect_ref': tuple,
    },
    'climb1': {
        'EAS_target': tuple,
        'mach_cruise': float,
        'target_mach': bool,
        'final_altitude': tuple,
        **common_duration,
        **common_alt,
        **common_mass,
        **common_distance,
        'distance_ref0': tuple,
    },
    'climb2': {
        'EAS_target': tuple,
        'mach_cruise': float,
        'target_mach': bool,
        'final_altitude': tuple,
        'required_available_climb_rate': tuple,
        **common_duration,
        **common_alt,
        **common_mass,
        **common_distance,
        'alt_ref0': tuple,
        'distance_ref0': tuple,
        'distance_defect_ref': tuple,
    },
    'cruise': {
        'mach_cruise': float,
        'alt_cruise': tuple,
    },
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


def check_phase_info(phase_info, mission_method):
    """
    Check if all phases exist in phase_info for the given mission_method.

    Parameters
    ----------
    phase_info : dict
        Dictionary of phase settings for a mission profile
    mission_method : str
        The mission method

    Returns
    -------
        True if all phases exist in phase_info
        Otherwise, raise key not found in phase exception
    """
    phase_keys = {}
    if mission_method is TWO_DEGREES_OF_FREEDOM:
        for phase in phase_info:
            base_phase = phase.removeprefix('reserve_')
            if base_phase != 'pre_mission' and base_phase != 'post_mission':
                if 'cruise' in base_phase:
                    phase_keys[phase] = {**phase_keys_gasp['cruise']}
                else:
                    phase_keys[phase] = {**common_keys, **phase_keys_gasp[base_phase]}
    elif mission_method is SOLVED_2DOF:
        return
    elif mission_method is HEIGHT_ENERGY:
        for phase in phase_info:
            if phase != 'pre_mission' and phase != 'post_mission':
                phase_keys[phase] = {**common_keys, **common_entries}
            else:
                phase_keys[phase] = phase_keys_height_energy[phase]
    else:
        possible_values = ["'"+e.value+"'" for e in EquationsOfMotion]
        possible_values[-1] = "or " + possible_values[-1]
        raise ValueError("Invalid mission_method. Please choose from " +
                         ", ".join(possible_values) + ".")

    # Check if all phases exist in phase_info
    for phase in phase_info:
        if mission_method is TWO_DEGREES_OF_FREEDOM:
            base_phase = phase.removeprefix('reserve_')
            if 'cruise' in base_phase:
                base_phase = 'cruise'
        else:
            base_phase = phase
        if 'user_options' in phase_info[base_phase]:
            phase_options = phase_info[base_phase]['user_options']
        else:
            phase_options = phase_info[base_phase]

        if phase_options.get('target_range', False) and phase_options.get('target_duration', False):
            raise ValueError(
                f"target_range and target_duration have both been set to True for {phase}, please pick one.")

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
