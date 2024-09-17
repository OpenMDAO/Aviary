from aviary.variable_info.variables import Mission

# defaults for height energy based phases

phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": True},
    "climb": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.2, "unitless"),
            "final_mach": (0.72, "unitless"),
            "mach_bounds": ((0.18, 0.74), "unitless"),
            "initial_altitude": (0.0, "ft"),
            "final_altitude": (32000.0, "ft"),
            "altitude_bounds": ((0.0, 34000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": True,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 0.0), "min"),
            "duration_bounds": ((64.0, 192.0), "min"),
            "add_initial_mass_constraint": False,
        },
    },
    "cruise": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.72, "unitless"),
            "final_mach": (0.72, "unitless"),
            "mach_bounds": ((0.7, 0.74), "unitless"),
            "initial_altitude": (32000.0, "ft"),
            "final_altitude": (34000.0, "ft"),
            "altitude_bounds": ((23000.0, 38000.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((64.0, 192.0), "min"),
            "duration_bounds": ((56.5, 169.5), "min"),
        },
    },
    "descent": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.72, "unitless"),
            "final_mach": (0.36, "unitless"),
            "mach_bounds": ((0.34, 0.74), "unitless"),
            "initial_altitude": (34000.0, "ft"),
            "final_altitude": (500.0, "ft"),
            "altitude_bounds": ((0.0, 38000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((120.5, 361.5), "min"),
            "duration_bounds": ((29.0, 87.0), "min"),
        },
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": True,
        "target_range": (1906., "nmi"),
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
    old_alt_cruise = 32000.
    if alt_cruise != old_alt_cruise:
        phase_info['climb']['user_options']['final_altitude'] = (alt_cruise, 'ft')
        phase_info['cruise']['user_options']['initial_altitude'] = (alt_cruise, 'ft')
        phase_info['cruise']['user_options']['final_altitude'] = (alt_cruise, 'ft')
        phase_info['descent']['user_options']['initial_altitude'] = (alt_cruise, 'ft')

    # Mach
    old_mach_cruise = 0.72
    if mach_cruise != old_mach_cruise:
        phase_info['climb']['user_options']['final_mach'] = (mach_cruise, "unitless")
        phase_info['cruise']['user_options']['initial_mach'] = (mach_cruise, "unitless")
        phase_info['cruise']['user_options']['final_mach'] = (mach_cruise, "unitless")
        phase_info['descent']['user_options']['initial_mach'] = (mach_cruise, "unitless")

    return phase_info, post_mission_info
