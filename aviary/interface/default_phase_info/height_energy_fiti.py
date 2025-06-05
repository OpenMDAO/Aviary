from aviary.mission.flops_based.phases.time_integration_phases import SGMHeightEnergy
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AlphaModes, SpeedType
from aviary.variable_info.variables import Dynamic, Mission

# defaults for height energy based forward in time integration phases
cruise_mach = (0.8,)
cruise_alt = (35e3,)

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': True},
    'climb': {
        'builder': SGMHeightEnergy,
        'user_options': {
            'mach': (cruise_mach, 'unitless'),
            'alt_trigger': (cruise_alt, 'ft'),
        },
    },
    'cruise': {
        'kwargs': dict(
            input_speed_type=SpeedType.MACH,
            input_speed_units='unitless',
            alpha_mode=AlphaModes.REQUIRED_LIFT,
        ),
        'builder': SGMHeightEnergy,
        'user_options': {
            'mach': (cruise_mach, 'unitless'),
        },
    },
    'descent': {
        'builder': SGMHeightEnergy,
        'user_options': {
            'mach': (cruise_mach, 'unitless'),
            'alt_trigger': (1000, 'ft'),
            Dynamic.Vehicle.Propulsion.THROTTLE: (0, 'unitless'),
        },
    },
    'post_mission': {
        'include_landing': False,
        'constrain_range': True,
        'target_range': (1906.0, 'nmi'),
    },
}


def phase_info_parameterization(phase_info, post_mission_info, aviary_inputs: AviaryValues):
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
    alt_cruise = aviary_inputs.get_item(Mission.Design.CRUISE_ALTITUDE)
    mach_cruise = aviary_inputs.get_item(Mission.Design.MACH)

    phase_info['climb']['user_options']['alt_trigger'] = alt_cruise
    phase_info['climb']['user_options']['mach'] = mach_cruise

    phase_info['cruise']['user_options']['mach'] = mach_cruise

    phase_info['descent']['user_options']['mach'] = mach_cruise

    return phase_info, post_mission_info
