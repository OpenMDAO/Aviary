from aviary.mission.gasp_based.phases.time_integration_phases import (
    SGMAccel,
    SGMAscentCombined,
    SGMClimb,
    SGMCruise,
    SGMDescent,
    SGMGroundroll,
    SGMRotation,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AlphaModes, SpeedType, Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

# defaults for 2DOF based forward in time integration phases
cruise_alt = (35e3,)
cruise_mach = (0.8,)

takeoff_phases = {
    'groundroll': {
        'builder': SGMGroundroll,
        'user_options': {
            # special case
            'attr:VR_value': ('SGMGroundroll_velocity_trigger', 'kn'),
        },
    },
    'rotation': {
        'builder': SGMRotation,
        'user_options': {},
    },
    'ascent': {
        'builder': SGMAscentCombined,
        'user_options': {
            't_init_gear': (10000, 's'),
            't_init_flaps': (10000, 's'),
            # special case
            'rotation.start_rotation': (10000, 's'),
            # special case
            'attr:fuselage_angle_max': (Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE, 'deg'),
        },
    },
    'accel': {
        'builder': SGMAccel,
        'user_options': {},
    },
}
climb_phases = {
    'climb1': {
        'kwargs': dict(
            input_speed_type=SpeedType.EAS,
            input_speed_units='kn',
            speed_trigger_units='unitless',
        ),
        'builder': SGMClimb,
        'user_options': {
            'alt_trigger': (10000, 'ft'),
            'EAS': (250, 'kn'),
            'speed_trigger': (cruise_mach, 'unitless'),
        },
    },
    'climb2': {
        'kwargs': dict(
            input_speed_type=SpeedType.EAS,
            input_speed_units='kn',
            speed_trigger_units='unitless',
        ),
        'builder': SGMClimb,
        'user_options': {
            'alt_trigger': (cruise_alt, 'ft'),
            'EAS': (270, 'kn'),
            'speed_trigger': (cruise_mach, 'unitless'),
        },
    },
    'climb3': {
        'kwargs': dict(
            input_speed_type=SpeedType.MACH,
            input_speed_units='unitless',
            speed_trigger_units='kn',
        ),
        'builder': SGMClimb,
        'user_options': {
            'alt_trigger': (cruise_alt, 'ft'),
            'mach': (cruise_mach, 'unitless'),
            'speed_trigger': (0, 'kn'),
        },
    },
}
ascent_phases = {**takeoff_phases, **climb_phases}
cruise_phase = {
    'cruise': {
        'kwargs': dict(
            input_speed_type=SpeedType.MACH,
            input_speed_units='unitless',
            alpha_mode=AlphaModes.REQUIRED_LIFT,
        ),
        'builder': SGMCruise,
        'user_options': {
            'mach': (cruise_mach, 'unitless'),
            'attr:mass_trigger': ('SGMCruise_mass_trigger', 'lbm'),
        },
    },
}
descent_phases = {
    'desc1': {
        'kwargs': dict(
            input_speed_type=SpeedType.MACH,
            input_speed_units='unitless',
            speed_trigger_units='kn',
        ),
        'builder': SGMDescent,
        'user_options': {
            'alt_trigger': (10000, 'ft'),
            'mach': (cruise_mach, 'unitless'),
            'speed_trigger': (350, 'kn'),
            Dynamic.Vehicle.Propulsion.THROTTLE: (0, 'unitless'),
        },
        'descent_phase': True,
    },
    'desc2': {
        'kwargs': dict(
            input_speed_type=SpeedType.EAS,
            input_speed_units='kn',
            speed_trigger_units='kn',
        ),
        'builder': SGMDescent,
        'user_options': {
            'alt_trigger': (10000, 'ft'),
            'EAS': (350, 'kn'),
            'speed_trigger': (0, 'kn'),
            Dynamic.Vehicle.Propulsion.THROTTLE: (0, 'unitless'),
        },
        'descent_phase': True,
    },
    'desc3': {
        'kwargs': dict(
            input_speed_type=SpeedType.EAS,
            input_speed_units='kn',
            speed_trigger_units='kn',
        ),
        'builder': SGMDescent,
        'user_options': {
            'alt_trigger': (1000, 'ft'),
            'EAS': (250, 'kn'),
            'speed_trigger': (0, 'kn'),
            Dynamic.Vehicle.Propulsion.THROTTLE: (0, 'unitless'),
        },
        'descent_phase': True,
    },
}

phase_info = {
    **ascent_phases,
    **cruise_phase,
    **descent_phases,
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
    aviary_inputs : <AviaryValues>
        Object containing values and units for all aviary inputs and options

    Returns
    -------
    dict
        Modified phase_info that has been changed to match the new mission
        parameters
    """
    # range_cruise = aviary_inputs.get_item(Mission.Design.RANGE)
    alt_cruise = aviary_inputs.get_item(Mission.Design.CRUISE_ALTITUDE)
    # gross_mass = aviary_inputs.get_item(Mission.Design.GROSS_MASS)
    mach_cruise = aviary_inputs.get_item(Mission.Design.MACH)

    phase_info['climb1']['user_options']['speed_trigger'] = mach_cruise

    phase_info['climb2']['user_options']['alt_trigger'] = alt_cruise
    phase_info['climb2']['user_options']['speed_trigger'] = mach_cruise

    phase_info['climb3']['user_options']['alt_trigger'] = alt_cruise
    phase_info['climb3']['user_options']['mach'] = mach_cruise

    phase_info['cruise']['user_options']['mach'] = mach_cruise

    phase_info['desc1']['user_options']['mach'] = mach_cruise

    return phase_info, post_mission_info


def add_default_sgm_args(phase_info: dict, ode_args: dict, verbosity=None):
    """Add default SGM arguments to phase_info."""
    for phase_name, info in phase_info.items():
        kwargs = info.get('kwargs', {})
        if 'ode_args' not in kwargs:
            kwargs['ode_args'] = ode_args
        if 'simupy_args' not in kwargs:
            if verbosity is None:
                verbosity, _ = ode_args['aviary_options'].get_item(
                    Settings.VERBOSITY, default=(Verbosity.QUIET, 'unitless')
                )
            kwargs['simupy_args'] = {'verbosity': verbosity}
        info['kwargs'] = kwargs
