import warnings

from aviary.mission.gasp_based.phases.time_integration_phases import (
    SGMAccel,
    SGMAscentCombined,
    SGMClimb,
    SGMDescent,
    SGMGroundroll,
    SGMRotation,
)
from aviary.variable_info.enums import SpeedType, Verbosity
from aviary.variable_info.variables import Aircraft

# defaults for 2DOF based forward in time integration phases


def create_2dof_based_ascent_phases(
    ode_args,
    cruise_alt=35e3,
    cruise_mach=0.8,
    simupy_args=dict(
        verbosity=Verbosity.QUIET,
    ),
):
    warnings.warn(
        '`descent_range_and_fuel` has been replaced with `add_descent_estimation_as_submodel`'
        '\nThe new methodology uses a phase_info that more closely matches that of collocation',
        DeprecationWarning,
    )

    groundroll_kwargs = dict(
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    groundroll_vals = {
        # special case
        'attr:VR_value': {'val': 'SGMGroundroll_velocity_trigger', 'units': 'kn'},
    }

    rotation_kwargs = dict(
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    rotation_vals = {}

    ascent_kwargs = dict(
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    ascent_vals = {
        't_init_gear': {'val': 10000, 'units': 's'},
        't_init_flaps': {'val': 10000, 'units': 's'},
        'rotation.start_rotation': {'val': 10000, 'units': 's'},  # special case
        # special case
        'attr:fuselage_angle_max': {
            'val': Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE,
            'units': 'deg',
        },
    }

    accel_kwargs = dict(
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    accel_vals = {}

    # need to set trigger units
    climb1_kwargs = dict(
        input_speed_type=SpeedType.EAS,
        input_speed_units='kn',
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    climb1_vals = {
        'alt_trigger': {'val': 10000, 'units': 'ft'},
        'EAS': {'val': 250, 'units': climb1_kwargs['input_speed_units']},
        'speed_trigger': {'val': cruise_mach, 'units': None},
    }

    climb2_kwargs = dict(
        input_speed_type=SpeedType.EAS,
        input_speed_units='kn',
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    climb2_vals = {
        'alt_trigger': {'val': cruise_alt, 'units': 'ft'},
        'EAS': {'val': 270, 'units': climb2_kwargs['input_speed_units']},
        'speed_trigger': {'val': cruise_mach, 'units': None},
    }

    climb3_kwargs = dict(
        input_speed_type=SpeedType.MACH,
        input_speed_units='unitless',
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    climb3_vals = {
        'alt_trigger': {'val': cruise_alt, 'units': 'ft'},
        'mach': {'val': cruise_mach, 'units': climb3_kwargs['input_speed_units']},
        'speed_trigger': {'val': 0, 'units': None},
    }

    phases = {
        'groundroll': {
            'ode': SGMGroundroll(**groundroll_kwargs),
            'vals_to_set': groundroll_vals,
        },
        'rotation': {
            'ode': SGMRotation(**rotation_kwargs),
            'vals_to_set': rotation_vals,
        },
        'ascent': {
            'ode': SGMAscentCombined(**ascent_kwargs),
            'vals_to_set': ascent_vals,
        },
        'accel': {
            'ode': SGMAccel(**accel_kwargs),
            'vals_to_set': accel_vals,
        },
        'climb1': {
            'ode': SGMClimb(**climb1_kwargs),
            'vals_to_set': climb1_vals,
        },
        'climb2': {
            'ode': SGMClimb(**climb2_kwargs),
            'vals_to_set': climb2_vals,
        },
        'climb3': {
            'ode': SGMClimb(**climb3_kwargs),
            'vals_to_set': climb3_vals,
        },
    }

    return phases


def create_2dof_based_descent_phases(
    ode_args,
    cruise_mach=0.8,
    simupy_args=dict(
        verbosity=Verbosity.QUIET,
    ),
):
    warnings.warn(
        '`descent_range_and_fuel` has been replaced with `add_descent_estimation_as_submodel`'
        '\nThe new methodology uses a phase_info that more closely matches that of collocation',
        DeprecationWarning,
    )

    descent1_kwargs = dict(
        input_speed_type=SpeedType.MACH,
        input_speed_units='unitless',
        speed_trigger_units='kn',
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    descent1_vals = {
        'alt_trigger': {'val': 10000, 'units': 'ft'},
        'mach': {'val': cruise_mach, 'units': descent1_kwargs['input_speed_units']},
        'speed_trigger': {'val': 350, 'units': descent1_kwargs['speed_trigger_units']},
    }

    descent2_kwargs = dict(
        input_speed_type=SpeedType.EAS,
        input_speed_units='kn',
        speed_trigger_units='kn',
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    descent2_vals = {
        'alt_trigger': {'val': 10000, 'units': 'ft'},
        'EAS': {'val': 350, 'units': descent2_kwargs['input_speed_units']},
        'speed_trigger': {'val': 0, 'units': descent2_kwargs['speed_trigger_units']},
    }

    descent3_kwargs = dict(
        input_speed_type=SpeedType.EAS,
        input_speed_units='kn',
        speed_trigger_units='kn',
        ode_args=ode_args,
        simupy_args=simupy_args,
    )
    descent3_vals = {
        'alt_trigger': {'val': 1000, 'units': 'ft'},
        'EAS': {'val': 250, 'units': descent3_kwargs['input_speed_units']},
        'speed_trigger': {'val': 0, 'units': descent3_kwargs['speed_trigger_units']},
    }

    phases = {
        'descent1': {
            'ode': SGMDescent(**descent1_kwargs),
            'vals_to_set': descent1_vals,
        },
        'descent2': {
            'ode': SGMDescent(**descent2_kwargs),
            'vals_to_set': descent2_vals,
        },
        'descent3': {
            'ode': SGMDescent(**descent3_kwargs),
            'vals_to_set': descent3_vals,
        },
    }

    return phases
