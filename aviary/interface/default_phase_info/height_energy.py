from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Dynamic, Mission
from aviary.variable_info.enums import LegacyCode

FLOPS = LegacyCode.FLOPS


prop = CorePropulsionBuilder('core_propulsion', BaseMetaData)
mass = CoreMassBuilder('core_mass', BaseMetaData, FLOPS)
aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
geom = CoreGeometryBuilder('core_geometry', BaseMetaData, FLOPS)

default_premission_subsystems = [prop, geom, mass, aero]
default_mission_subsystems = [aero, prop]

phase_info = {
    'pre_mission': {
        'include_takeoff': True,
        'optimize_mass': True,
    },
    'climb': {
        'subsystem_options': {
            'core_aerodynamics': {'method': 'computed'}
        },
        'user_options': {
            'fix_initial': {Dynamic.Mission.MASS: False, Dynamic.Mission.RANGE: False},
            'fix_initial_time': True,
            'fix_duration': False,
            'num_segments': 6,
            'order': 3,
            'initial_altitude': (0., 'ft'),
            'initial_ref': (1., 's'),
            'initial_bounds': ((0., 500.), 's'),
            'duration_ref': (1452., 's'),
            'duration_bounds': ((726., 2904.), 's'),
            'final_altitude': (10668, 'm'),
            'input_initial': True,
            'no_descent': False,
            'initial_mach': 0.1,
            'final_mach': 0.79,
            'fix_range': False,
            'add_initial_mass_constraint': False,
        },
        'initial_guesses': {
            'times': ([2., 24.2], 'min'),
            'altitude': ([0., 35.e3], 'ft'),
            'velocity': ([220., 455.49], 'kn'),
            'mass': ([170.e3, 165.e3], 'lbm'),
            'range': ([0., 160.3], 'nmi'),
            'velocity_rate': ([0.25, 0.05], 'm/s**2'),
            'throttle': ([0.5, 0.5], 'unitless'),
        }
    },
    'cruise': {
        'subsystem_options': {
            'core_aerodynamics': {'method': 'computed'}
        },
        'user_options': {
            'fix_initial': False,
            'fix_final': False,
            'fix_duration': False,
            'num_segments': 1,
            'order': 3,
            'initial_ref': (1., 's'),
            'initial_bounds': ((500., 4000.), 's'),
            'duration_ref': (24370.8, 's'),
            'duration_bounds': ((726., 48741.6), 's'),
            'min_altitude': (10.668e3, 'm'),
            'max_altitude': (10.668e3, 'm'),
            'min_mach': 0.79,
            'max_mach': 0.79,
            'required_available_climb_rate': (1.524, 'm/s'),
            'mass_f_cruise': (1.e4, 'kg'),
            'range_f_cruise': (1.e6, 'm'),
        },
        'initial_guesses': {
            'times': ([26.2, 406.18], 'min'),
            'altitude': ([35.e3, 35.e3], 'ft'),
            'velocity': ([455.49, 455.49], 'kn'),
            'mass': ([165.e3, 140.e3], 'lbm'),
            'range': ([160.3, 3243.9], 'nmi'),
            'velocity_rate': ([0., 0.], 'm/s**2'),
            'throttle': ([0.95, 0.9], 'unitless'),
        }
    },
    'descent': {
        'subsystem_options': {
            'core_aerodynamics': {'method': 'computed'}
        },
        'user_options': {
            'fix_initial': False,
            'fix_range': True,
            'fix_duration': False,
            'num_segments': 5,
            'order': 3,
            'initial_ref': (1., 's'),
            'initial_bounds': ((10.e3, 30.e3), 's'),
            'duration_ref': (1754.4, 's'),
            'duration_bounds': ((726., 3508.8), 's'),
            'initial_altitude': (10.668e3, 'm'),
            'final_altitude': (10.668, 'm'),
            'initial_mach': 0.79,
            'final_mach': 0.3,
            'no_climb': False,
        },
        'initial_guesses': {
            'times': ([432.38, 29.24], 'min'),
            'altitude': ([35.e3, 35.], 'ft'),
            'velocity': ([455.49, 198.44], 'kn'),
            'mass': ([120.e3, 115.e3], 'lbm'),
            'range': ([3243.9, 3378.7], 'nmi'),
            'velocity_rate': ([-0.25, 0.0], 'm/s**2'),
            'throttle': ([0., 0.], 'unitless'),
        }
    },
    'post_mission': {
        'include_landing': True,
    },
}


def phase_info_parameterization(phase_info, aviary_inputs):
    """
    Modify the values in the phase_info dictionary to accomodate different values
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

    range_cruise = aviary_inputs.get_val(Mission.Design.RANGE, units='NM')
    alt_cruise = aviary_inputs.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft')
    gross_mass = aviary_inputs.get_val(Mission.Design.GROSS_MASS, units='lbm')
    mach_cruise = aviary_inputs.get_val(Mission.Summary.CRUISE_MACH)

    # Range
    range_scale = 1.0
    old_range_cruise = 3500.0
    if range_cruise != old_range_cruise:
        range_scale = range_cruise / old_range_cruise

        vals = phase_info['descent']['initial_guesses']['range'][0]
        new_vals = [vals[0] * range_scale, vals[1] * range_scale]
        phase_info['descent']['initial_guesses']['range'] = (new_vals, 'NM')

        vals = phase_info['cruise']['initial_guesses']['range'][0]
        new_val = vals[1] * range_scale
        phase_info['cruise']['initial_guesses']['range'] = ([vals[0], new_val], 'NM')

    # Altitude

    # This doesn't seem to be stored regularly in some of the files.
    old_alt_cruise = 35000
    if alt_cruise != old_alt_cruise:

        phase_info['climb']['initial_guesses']['altitude'] = ([0.0, alt_cruise], 'ft')
        phase_info['climb']['user_options']['final_altitude'] = (alt_cruise, 'ft')
        phase_info['cruise']['initial_guesses']['altitude'] = \
            ([alt_cruise, alt_cruise], 'ft')
        phase_info['cruise']['user_options']['min_altitude'] = (alt_cruise, 'ft')
        phase_info['cruise']['user_options']['max_altitude'] = (alt_cruise, 'ft')
        phase_info['descent']['initial_guesses']['altitude'] = ([alt_cruise, 0.0], 'ft')

        # TODO - Could adjust time guesses/bounds in climb2 and desc2.

    # Mass
    old_gross_mass = 175400.0
    if gross_mass != old_gross_mass:

        # Note, this requires that the guess for gross mass is pretty close to the
        # compute mass.

        mass_scale = gross_mass / old_gross_mass

        for phase in ['climb', 'cruise', 'descent']:
            vals = phase_info[phase]['initial_guesses']['mass'][0]
            new_vals = [vals[0] * mass_scale, vals[1] * mass_scale]
            phase_info[phase]['initial_guesses']['mass'] = (new_vals, 'lbm')

    # Mach
    old_mach_cruise = 0.79
    if mach_cruise != old_mach_cruise:

        phase_info['climb']['user_options']['final_mach'] = mach_cruise
        phase_info['cruise']['user_options']['min_mach'] = mach_cruise
        phase_info['cruise']['user_options']['max_mach'] = mach_cruise
        phase_info['descent']['user_options']['initial_mach'] = mach_cruise

    return phase_info
