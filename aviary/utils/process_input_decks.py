"""
This module, process_input_decks.py, is responsible for reading vehicle input decks, initializing options,
and setting initial guesses for aircraft design parameters. It works primarily with .csv files,
allowing for the specification of units, comments, and lists within these files.

The module supports various functions like creating a vehicle, parsing input files, updating options based
on inputs, and handling initial guesses for different aircraft design aspects. It heavily relies on the
aviary and openMDAO libraries for processing and interpreting the aircraft design parameters.

Functions:
    create_vehicle(vehicle_deck=''): Create and initialize a vehicle with default or specified parameters.
    parse_inputs(vehicle_deck, aircraft_values): Parse input files and update aircraft values and initial guesses.
    update_options(aircraft_values, initialization_guesses): Update dependent options based on current aircraft values.
    update_dependent_options(aircraft_values, dependent_options): Update options that depend on the value of an input variable.
    initialization_guessing(aircraft_values): Set initial guesses for aircraft parameters based on problem type and other factors.
"""

import warnings
from operator import eq, ge, gt, le, lt, ne

import numpy as np
from openmdao.utils.units import valid_units

from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.utils.functions import convert_strings_to_data, get_path
from aviary.utils.preprocessors import remove_preprocessed_options
from aviary.variable_info.enums import ProblemType, Verbosity
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Settings

operation_dict = {
    '<': lt,
    '<=': le,
    '==': eq,
    '!=': ne,
    '>=': ge,
    '>': gt,
    'isinstance': isinstance,
}
problem_types = {
    'sizing': ProblemType.SIZING,
    'alternate': ProblemType.ALTERNATE,
    'fallout': ProblemType.FALLOUT,
}


def create_vehicle(vehicle_deck='', meta_data=_MetaData, verbosity=Verbosity.BRIEF):
    """
    Creates and initializes a vehicle with default or specified parameters. It sets up the aircraft values
    and initial guesses based on the input from the vehicle deck.

    Parameters
    ----------
    vehicle_deck (str, AviaryValues):
        Path to the vehicle deck file, or an AviaryValues object that contains aircraft
        inputs. Default is an empty string.
    meta_data (dict):
        Variable metadata used when reading input file for unit validation,
        default values, and other checks.
    verbosity (int, Verbosity):
        Verbosity level for the AviaryProblem. If provided, this overrides verbosity
        specified in the aircraft data. Default is None, and verbosity will be taken
        from aircraft data or defaulted to Verbosity.BRIEF if not found.

    Returns
    -------
    (aircraft_values, initialization_guesses): (tuple)
        Returns a tuple containing aircraft values and initial guesses.
    """
    if verbosity is None:
        verbosity = Verbosity.BRIEF

    aircraft_values = get_option_defaults(engine=False)
    remove_preprocessed_options(aircraft_values)

    # TODO remove all hardcoded GASP values here, find appropriate place for them
    aircraft_values.set_val('INGASP.JENGSZ', val=4)
    aircraft_values.set_val('test_mode', val=False)
    aircraft_values.set_val('use_surrogates', val=True)
    aircraft_values.set_val('mass_defect', val=10000, units='lbm')
    # TODO problem_type should get set by get_option_defaults??
    aircraft_values.set_val(Settings.PROBLEM_TYPE, val=ProblemType.SIZING)
    aircraft_values.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False)

    initialization_guesses = {
        # initialization_guesses is a dictionary that contains values used to initialize the trajectory
        'actual_takeoff_mass': 0,
        'rotation_mass': 0,
        'operating_empty_mass': 0,
        'fuel_burn_per_passenger_mile': 0,
        'cruise_mass_final': 0,
        'flight_duration': 0,
        'time_to_climb': 0,
        'climb_range': 0,
        'reserves': 0,
    }

    if isinstance(vehicle_deck, AviaryValues):
        for key, (val, units) in vehicle_deck:
            if key.startswith('initialization_guesses:'):
                initialization_guesses[key.removeprefix('initialization_guesses:')] = val
        aircraft_values.update(vehicle_deck)
    else:
        if verbosity >= Verbosity.BRIEF:
            verbose = True
        else:
            verbose = False
        vehicle_deck = get_path(vehicle_deck, verbose)
        aircraft_values, initialization_guesses = parse_inputs(
            vehicle_deck=vehicle_deck,
            aircraft_values=aircraft_values,
            initialization_guesses=initialization_guesses,
            meta_data=meta_data,
        )

    # make sure verbosity is always set
    # if verbosity set via parameter, use that - override what is in the file
    if verbosity is not None:
        # Enum conversion here, so user can pass either number or actual Enum as parameter
        aircraft_values.set_val(Settings.VERBOSITY, Verbosity(verbosity))
    # else, if verbosity not specified anywhere, use default of BRIEF
    elif verbosity is None and Settings.VERBOSITY not in aircraft_values:
        aircraft_values.set_val(Settings.VERBOSITY, _MetaData[Settings.VERBOSITY]['default_value'])

    return aircraft_values, initialization_guesses


def parse_inputs(
    vehicle_deck,
    aircraft_values: AviaryValues = None,
    initialization_guesses=None,
    meta_data=_MetaData,
):
    """
    Parses the input files and updates the aircraft values and initial guesses. The function reads the
    vehicle deck file, processes each line, and updates the aircraft_values object based on the data found.

    Parameters
    ----------
    vehicle_deck (str): The vehicle deck file path.
    aircraft_values (AviaryValues): An instance of AviaryValues to be updated.
    initialization_guesses: An initialized dictionary of trajectory values to be updated.

    Returns
    -------
    tuple: Updated aircraft values and initial guesses.
    """
    if aircraft_values is None:
        aircraft_values = AviaryValues()

    if initialization_guesses is None:
        initialization_guesses = {}

    guess_names = list(initialization_guesses.keys())

    with open(vehicle_deck, newline='') as f_in:
        for line in f_in:
            data_units = None

            tmp = [*line.split('#', 1), '']
            line, comment = tmp[0], tmp[1]  # anything after the first # is a comment

            data = ''.join(line.rstrip(',').split())  # remove all white space

            if len(data) == 0:
                continue  # skip line it contained only commas

            # remove any elements that are empty (caused by trailing commas or extra commas)
            data_list = [dat for dat in data.split(',') if dat != '']

            # continue if there's no data in the line but there are commas
            # this might occur if someone edits a .csv file in Excel
            if len(data_list) == 0:
                continue
            var_name = data_list.pop(0)
            if valid_units(data_list[-1]):
                # if the last element is a unit, remove it from the list and update the variable's units
                data_units = data_list.pop()

            var_value = convert_strings_to_data(data_list)
            # If var_value is length 1 list and is not supposed to be a list, pull out
            # individual value. Otherwise, convert list to numpy array
            if len(var_value) <= 1:
                if var_name in meta_data and meta_data[var_name]['multivalue']:
                    # if data is numeric, convert to numpy array
                    if isinstance(var_value[0], (int, float)):
                        var_value = np.array(var_value)
                else:
                    var_value = var_value[0]

            if var_name in meta_data.keys():
                if data_units is None:
                    data_units = meta_data[var_name]['units']
                aircraft_values.set_val(var_name, var_value, data_units, meta_data)
                continue

            elif var_name in guess_names:
                # all initial guesses take only a single value
                # get values from supplied dictionary
                initialization_guesses[var_name] = var_value
                continue

            elif var_name.startswith('initialization_guesses:'):
                # get values labeled as initialization_guesses in .csv input file
                initialization_guesses[var_name.removeprefix('initialization_guesses:')] = var_value
                continue

            elif ':' in var_name:
                warnings.warn(
                    f"Variable '{var_name}' is not in meta_data nor in 'guess_names'. "
                    'It will be ignored.',
                    UserWarning,
                )
                continue

            if aircraft_values.get_val(Settings.VERBOSITY) >= Verbosity.VERBOSE:
                print('Unused:', var_name, var_value, comment)

    return aircraft_values, initialization_guesses


# TODO this should be a preprocessor, and tasks split to be specific to subsystem
#      e.g. aero preprocessor, mass preprocessor, 2DOF preprocessor, etc.


def update_GASP_options(aircraft_values: AviaryValues):
    """
    Updates options based on the current values in aircraft_values. This function also handles special cases
    and prints debug information if the debug mode is active.

    Parameters
    ----------
    aircraft_values (AviaryValues): An instance of AviaryValues containing current aircraft values.

    Returns
    -------
    tuple: Updated aircraft values and initial guesses.
    """
    # update the options that depend on variables
    update_dependent_options(aircraft_values, dependent_options)

    ## STRUT AND FOLD ##
    if not aircraft_values.get_val(Aircraft.Wing.HAS_STRUT):
        aircraft_values.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=False)

    if aircraft_values.get_val(Aircraft.Wing.HAS_FOLD):
        if not aircraft_values.get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION):
            aircraft_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True)
        else:
            dim_loc_spec = aircraft_values.get_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED)
            aircraft_values.set_val(
                Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=dim_loc_spec
            )
    else:
        aircraft_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True)
        aircraft_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=False)

    if aircraft_values.get_val(Settings.VERBOSITY) >= Verbosity.VERBOSE:
        print('\nOptions')
        for key in get_keys(aircraft_values):
            val, units = aircraft_values.get_item(key)
            print(key, val, units)

    return aircraft_values


def update_dependent_options(aircraft_values: AviaryValues, dependent_options):
    """
    Updates options that are dependent on the value of an input variable or option. The function iterates
    through each dependent option and sets its value based on the current aircraft values.

    Parameters
    ----------
    aircraft_values (AviaryValues): An instance of AviaryValues containing current aircraft values.
    dependent_options (list): A list of dependent options and their dependencies.

    Returns
    -------
    AviaryValues: Updated aircraft values with modified dependent options.
    """
    # gets the names of all the variables that affect dependent options
    for var_name, dependency in dependent_options:
        if var_name in get_keys(aircraft_values):
            var_value, var_units = aircraft_values.get_item(var_name)
            # dependency is a dictionary that contains the target option, the relationship to the variable and the output values
            if dependency['relation'] in operation_dict:
                comp = operation_dict[dependency['relation']]
                outcome = (
                    dependency['result']
                    if comp(var_value, dependency['val'])
                    else dependency['alternate']
                )
            elif dependency['relation'] == 'in':
                outcome = (
                    dependency['result']
                    if var_value in dependency['val']
                    else dependency['alternate']
                )
            else:
                warnings.warn(dependency['relation'] + ' is not a valid selection')
            aircraft_values.set_val(dependency['target'], val=outcome)
    return aircraft_values


def initialization_guessing(aircraft_values: AviaryValues, initialization_guesses, engine_builders):
    """
    Sets initial guesses for various aircraft parameters based on the current problem type, aircraft values,
    and other factors. It calculates and sets values like takeoff mass, cruise mass, flight duration, etc.

    Parameters
    ----------
    aircraft_values : AviaryValues
        An instance of AviaryValues containing current aircraft values.

    initialization_guesses : dict
        Initial guesses.

    engine_builders : list or None
        List of engine builders. This is needed if there are multiple engine models.

    Returns
    -------
    tuple
        Updated aircraft values and initial guesses.
    """
    problem_type = aircraft_values.get_val(Settings.PROBLEM_TYPE)
    num_pax = aircraft_values.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
    reserve_val = aircraft_values.get_val(Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm')
    reserve_frac = aircraft_values.get_val(Aircraft.Design.RESERVE_FUEL_FRACTION, units='unitless')

    if initialization_guesses['fuel_burn_per_passenger_mile'] <= 0:
        initialization_guesses['fuel_burn_per_passenger_mile'] = 0.1

    reserves = initialization_guesses['reserves']
    if reserves < 0.0:
        raise ValueError('initialization_guesses["reserves"] must be greater than or equal to 0.')
    elif reserves == 0:
        reserves += reserve_val
        reserves += reserve_frac * (
            num_pax
            * initialization_guesses['fuel_burn_per_passenger_mile']
            * aircraft_values.get_val(Mission.Design.RANGE, units='NM')
        )
    elif reserves < 10:
        reserves *= (
            num_pax
            * initialization_guesses['fuel_burn_per_passenger_mile']
            * aircraft_values.get_val(Mission.Design.RANGE, units='NM')
        )

    initialization_guesses['reserves'] = reserves

    if Mission.Summary.GROSS_MASS in aircraft_values:
        mission_mass = aircraft_values.get_val(Mission.Summary.GROSS_MASS, units='lbm')
    else:
        mission_mass = aircraft_values.get_val(Mission.Design.GROSS_MASS, units='lbm')

    if Mission.Summary.CRUISE_MASS_FINAL in aircraft_values:
        cruise_mass_final = aircraft_values.get_val(Mission.Summary.CRUISE_MASS_FINAL, units='lbm')
    else:
        cruise_mass_final = initialization_guesses['cruise_mass_final']

    # takeoff mass not given
    if mission_mass <= 0:
        if problem_type == ProblemType.ALTERNATE:
            fuel_mass = (
                num_pax
                * (
                    initialization_guesses['fuel_burn_per_passenger_mile']
                    * aircraft_values.get_val(Mission.Design.RANGE, units='NM')
                )
                + reserves
            )
            mission_mass = (
                initialization_guesses['operating_empty_mass']
                + (
                    num_pax
                    * aircraft_values.get_val(
                        Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, units='lbm'
                    )
                )
                + fuel_mass
            )
        elif problem_type == ProblemType.FALLOUT or problem_type == ProblemType.SIZING:
            mission_mass = aircraft_values.get_val(Mission.Design.GROSS_MASS, units='lbm')
    initialization_guesses['actual_takeoff_mass'] = mission_mass

    if cruise_mass_final == 0:  # no guess given
        if problem_type == ProblemType.SIZING:
            cruise_mass_final = 0.8
        elif problem_type == ProblemType.ALTERNATE:
            cruise_mass_final = -1
    # estimation based on payload and fuel
    if cruise_mass_final <= 0:
        cruise_mass_final = (
            initialization_guesses['operating_empty_mass']
            + num_pax
            * aircraft_values.get_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, units='lbm')
            + reserves
        )
    # fraction of takeoff mass
    elif cruise_mass_final <= 1:
        cruise_mass_final = mission_mass * cruise_mass_final
    initialization_guesses['cruise_mass_final'] = cruise_mass_final

    if initialization_guesses['rotation_mass'] <= 0:
        initialization_guesses['rotation_mass'] = 0.99
    if initialization_guesses['rotation_mass'] <= 1:  # fraction of takeoff mass
        initialization_guesses['rotation_mass'] = (
            mission_mass * initialization_guesses['rotation_mass']
        )

    if Mission.Design.MACH in aircraft_values:
        cruise_mach = aircraft_values.get_val(Mission.Design.MACH)
    else:
        cruise_mach = aircraft_values.get_val(Mission.Summary.CRUISE_MACH)

    if initialization_guesses['flight_duration'] <= 0:  # estimation based on mach
        initialization_guesses['flight_duration'] = (
            aircraft_values.get_val(Mission.Design.RANGE, units='NM')
            / (667 * cruise_mach)
            * (60 * 60)
        )
    elif initialization_guesses['flight_duration'] <= 15:  # duration entered in hours
        initialization_guesses['flight_duration'] = initialization_guesses['flight_duration'] * (
            60 * 60
        )

    # TODO this does not work at all for mixed-type engines (some propeller and some not)
    try:
        num_engines = aircraft_values.get_val(Aircraft.Engine.NUM_ENGINES)

        # This happens before preprocessing, and we end up with the default when unspecified.
        # num_engines = np.array(num_engines)

        if aircraft_values.get_val(Aircraft.Engine.HAS_PROPELLERS):
            # For large turboprops, 1 pound of thrust per hp at takeoff seems to be close enough
            total_thrust = np.dot(
                aircraft_values.get_val(Aircraft.Engine.Gearbox.SHAFT_POWER_DESIGN, 'hp'),
                aircraft_values.get_val(Aircraft.Engine.NUM_ENGINES),
            )
        else:
            total_thrust = np.dot(
                aircraft_values.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 'lbf')
                * aircraft_values.get_val(Aircraft.Engine.SCALE_FACTOR),
                aircraft_values.get_val(Aircraft.Engine.NUM_ENGINES),
            )

    except KeyError:
        if engine_builders is not None and len(engine_builders) > 1:
            # heterogeneous engine-model case. Get thrust from the engine models instead.
            total_thrust = 0
            for model in engine_builders:
                thrust = model.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 'lbf') * model.get_val(
                    Aircraft.Engine.SCALE_FACTOR
                )
                num_engines = model.get_val(Aircraft.Engine.NUM_ENGINES)
                total_thrust += thrust * num_engines

        else:
            total_thrust = np.dot(
                aircraft_values.get_val(Aircraft.Engine.SCALED_SLS_THRUST, 'lbf'),
                aircraft_values.get_val(Aircraft.Engine.NUM_ENGINES),
            )

    gamma_guess = np.arcsin(0.5 * total_thrust / mission_mass)
    avg_speed_guess = 0.5 * 667 * cruise_mach  # kts

    if initialization_guesses['time_to_climb'] <= 0:  # no guess given
        initialization_guesses['time_to_climb'] = aircraft_values.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft'
        ) / (avg_speed_guess * np.sin(gamma_guess))
    elif initialization_guesses['time_to_climb'] <= 2:  # duration entered in hours
        initialization_guesses['time_to_climb'] = initialization_guesses['time_to_climb'] * (
            60 * 60
        )
    elif initialization_guesses['time_to_climb'] <= 200:  # average climb rate in ft/s
        initialization_guesses['time_to_climb'] = (
            aircraft_values.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft')
            / initialization_guesses['time_to_climb']
        )

    # range covered using an average speed from 0 to cruise
    if initialization_guesses['climb_range'] <= 0:
        initialization_guesses['climb_range'] = (
            initialization_guesses['time_to_climb']
            / (60 * 60)
            * (avg_speed_guess * np.cos(gamma_guess))
        )

    if aircraft_values.get_val(Settings.VERBOSITY) >= Verbosity.VERBOSE:
        print('\nInitial Guesses')
        for key, value in initialization_guesses.items():
            print(key, value)

    return initialization_guesses


dependent_options = [
    # dependent_options is a list that is used to update options that depend on the value of
    # an input variable or option.
    # Each dependency comes in the form of a list with the variable name and a dictionary.
    # The dictionary contains a value that the variable will be compared against (val), the
    # particular mathematical comparison (relation), the option that is effected (target), and
    # the value the option should be set to based on whether the relation is true (result) or
    # false (alternate)
    # For example, an aircraft's engines are on the fuselage if the span fraction on the wing is 0.
    # In this case the value of Aircraft.Engine.WING_LOCATIONS is compared to 0. If the span location
    # is exactly equal to zero, engine_on_fuselage is set to to True, otherwise it is set to False.
    # One variable can be used to set the value of any number of options, but an option can only be
    # set by one variable
    # [Aircraft.Engine.WING_LOCATIONS, {
    #     'val': 0, 'relation': '==', 'target': Aircraft.Engine.FUSELAGE_MOUNTED, 'result': True, 'alternate': False}],
    [
        Aircraft.Wing.LOADING,
        {
            'val': 20,
            'relation': '>',
            'target': Aircraft.Wing.LOADING_ABOVE_20,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.Strut.ATTACHMENT_LOCATION,
        {
            'val': 0,
            'relation': '!=',
            'target': Aircraft.Wing.HAS_STRUT,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.Strut.ATTACHMENT_LOCATION,
        {
            'val': 1,
            'relation': '>',
            'target': Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.Wing.FOLD_MASS_COEFFICIENT,
        {
            'val': 0,
            'relation': '>',
            'target': Aircraft.Wing.HAS_FOLD,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.Wing.FOLDED_SPAN,
        {
            'val': 1,
            'relation': '>',
            'target': Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
        {
            'val': 0,
            'relation': '<',
            'target': Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.Engine.TYPE,
        {
            'val': [1, 2, 3, 4, 6, 11, 12, 13, 14],
            'relation': 'in',
            'target': Aircraft.Engine.HAS_PROPELLERS,
            'result': True,
            'alternate': False,
        },
    ],
    [
        'JENGSZ',
        {
            'val': 4,
            'relation': '!=',
            'target': Aircraft.Engine.SCALE_PERFORMANCE,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.HorizontalTail.VOLUME_COEFFICIENT,
        {
            'val': 0,
            'relation': '==',
            'target': Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
            'result': True,
            'alternate': False,
        },
    ],
    [
        Aircraft.VerticalTail.VOLUME_COEFFICIENT,
        {
            'val': 0,
            'relation': '==',
            'target': Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
            'result': True,
            'alternate': False,
        },
    ],
]
