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
    update_options(aircraft_values, initial_guesses): Update dependent options based on current aircraft values.
    update_dependent_options(aircraft_values, dependent_options): Update options that depend on the value of an input variable.
    initial_guessing(aircraft_values): Set initial guesses for aircraft parameters based on problem type and other factors.
"""

import warnings
from operator import eq, ge, gt, le, lt, ne

import numpy as np
from openmdao.utils.units import valid_units

from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.utils.functions import convert_strings_to_data, set_value
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.enums import ProblemType, Verbosity
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Settings
from aviary.utils.functions import get_path


operation_dict = {"<": lt, "<=": le, "==": eq, "!=": ne,
                  ">=": ge, ">": gt, "isinstance": isinstance}
problem_types = {'sizing': ProblemType.SIZING,
                 'alternate': ProblemType.ALTERNATE, 'fallout': ProblemType.FALLOUT}


def create_vehicle(vehicle_deck='', meta_data=_MetaData, verbosity=None):
    """
    Creates and initializes a vehicle with default or specified parameters. It sets up the aircraft values
    and initial guesses based on the input from the vehicle deck.

    Parameters
    ----------
    vehicle_deck (str):
        Path to the vehicle deck file. Default is an empty string.
    meta_data (dict):
        Variable metadata used when reading input file for unit validation,
        default values, and other checks

    Returns
    -------
    (aircraft_values, initial_guesses): (tuple)
        Returns a tuple containing aircraft values and initial guesses.
    """
    aircraft_values = get_option_defaults(engine=False)

    # TODO remove all hardcoded GASP values here, find appropriate place for them
    aircraft_values.set_val('INGASP.JENGSZ', val=4)
    aircraft_values.set_val('test_mode', val=False)
    aircraft_values.set_val('use_surrogates', val=True)
    aircraft_values.set_val('mass_defect', val=10000, units='lbm')
    # TODO problem_type should get set by get_option_defaults??
    aircraft_values.set_val(Settings.PROBLEM_TYPE, val=ProblemType.SIZING)
    aircraft_values.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False)

    initial_guesses = {
        # initial_guesses is a dictionary that contains values used to initialize the trajectory
        'actual_takeoff_mass': 0,
        'rotation_mass': 0,
        'operating_empty_mass': 0,
        'fuel_burn_per_passenger_mile': 0,
        'cruise_mass_final': 0,
        'flight_duration': 0,
        'time_to_climb': 0,
        'climb_range': 0,
        'reserves': 0
    }

    initial_guesses = {
        # initial_guesses is a dictionary that contains values used to initialize the trajectory
        'actual_takeoff_mass': 0,
        'rotation_mass': 0,
        'operating_empty_mass': 0,
        'fuel_burn_per_passenger_mile': 0,
        'cruise_mass_final': 0,
        'flight_duration': 0,
        'time_to_climb': 0,
        'climb_range': 0,
        'reserves': 0
    }

    if isinstance(vehicle_deck, AviaryValues):
        aircraft_values.update(vehicle_deck)
    else:
        vehicle_deck = get_path(vehicle_deck)
        aircraft_values, initial_guesses = parse_inputs(
            vehicle_deck=vehicle_deck, aircraft_values=aircraft_values, initial_guesses=initial_guesses, meta_data=meta_data)

    # make sure verbosity is always set
    # if verbosity set via parameter, use that
    if verbosity is not None:
        # Enum conversion here, so user can pass either number or actual Enum as parameter
        aircraft_values.set_val(Settings.VERBOSITY, Verbosity(verbosity))
    # else, if verbosity not specified anywhere, use default of BRIEF
    elif verbosity is None and Settings.VERBOSITY not in aircraft_values:
        aircraft_values.set_val(Settings.VERBOSITY, Verbosity.BRIEF)

    return aircraft_values, initial_guesses


def parse_inputs(vehicle_deck, aircraft_values: AviaryValues = None, initial_guesses=None, meta_data=_MetaData):
    """
    Parses the input files and updates the aircraft values and initial guesses. The function reads the
    vehicle deck file, processes each line, and updates the aircraft_values object based on the data found.

    Parameters
    ----------
    vehicle_deck (str): The vehicle deck file path.
    aircraft_values (AviaryValues): An instance of AviaryValues to be updated.
    initial_guesses: An initialized dictionary of trajectory values to be updated.

    Returns
    -------
    tuple: Updated aircraft values and initial guesses.
    """
    if aircraft_values is None:
        aircraft_values = AviaryValues()
        aircraft_values.set_val(Settings.VERBOSITY, Verbosity.BRIEF)

    if initial_guesses is None:
        initial_guesses = {}

    guess_names = list(initial_guesses.keys())

    with open(vehicle_deck, newline='') as f_in:
        for line in f_in:
            used, data_units = False, None

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

            is_array = False
            if '[' in data_list[0]:
                is_array = True

            var_values = convert_strings_to_data(data_list)

            if var_name in meta_data.keys():
                aircraft_values = set_value(
                    var_name, var_values, aircraft_values, units=data_units, is_array=is_array, meta_data=meta_data)
                continue

            elif var_name in guess_names:
                # all initial guesses take only a single value
                # get values from supplied dictionary
                initial_guesses[var_name] = float(var_values[0])
                continue

            elif var_name.startswith('initial_guesses:'):
                # get values labelled as initial_guesses in .csv input file
                initial_guesses[var_name.split(':')[-1]] = float(var_values[0])
                continue

            elif ":" in var_name:
                warnings.warn(
                    f"Variable '{var_name}' is not in meta_data nor in 'guess_names'. It will be ignored.",
                    UserWarning)
                continue

            if aircraft_values.get_val(Settings.VERBOSITY).value >= 2:
                print('Unused:', var_name, var_values, comment)

    return aircraft_values, initial_guesses

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
        aircraft_values.set_val(
            Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=False)

    if aircraft_values.get_val(Aircraft.Wing.HAS_FOLD):
        if not aircraft_values.get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION):
            aircraft_values.set_val(
                Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True)
        else:
            dim_loc_spec = aircraft_values.get_val(
                Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED)
            aircraft_values.set_val(
                Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=dim_loc_spec)
    else:
        aircraft_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True)
        aircraft_values.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=False)

    if aircraft_values.get_val(Settings.VERBOSITY).value >= 2:
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
                outcome = dependency['result'] if comp(
                    var_value, dependency['val']) else dependency['alternate']
            elif dependency['relation'] == "in":
                outcome = dependency['result'] if var_value in dependency['val'] else dependency['alternate']
            else:
                warnings.warn(dependency['relation'] +
                              ' is not a valid selection')
            aircraft_values.set_val(dependency['target'], val=outcome)
    return aircraft_values


def initial_guessing(aircraft_values: AviaryValues, initial_guesses):
    """
    Sets initial guesses for various aircraft parameters based on the current problem type, aircraft values,
    and other factors. It calculates and sets values like takeoff mass, cruise mass, flight duration, etc.

    Parameters
    ----------
    aircraft_values (AviaryValues): An instance of AviaryValues containing current aircraft values.

    Returns
    -------
    tuple: Updated aircraft values and initial guesses.
    """
    problem_type = aircraft_values.get_val(Settings.PROBLEM_TYPE)
    num_pax = aircraft_values.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
    reserve_val = aircraft_values.get_val(
        Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm')
    reserve_frac = aircraft_values.get_val(
        Aircraft.Design.RESERVE_FUEL_FRACTION, units='unitless')

    if initial_guesses['fuel_burn_per_passenger_mile'] <= 0:
        initial_guesses['fuel_burn_per_passenger_mile'] = 0.1

    reserves = initial_guesses['reserves']
    if reserves < 0.0:
        raise ValueError(
            'initial_guesses["reserves"] must be greater than or equal to 0.')
    elif reserves == 0:
        reserves += reserve_val
        reserves += (reserve_frac * (num_pax * initial_guesses['fuel_burn_per_passenger_mile'] *
                                     aircraft_values.get_val(Mission.Design.RANGE, units='NM')))
    elif reserves < 10:
        reserves *= (num_pax * initial_guesses['fuel_burn_per_passenger_mile'] *
                     aircraft_values.get_val(Mission.Design.RANGE, units='NM'))

    initial_guesses['reserves'] = reserves

    if Mission.Summary.GROSS_MASS in aircraft_values:
        mission_mass = aircraft_values.get_val(Mission.Summary.GROSS_MASS, units='lbm')
    else:
        mission_mass = aircraft_values.get_val(Mission.Design.GROSS_MASS, units='lbm')

    if Mission.Summary.CRUISE_MASS_FINAL in aircraft_values:
        cruise_mass_final = aircraft_values.get_val(
            Mission.Summary.CRUISE_MASS_FINAL, units='lbm')
    else:
        cruise_mass_final = initial_guesses['cruise_mass_final']

    # takeoff mass not given
    if mission_mass <= 0:
        if problem_type == ProblemType.ALTERNATE:
            fuel_mass = num_pax * (
                initial_guesses['fuel_burn_per_passenger_mile'] * aircraft_values.get_val(Mission.Design.RANGE, units='NM')) + reserves
            mission_mass = initial_guesses['operating_empty_mass'] + (
                num_pax * aircraft_values.get_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, units='lbm')) + fuel_mass
        elif problem_type == ProblemType.FALLOUT or problem_type == ProblemType.SIZING:
            mission_mass = aircraft_values.get_val(
                Mission.Design.GROSS_MASS, units='lbm')
    initial_guesses['actual_takeoff_mass'] = mission_mass

    if cruise_mass_final == 0:  # no guess given
        if problem_type == ProblemType.SIZING:
            cruise_mass_final = .8
        elif problem_type == ProblemType.ALTERNATE:
            cruise_mass_final = -1
    # estimation based on payload and fuel
    if cruise_mass_final <= 0:
        cruise_mass_final = initial_guesses['operating_empty_mass'] + \
            num_pax * \
            aircraft_values.get_val(
                Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, units='lbm') + reserves
    # fraction of takeoff mass
    elif cruise_mass_final <= 1:
        cruise_mass_final = mission_mass * \
            cruise_mass_final
    initial_guesses['cruise_mass_final'] = cruise_mass_final

    if initial_guesses['rotation_mass'] <= 0:
        initial_guesses['rotation_mass'] = 0.99
    if initial_guesses['rotation_mass'] <= 1:  # fraction of takeoff mass
        initial_guesses['rotation_mass'] = mission_mass * \
            initial_guesses['rotation_mass']

    if Mission.Design.MACH in aircraft_values:
        cruise_mach = aircraft_values.get_val(Mission.Design.MACH)
    else:
        cruise_mach = aircraft_values.get_val(Mission.Summary.CRUISE_MACH)

    if initial_guesses['flight_duration'] <= 0:  # estimation based on mach
        initial_guesses['flight_duration'] = aircraft_values.get_val(
            Mission.Design.RANGE, units='NM') / (667 * cruise_mach) * (60 * 60)
    elif initial_guesses['flight_duration'] <= 15:  # duration entered in hours
        initial_guesses['flight_duration'] = initial_guesses['flight_duration'] * \
            (60 * 60)

    total_thrust = aircraft_values.get_val(
        Aircraft.Engine.SCALED_SLS_THRUST, 'lbf') * aircraft_values.get_val(Aircraft.Engine.NUM_ENGINES)
    gamma_guess = np.arcsin(.5*total_thrust / mission_mass)
    avg_speed_guess = (.5 * 667 * cruise_mach)  # kts

    if initial_guesses['time_to_climb'] <= 0:  # no guess given
        initial_guesses['time_to_climb'] = aircraft_values.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft') / \
            (avg_speed_guess * np.sin(gamma_guess))
    elif initial_guesses['time_to_climb'] <= 2:  # duration entered in hours
        initial_guesses['time_to_climb'] = initial_guesses['time_to_climb'] * (60 * 60)
    elif initial_guesses['time_to_climb'] <= 200:  # average climb rate in ft/s
        initial_guesses['time_to_climb'] = aircraft_values.get_val(Mission.Design.CRUISE_ALTITUDE, units='ft') / \
            initial_guesses['time_to_climb']

    # range covered using an average speed from 0 to cruise
    if initial_guesses['climb_range'] <= 0:
        initial_guesses['climb_range'] = initial_guesses['time_to_climb'] / \
            (60 * 60) * (avg_speed_guess * np.cos(gamma_guess))

    if aircraft_values.get_val(Settings.VERBOSITY).value >= 2:
        print('\nInitial Guesses')
        for key, value in initial_guesses.items():
            print(key, value)

    return initial_guesses


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
    [Aircraft.Fuselage.WETTED_AREA_FACTOR, {
        'val': 10, 'relation': '>', 'target': Aircraft.Fuselage.PROVIDE_SURFACE_AREA, 'result': True, 'alternate': False}],
    [Aircraft.Wing.LOADING, {'val': 20, 'relation': '>',
                             'target': Aircraft.Wing.LOADING_ABOVE_20, 'result': True, 'alternate': False}],
    [Aircraft.Strut.ATTACHMENT_LOCATION, {
        'val': 0, 'relation': '!=', 'target': Aircraft.Wing.HAS_STRUT, 'result': True, 'alternate': False}],
    [Aircraft.Strut.ATTACHMENT_LOCATION, {
        'val': 1, 'relation': '>', 'target': Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, 'result': True, 'alternate': False}],
    [Aircraft.Wing.FOLD_MASS_COEFFICIENT, {
        'val': 0, 'relation': '>', 'target': Aircraft.Wing.HAS_FOLD, 'result': True, 'alternate': False}],
    [Aircraft.Wing.FOLDED_SPAN, {'val': 1, 'relation': '>', 'target': Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                                 'result': True, 'alternate': False}],
    [Aircraft.Design.PART25_STRUCTURAL_CATEGORY, {
        'val': 0, 'relation': '<', 'target': Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER, 'result': True, 'alternate': False}],
    [Aircraft.Engine.TYPE, {
        'val': [1, 2, 3, 4, 11, 12, 13, 14], 'relation': 'in', 'target': Aircraft.Engine.HAS_PROPELLERS, 'result': True, 'alternate': False}],
    ['JENGSZ', {
        'val': 4, 'relation': '!=', 'target': Aircraft.Engine.SCALE_PERFORMANCE, 'result': True, 'alternate': False}],
    [Aircraft.HorizontalTail.VOLUME_COEFFICIENT, {
        'val': 0, 'relation': '==', 'target': Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, 'result': True, 'alternate': False}],
    [Aircraft.VerticalTail.VOLUME_COEFFICIENT, {
        'val': 0, 'relation': '==', 'target': Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, 'result': True, 'alternate': False}],
]
