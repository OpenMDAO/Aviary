"""
Fortran_to_Aviary.py is used to read in Fortran based vehicle decks and convert them to Aviary decks.

FLOPS, GASP, or Aviary names can be used for variables (Ex WG or Mission:Design:GROSS_MASS)
When specifying variables from FORTRAN, they should be in the appropriate NAMELIST.
Aviary variable names should be specified outside any NAMELISTS.
Names are not case-sensitive.
Units can be specified using any of the openMDAO valid units.
Comments can be added using !
Lists can be entered by separating values with commas.
Individual list elements can be specified by adding an index after the variable name.
(NOTE: 1 indexing is used inside NAMELISTS, while 0 indexing is used outside NAMELISTS)

Example inputs:
aircraft:fuselage:pressure_differential = .5, atm !DELP in GASP, but using atmospheres instead of psi
ARNGE(1) = 3600 !target range in nautical miles
pyc_phases = taxi, groundroll, rotation, landing
debug_mode = True
"""

import csv
import re
from enum import Enum
from pathlib import Path

from openmdao.utils.units import valid_units

from aviary.utils.functions import convert_strings_to_data
from aviary.utils.named_values import NamedValues, get_items
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.enums import LegacyCode
from aviary.utils.functions import get_path
from aviary.utils.legacy_code_data.deprecated_vars import flops_deprecated_vars, gasp_deprecated_vars


FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP


def create_aviary_deck(fortran_deck: str, legacy_code=None, defaults_deck=None,
                       out_file=None, force=False):
    '''
    Create an Aviary CSV file from a Fortran input deck
    Required input is the filepath to the input deck and legacy code. Optionally, a
    deck of default values can be specified, this is useful if an input deck
    assumes certain values for any unspecified variables
    If an invalid filepath is given, pre-packaged resources will be checked for
    input decks with a matching name.
    '''
    # TODO generate both an Aviary input file and a phase_info file

    vehicle_data = {'input_values': NamedValues(), 'unused_values': NamedValues(),
                    'initial_guesses': initial_guesses, 'debug_mode': False}

    fortran_deck: Path = get_path(fortran_deck, verbose=False)

    if not out_file:
        name = fortran_deck.stem
        out_file: Path = fortran_deck.parents / name + '_converted.csv'

    if legacy_code is GASP:
        default_extension = '.dat'
        deprecated_vars = gasp_deprecated_vars
    elif legacy_code is FLOPS:
        default_extension = '.txt'
        deprecated_vars = flops_deprecated_vars

    if not defaults_deck:
        defaults_filename = legacy_code.value.lower() + '_default_values' + default_extension
        defaults_deck = Path(__file__).parent.resolve().joinpath(
            'legacy_code_data', defaults_filename)

    # create dictionary to convert legacy code variables to Aviary variables
    aviary_variable_dict = generate_aviary_names([legacy_code.value])

    if defaults_deck:  # If defaults are specified, initialize the vehicle with them
        vehicle_data = input_parser(defaults_deck, vehicle_data,
                                    aviary_variable_dict, deprecated_vars, legacy_code)

    vehicle_data = input_parser(fortran_deck, vehicle_data,
                                aviary_variable_dict, deprecated_vars, legacy_code)
    if legacy_code is GASP:
        vehicle_data = update_gasp_options(vehicle_data)

    if not out_file.is_file():  # default outputted file to be in same directory as input
        out_file = fortran_deck.parent / out_file

    if out_file.is_file():
        if force:
            print(f'Overwriting existing file: {out_file.name}')
        else:
            raise RuntimeError(f'{out_file} already exists. Choose a new name or enable '
                               '--force')
    else:
        # create any directories defined by the new filename if they don't already exist
        out_file.parent.mkdir(parents=True, exist_ok=True)
        print('Writing to:', out_file)

    # open the file in write mode
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['debug_mode', vehicle_data['debug_mode']])
        writer.writerow([])

        # Values that have been successfully translated to Aviary variables
        writer.writerow(['# Input Values'])
        for var, (val, units) in sorted(vehicle_data['input_values']):
            writer.writerow([var] + val + [units])
        if legacy_code is FLOPS:
            EOM = 'height_energy'
            mass = 'FLOPS'
        if legacy_code is GASP:
            EOM = '2DOF'
            mass = 'GASP'
        writer.writerow(['settings:equations_of_motion'] + [EOM])
        writer.writerow(['settings:mass_method'] + [mass])

        if legacy_code is GASP:
            # Values used in initial guessing of the trajectory
            writer.writerow([])
            writer.writerow(['# Initial Guesses'])
            for var_name in sorted(vehicle_data['initial_guesses']):
                row = [var_name, vehicle_data['initial_guesses'][var_name]]
                writer.writerow(row)

        # Values that were not successfully converted
        writer.writerow([])
        writer.writerow(['# Unconverted Values'])
        for var, (val, _) in sorted(vehicle_data['unused_values']):
            writer.writerow([var] + val)


def input_parser(fortran_deck, vehicle_data, alternate_names, unused_vars, legacy_code):
    '''
    input_parser will modify the values in the vehicle_data dictionary using the data in the
    fortran_deck.
    Lines are read one by one, comments are removed, and namelists are tracked.
    Lines with multiple variable-data pairs are supported, but the last value per variable must
    be followed by a trailing comma.
    '''
    with open(fortran_deck, 'r') as f_in:
        current_namelist = current_tag = ''
        for line in f_in:
            terminate_namelist = False

            tmp = [*line.split('!', 1), '']
            line, comment = tmp[0], tmp[1]  # anything after the first ! is a comment

            # remove all white space and trailing commas
            line = ''.join(line.split()).rstrip(',')
            if len(line.split()) == 0:
                continue  # skip line if it contains only white space

            # Track when namelists are opened and closed
            if (line.lstrip()[0] in ['$', '&']) and current_tag == '':
                current_tag = line.lstrip()[0]
                current_namelist = line.split(current_tag)[1].split()[0] + '.'
            elif (line.lstrip()[0] == current_tag) or (line.rstrip()[-1] == '/'):
                line = line.replace('/', '')
                terminate_namelist = True

            number_of_variables = line.count('=')
            if number_of_variables == 1:
                # get the first element and remove white space
                var_name = ''.join(line.split('=')[0].split())
                # everything after the = is the data
                data = line.split('=')[1]
                try:
                    vehicle_data = process_and_store_data(
                        data, var_name, legacy_code, current_namelist, alternate_names, vehicle_data, unused_vars, comment)
                except Exception as err:
                    if current_namelist == '':
                        raise RuntimeError(line + ' could not be parsed successfully.'
                                           '\nIf this was intended as a comment, '
                                           'add an "!" at the beginning of the line.') from err
                    else:
                        raise err
            elif number_of_variables > 1:
                sub_line = line.split('=')  # split the line at each =
                var_name = sub_line[0]  # the first element is the first name
                for ii in range(number_of_variables):
                    # Each of the following elements contains all of the data for the current variable
                    # and the last element is the name of the next variable
                    sub_list = sub_line[ii+1].split(',')
                    if ii+1 < number_of_variables:
                        next_var_name = sub_list.pop()
                        if not next_var_name[0].isalpha():
                            index = next((i for i, c in enumerate(
                                next_var_name) if c.isalpha()), len(next_var_name))
                            sub_list.append(next_var_name[:index])
                            next_var_name = next_var_name[index:]

                    data = ','.join(sub_list)
                    try:
                        vehicle_data = process_and_store_data(
                            data, var_name, legacy_code, current_namelist, alternate_names, vehicle_data, unused_vars, comment)
                    except Exception as err:
                        if current_namelist == '':
                            raise RuntimeError(line + ' could not be parsed successfully.'
                                               '\nIf this was intended as a comment, '
                                               'add an "!" at the beginning of the line.') from err
                        else:
                            raise err
                    var_name = next_var_name

            if terminate_namelist:
                current_namelist = current_tag = ''

    return vehicle_data


def process_and_store_data(data, var_name, legacy_code, current_namelist, alternate_names, vehicle_data, unused_vars, comment=''):
    '''
    process_and_store_data takes in a string that contains the data, the current variable's name and
    namelist, the dictionary of alternate names, and the current vehicle data.
    It will convert the string of data into a list, get units, check whether the data specified is
    part of a list or a single element, and update the current name to it's equivalent Aviary name.
    The variables are also sorted based on whether they will set an Aviary variable or they are for initial guessing
    '''

    guess_names = list(initial_guesses.keys())
    var_ind = data_units = None
    skip_variable = False
    # skip any variables that shouldn't get converted
    if re.search(current_namelist+var_name+'\Z', str(unused_vars), re.IGNORECASE):
        return vehicle_data
    # remove any elements that are empty (caused by trailing commas or extra commas)
    data_list = [dat for dat in data.split(',') if dat != '']
    if len(data_list) > 0:
        if valid_units(data_list[-1]):
            # if the last element is a unit, remove it from the list and update the variable's units
            data_units = data_list.pop()
        var_values = convert_strings_to_data(data_list)
    else:
        skip_variable = True
        var_values = []

    if '(' in var_name:  # some GASP lists are given as individual elements
        # get the target index (Fortran uses 1 indexing, Python uses 0 indexing)
        fortran_offset = 1 if current_namelist else 0
        var_ind = int(var_name.split('(')[1].split(')')[0])-fortran_offset
        var_name = var_name.split('(')[0]  # remove the index formatting

    list_of_equivalent_aviary_names = update_name(
        alternate_names, current_namelist+var_name, vehicle_data['debug_mode'])
    for name in list_of_equivalent_aviary_names:
        if not skip_variable:
            if name == 'debug_mode':
                vehicle_data['debug_mode'] = var_values[0]
                continue

            elif name in guess_names and legacy_code is GASP:
                # all initial guesses take only a single value
                vehicle_data['initial_guesses'][name] = float(var_values[0])
                continue

            elif name in _MetaData:
                vehicle_data['input_values'] = set_value(name, var_values, vehicle_data['input_values'],
                                                         var_ind=var_ind, units=data_units)
                continue

        vehicle_data['unused_values'] = set_value(name, var_values, vehicle_data['unused_values'],
                                                  var_ind=var_ind, units=data_units)
        if vehicle_data['debug_mode']:
            print('Unused:', name, var_values, comment)

    return vehicle_data


def set_value(var_name, var_value, value_dict: NamedValues, var_ind=None, units=None):
    ''' 
    set_value will update the current value of a variable in a value dictionary that contains a value
    and it's associated units.
    If units are specified for the new value, they will be used, otherwise the current units in the
    value dictionary or the default units from _MetaData are used.
    If the new variable is part of a list, the current list will be extended if needed.
    '''

    if var_name in value_dict:
        current_value, units = value_dict.get_item(var_name)
    else:
        current_value = None
        if var_name in _MetaData:
            units = _MetaData[var_name]['units']
        else:
            units = 'unitless'
    if not units:
        units = 'unitless'

    if var_ind != None:
        # if an index is specified, use it, otherwise store the input as the whole value
        if isinstance(current_value, list):
            max_ind = len(current_value) - 1
            if var_ind > max_ind:
                current_value.extend((var_ind-max_ind)*[0])
        else:
            current_value = [current_value]+[0]*var_ind
        current_value[var_ind] = var_value[0]
        value_dict.set_val(var_name, current_value, units)
    else:
        value_dict.set_val(var_name, var_value, units)
    return value_dict


def generate_aviary_names(code_bases):
    '''
    Create a dictionary for each of the specified Fortran code bases to map to the Aviary
    variable names. Each dictionary of Aviary names will have a list of Fortran names for
    each variable
    '''

    alternate_names = {}
    for code_base in code_bases:
        alternate_names[code_base] = {}
        for key in _MetaData.keys():
            historical_dict = _MetaData[key]['historical_name']
            if historical_dict and code_base in historical_dict:
                alt_name = _MetaData[key]['historical_name'][code_base]
                if isinstance(alt_name, str):
                    alt_name = [alt_name]
                alternate_names[code_base][key] = alt_name
    return alternate_names


def update_name(alternate_names, var_name, debug_mode=False):
    '''update_name will convert a Fortran name to a list of equivalent Aviary names.'''

    all_equivalent_names = []
    for code_base in alternate_names.keys():
        for key, list_of_names in alternate_names[code_base].items():
            if list_of_names is not None:
                if any([re.search(var_name+r'\Z', altname, re.IGNORECASE) for altname in list_of_names]):
                    all_equivalent_names.append(key)

    # if there are no equivalent variable names, return the original name
    if len(all_equivalent_names) == 0:
        if debug_mode:
            print('passing: ', var_name)
        all_equivalent_names = [var_name]
    return all_equivalent_names


def update_gasp_options(vehicle_data):
    """
    Handles variables that are affected by the values of others
    """
    input_values: NamedValues = vehicle_data['input_values']

    flap_types = ["plain", "split", "single_slotted", "double_slotted",
                  "triple_slotted", "fowler", "double_slotted_fowler"]

    ## PROBLEM TYPE ##
    # if multiple values of target_range are specified, use the one that corresponds to the problem_type
    design_range, range_units = input_values.get_item(Mission.Design.RANGE)
    try:
        problem_type = input_values.get_val('problem_type')[0]
    except KeyError:
        problem_type = 'sizing'

    if isinstance(design_range, list):
        # if the design range target_range value is 0, set the problem_type to fallout
        if design_range[0] == 0:
            input_values.set_val('problem_type', ['fallout'])
            design_range = 0
        if problem_type == 'sizing':
            design_range = design_range[0]
        elif problem_type == 'alternate':
            design_range = design_range[2]
        elif problem_type == 'fallout':
            design_range = 0
    else:
        if design_range == 0:
            input_values.set_val('problem_type', ['fallout'])
    input_values.set_val(Mission.Design.RANGE, [design_range], range_units)

    ## STRUT AND FOLD ##
    strut_loc = input_values.get_val(Aircraft.Strut.ATTACHMENT_LOCATION, 'ft')[0]
    folded_span = input_values.get_val(Aircraft.Wing.FOLDED_SPAN, 'ft')[0]

    if strut_loc == 0:
        input_values.set_val(Aircraft.Wing.HAS_STRUT, [False], 'unitless')
    else:
        input_values.set_val(Aircraft.Wing.HAS_STRUT, [True], 'unitless')
    if folded_span == 0:
        input_values.set_val(Aircraft.Wing.HAS_FOLD, [False], 'unitless')
    else:
        input_values.set_val(Aircraft.Wing.HAS_FOLD, [True], 'unitless')

    if strut_loc < 0:
        input_values.set_val(Aircraft.Wing.HAS_FOLD, [True], 'unitless')
        input_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, [False], 'unitless')
        strut_loc = abs(strut_loc)

    if strut_loc < 1:
        input_values.set_val(Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS,
                             [strut_loc], 'unitless')
        input_values.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, [
                             False], 'unitless')
    else:
        input_values.set_val(Aircraft.Strut.ATTACHMENT_LOCATION, [strut_loc], 'ft')
        input_values.set_val(
            Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, [True], 'unitless')

    if input_values.get_val(Aircraft.Wing.HAS_FOLD)[0]:
        if not input_values.get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION)[0]:
            input_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                                 [True], 'unitless')
        else:
            if input_values.get_val(Aircraft.Wing.FOLDED_SPAN, 'ft')[0] > 1:
                input_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                                     [True], 'unitless')
            else:
                input_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                                     [False], 'unitless')
    else:
        input_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, [True], 'unitless')
        input_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, [
                             False], 'unitless')

    ## FLAPS ##
    flap_type = input_values.get_val(Aircraft.Wing.FLAP_TYPE)[0]
    if not isinstance(flap_type, str):
        flap_type = flap_types[flap_type-1]
        input_values.set_val(Aircraft.Wing.FLAP_TYPE, [flap_type])
    flap_ind = flap_types.index(flap_type)
    if input_values.get_val(Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT)[0] <= 0:
        input_values.set_val(Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
                             [[0.62, 1.0, 0.733, 1.2, 1.32, 0.633, 0.678][flap_ind]])
    if input_values.get_val(Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION, 'deg')[0] == 0:
        input_values.set_val(Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION,
                             [[60, 60, 40, 55, 55, 30, 30][flap_ind]], 'deg')
    if input_values.get_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM)[0] == 0:
        input_values.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
                             [[.9, .8, 1.18, 1.4, 1.6, 1.67, 2.25][flap_ind]])
    if input_values.get_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM)[0] == 0:
        input_values.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
                             [[.12, .23, .13, .23, .23, .1, .15][flap_ind]])

    vehicle_data['input_values'] = input_values
    return vehicle_data


def update_flops_options(vehicle_data):
    """
    Handles variables that are affected by the values of others
    """
    input_values: NamedValues = vehicle_data['input_values']

    for var_name in flops_scalar_variables.items():
        update_flops_scalar_variables(var_name, input_values)

    # TWR <= 0 is not valid in Aviary (parametric variation)
    if Aircraft.Design.THRUST_TO_WEIGHT_RATIO in input_values:
        if input_values.get_val(Aircraft.Design.THRUST_TO_WEIGHT_RATIO) <= 0:
            input_values.delete(Aircraft.Design.THRUST_TO_WEIGHT_RATIO)

    # WSR

    # Additional mass fraction scalar set to zero to not add mass twice
    if Aircraft.Engine.ADDITIONAL_MASS_FRACTION in input_values:
        if input_values.get_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION) >= 1:
            input_values.set_val(Aircraft.Engine.ADDITIONAL_MASS,
                                 input_values.get_val(
                                     Aircraft.Engine.ADDITIONAL_MASS_FRACTION),
                                 'lbm')
            input_values.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.0)

    # Miscellaneous propulsion mass trigger point 1 instead of 5
    if Aircraft.Propulsion.MISC_MASS_SCALER in input_values:
        if input_values.get_val(Aircraft.Propulsion.MISC_MASS_SCALER) >= 1:
            input_values.set_val(Aircraft.Propulsion.TOTAL_MISC_MASS,
                                 input_values.get_val(
                                     Aircraft.Propulsion.MISC_MASS_SCALER),
                                 'lbm')
            input_values.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, 0.0)

    vehicle_data['input_values'] = input_values
    return vehicle_data


def update_flops_scalar_variables(var_name, input_values: NamedValues):
    # The following parameters are used to modify or override
    # internally computed weights for various components as follows:
    # < 0., negative of starting weight which will be modified
    #   as appropriate during optimization or parametric
    #   variation, lb
    # = 0., no weight for that component
    # > 0. but < 5., scale factor applied to internally
    #   computed weight
    # > 5., actual fixed weight for component, lb
    # Same rules also applied to various other FLOPS scalar parameters
    scalar_name = var_name + '_scaler'
    if scalar_name not in input_values:
        return
    scalar_value = input_values[scalar_name]
    if scalar_value <= 0:
        input_values.delete(scalar_name)
    elif scalar_value < 5:
        return
    elif scalar_value > 5:
        input_values.set_val(var_name, scalar_value, 'lbm')
        input_values.set_val(scalar_name, 1.0)


# list storing information on Aviary variables that are split from single
# FLOPS variables that use the same value-based branching behavior
flops_scalar_variables = [
    Aircraft.AirConditioning.MASS,
    Aircraft.AntiIcing.MASS,
    Aircraft.APU.MASS,
    Aircraft.Avionics.MASS,
    Aircraft.Canard.MASS,
    Aircraft.Canard.WETTED_AREA,
    Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
    Aircraft.CrewPayload.FLIGHT_CREW_MASS,
    Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
    Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
    Aircraft.Design.EMPTY_MASS_MARGIN,
    Aircraft.Electrical.MASS,
    Aircraft.Engine.THRUST_REVERSERS_MASS,
    Aircraft.Fins.MASS,
    Aircraft.Fuel.FUEL_SYSTEM_MASS,
    Aircraft.Fuel.UNUSABLE_FUEL_MASS,
    Aircraft.Furnishings.MASS,
    Aircraft.Fuselage.MASS,
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.HorizontalTail.MASS,
    Aircraft.HorizontalTail.WETTED_AREA,
    Aircraft.Hydraulics.MASS,
    Aircraft.Instruments.MASS,
    Aircraft.LandingGear.MAIN_GEAR_MASS,
    Aircraft.LandingGear.NOSE_GEAR_MASS,
    Aircraft.Nacelle.MASS,
    Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
    Aircraft.VerticalTail.MASS_SCALER,
    Aircraft.VerticalTail.WETTED_AREA_SCALER,
    Aircraft.Wing.MASS,
    Aircraft.Wing.SHEAR_CONTROL_MASS,
    Aircraft.Wing.SURFACE_CONTROL_MASS,
    Aircraft.Wing.WETTED_AREA,
]

initial_guesses = {
    # initial_guesses is a dictionary that contains values used to initialize the trajectory
    'actual_takeoff_mass': 0,
    'rotation_mass': .99,
    'fuel_burn_per_passenger_mile': 0.1,
    'cruise_mass_final': 0,
    'flight_duration': 0,
    'time_to_climb': 0,
    'climb_range': 0,
    'reserves': 0
}


def _setup_F2A_parser(parser):
    '''
    Set up the subparser for the Fortran_to_aviary tool.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    '''
    parser.add_argument(
        "input_deck",
        type=str,
        nargs=1,
        help="Filename of vehicle input deck, including partial or complete path.",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        default=None,
        help="Filename for converted input deck, including partial or complete path."
    )
    parser.add_argument(
        "-l",
        "--legacy_code",
        type=LegacyCode,
        help="Name of the legacy code the deck originated from",
        choices=list(LegacyCode),
        required=True
    )
    parser.add_argument(
        "-d",
        "--defaults_deck",
        default=None,
        help="Deck of default values for unspecified variables"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing output files",
    )


def _exec_F2A(args, user_args):
    # check if args.input_deck is a list, if so, use the first element
    if isinstance(args.input_deck, list):
        args.input_deck = args.input_deck[0]
    filepath = args.input_deck

    create_aviary_deck(filepath, args.legacy_code, args.defaults_deck,
                       Path(args.out_file), args.force)
