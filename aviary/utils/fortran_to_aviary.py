"""
fortran_to_aviary.py is used to read in Fortran based vehicle decks and convert them to Aviary decks.

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
"""

import csv
import getpass
import re
from datetime import datetime
from pathlib import Path

from openmdao.utils.units import valid_units

from aviary.utils.functions import convert_strings_to_data, get_path
from aviary.utils.legacy_code_data.flops_defaults import flops_default_values, flops_deprecated_vars
from aviary.utils.legacy_code_data.gasp_defaults import gasp_default_values, gasp_deprecated_vars
from aviary.utils.named_values import NamedValues
from aviary.variable_info.enums import LegacyCode, Verbosity
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Settings

FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP


def fortran_to_aviary(
    fortran_deck: str,
    legacy_code=None,
    out_file=None,
    force=False,
    verbosity=Verbosity.BRIEF,
):
    """
    Create an Aviary CSV file from a Fortran input deck
    Required input is the filepath to the input deck and legacy code. Optionally, a
    deck of default values can be specified, this is useful if an input deck
    assumes certain values for any unspecified variables
    If an invalid filepath is given, pre-packaged resources will be checked for
    input decks with a matching name.
    """
    # compatibility with being passed int for verbosity
    verbosity = Verbosity(verbosity)

    # TODO generate both an Aviary input file and a phase_info file

    vehicle_data = {
        'input_values': NamedValues(),
        'unused_values': NamedValues(),
        'initialization_guesses': initialization_guesses,
    }

    fortran_deck: Path = get_path(fortran_deck, verbosity=verbosity)

    timestamp = datetime.now().strftime('%m/%d/%y at %H:%M')
    user = getpass.getuser()
    comments = []

    comments.append(f'# created {timestamp} by {user}')
    comments.append(
        f'# {legacy_code.value}-derived aircraft input deck converted from {fortran_deck.name}'
    )

    if out_file:
        out_file = Path(out_file)
    else:
        name = fortran_deck.stem
        out_file: Path = fortran_deck.parent.resolve().joinpath(name + '_converted.csv')

    # create dictionary to convert legacy code variables to Aviary variables
    # key: variable name, value: either None or relevant historical_name
    aviary_variable_dict = generate_aviary_names(legacy_code.value)

    # Get legacy-code based depreciated variable list and set vehicle data to defaults
    if legacy_code is GASP:
        default_values = gasp_default_values
        deprecated_vars = gasp_deprecated_vars
    elif legacy_code is FLOPS:
        default_values = flops_default_values
        deprecated_vars = flops_deprecated_vars

    # Convert default data to Aviary names, add to vehicle_data
    for item in default_values:
        name = item[0].split('.')
        val = str(item[1][0])
        vehicle_data = process_and_store_data(
            data=val,
            var_name=name[1],
            legacy_code=legacy_code,
            current_namelist=name[0],
            alternate_names=aviary_variable_dict,
            default_values=default_values,
            vehicle_data=vehicle_data,
            unused_vars=deprecated_vars,
            verbosity=verbosity,
        )

    # read in and convert input file
    vehicle_data = parse_input_file(
        fortran_deck,
        vehicle_data,
        aviary_variable_dict,
        default_values,
        deprecated_vars,
        legacy_code,
        verbosity,
    )

    # Postprocessing step to handle special cases for conversion (not 1-to-1 match),
    # per legacy code.
    if legacy_code is GASP:
        vehicle_data = update_gasp_options(vehicle_data, verbosity)
    elif legacy_code is FLOPS:
        vehicle_data = update_flops_options(vehicle_data)
    vehicle_data = update_aviary_options(vehicle_data)

    # Add settings and engine data file
    if legacy_code is FLOPS:
        eom = ['height_energy']
        aero = mass = ['FLOPS']
    if legacy_code is GASP:
        eom = ['2DOF']
        aero = mass = ['GASP']
    vehicle_data['input_values'].set_val(Settings.EQUATIONS_OF_MOTION, eom)
    vehicle_data['input_values'].set_val(Settings.MASS_METHOD, mass)
    vehicle_data['input_values'].set_val(Settings.AERODYNAMICS_METHOD, aero)

    if not out_file.is_file():
        # default outputted file to be in same directory as input
        out_file = fortran_deck.parent / out_file

    if out_file.is_file():
        if not force:
            raise RuntimeError(f'{out_file} already exists. Choose a new name or enable --force')
        elif verbosity >= Verbosity.BRIEF:
            print(f'Overwriting existing file: {out_file.name}')

    else:
        # create any directories defined by the new filename if they don't already exist
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if verbosity >= Verbosity.VERBOSE:
            print('Writing to:', out_file)

    # TODO Use the existing utilities to write this input file? It will be much more
    #      human-readable
    # open the file in write mode
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header info and comments
        for comment in comments:
            writer.writerow([comment])
        writer.writerow([])
        # Values that have been successfully translated to Aviary variables
        writer.writerow(['# Input Values'])
        for var, (val, units) in sorted(vehicle_data['input_values']):
            writer.writerow([var] + val + [units])

        if legacy_code is GASP:
            # Values used in initial guessing of the trajectory
            writer.writerow([])
            writer.writerow(['# Initialization Guesses'])
            for var_name in sorted(vehicle_data['initialization_guesses']):
                row = [var_name, vehicle_data['initialization_guesses'][var_name]]
                writer.writerow(row)

        # Values that were not successfully converted
        writer.writerow([])
        writer.writerow(['# Unconverted Values'])
        for var, (val, _) in sorted(vehicle_data['unused_values']):
            writer.writerow([var] + val)


def parse_input_file(
    fortran_deck,
    vehicle_data,
    alternate_names,
    default_values,
    unused_vars,
    legacy_code,
    verbosity=Verbosity.BRIEF,
):
    """
    parse_input_file reads the data in fortran_deck and adds it to vehicle_data.

    Lines are read one by one, comments are removed, and namelists are tracked.
    Lines with multiple variable-data pairs are supported, but the last value per
    variable must be followed by a trailing comma.
    """
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
                current_namelist = line.split(current_tag)[1].split()[0]
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
                        data,
                        var_name,
                        legacy_code,
                        current_namelist,
                        alternate_names,
                        default_values,
                        vehicle_data,
                        unused_vars,
                        comment,
                        verbosity,
                    )
                except Exception as err:
                    if current_namelist == '':
                        raise RuntimeError(
                            line + ' could not be parsed successfully.'
                            '\nIf this was intended as a comment, '
                            'add an "!" at the beginning of the line.'
                        ) from err
                    else:
                        raise err
            elif number_of_variables > 1:
                sub_line = line.split('=')  # split the line at each =
                var_name = sub_line[0]  # the first element is the first name
                for ii in range(number_of_variables):
                    # Each of the following elements contains all of the data for the current variable
                    # and the last element is the name of the next variable
                    sub_list = sub_line[ii + 1].split(',')
                    if ii + 1 < number_of_variables:
                        next_var_name = sub_list.pop()
                        if not next_var_name[0].isalpha():
                            index = next(
                                (i for i, c in enumerate(next_var_name) if c.isalpha()),
                                len(next_var_name),
                            )
                            sub_list.append(next_var_name[:index])
                            next_var_name = next_var_name[index:]

                    data = ','.join(sub_list)
                    try:
                        vehicle_data = process_and_store_data(
                            data,
                            var_name,
                            legacy_code,
                            current_namelist,
                            alternate_names,
                            default_values,
                            vehicle_data,
                            unused_vars,
                            comment,
                            verbosity,
                        )
                    except Exception as err:
                        if current_namelist == '':
                            raise RuntimeError(
                                line + ' could not be parsed successfully.'
                                '\nIf this was intended as a comment, '
                                'add an "!" at the beginning of the line.'
                            ) from err
                        else:
                            raise err
                    var_name = next_var_name

            if terminate_namelist:
                current_namelist = current_tag = ''

    return vehicle_data


def process_and_store_data(
    data,
    var_name,
    legacy_code,
    current_namelist,
    alternate_names,
    default_values,
    vehicle_data,
    unused_vars,
    comment='',
    verbosity=Verbosity.BRIEF,
):
    """
    process_and_store_data takes in a string that contains the data, the current variable's name and
    namelist, the dictionary of alternate names, and the current vehicle data.
    It will convert the string of data into a list, get units, check whether the data specified is
    part of a list or a single element, and update the current name to it's equivalent Aviary name.
    The variables are also sorted based on whether they will set an Aviary variable or they are for initial guessing.
    """
    guess_names = list(initialization_guesses.keys())
    var_ind = data_units = None
    skip_variable = False
    # skip any variables that shouldn't get converted
    if re.search(current_namelist + '.' + var_name, str(unused_vars), re.IGNORECASE):
        return vehicle_data
    # remove any elements that are empty (caused by trailing commas or extra commas)
    data_list = [dat for dat in data.split(',') if dat != '']
    if len(data_list) > 0:
        if valid_units(data_list[-1]):
            # if the last element is a unit, remove it from the list and update the
            # variable's units
            data_units = data_list.pop()
        var_values = convert_strings_to_data(data_list)
    else:
        skip_variable = True
        var_values = []

    list_of_equivalent_aviary_names, var_ind = update_name(
        alternate_names, current_namelist + '.' + var_name, verbosity
    )

    # Fortran uses 1 indexing, Python uses 0 indexing
    fortran_offset = 1 if current_namelist else 0
    if var_ind is not None:
        var_ind -= fortran_offset

    # Aviary has a reduction gearbox which is 1/gear ratio of GASP gearbox
    if current_namelist + '.' + var_name == 'INPROP.GR':
        var_values = [1 / var for var in var_values]
        vehicle_data['input_values'] = set_value(
            Aircraft.Engine.Gearbox.GEAR_RATIO,
            var_values,
            vehicle_data['input_values'],
            var_ind=var_ind,
            units=data_units,
        )

    for name in list_of_equivalent_aviary_names:
        if not skip_variable:
            if name in guess_names and legacy_code is GASP:
                # all initial guesses take only a single value
                vehicle_data['initialization_guesses'][name] = float(var_values[0])
                continue

            elif name in _MetaData:
                if current_namelist + '.' + var_name in default_values:
                    data_units = default_values.get_item(current_namelist + '.' + var_name)[1]
                else:
                    data_units = None
                vehicle_data['input_values'] = set_value(
                    name,
                    var_values,
                    data_units,
                    vehicle_data['input_values'],
                    var_ind=var_ind,
                )
                continue

        vehicle_data['unused_values'] = set_value(
            name,
            var_values,
            data_units,
            vehicle_data['unused_values'],
            var_ind=var_ind,
        )
        if verbosity >= Verbosity.VERBOSE:
            print('Unused:', name, var_values, comment)

    return vehicle_data


def set_value(var_name, var_value, units=None, value_dict: NamedValues = None, var_ind=None):
    """
    set_value will update the current value of a variable in a value dictionary that contains a value
    and it's associated units.
    If units are specified for the new value, they will be used, otherwise the current units in the
    value dictionary or the default units from _MetaData are used.
    If the new variable is part of a list, the current list will be extended if needed.
    """
    if var_name in value_dict:
        current_value, units = value_dict.get_item(var_name)
    else:
        current_value = None
        if var_name in _MetaData:
            if not units:
                units = _MetaData[var_name]['units']
    if not units:
        units = 'unitless'

    if var_ind is not None:
        # if an index is specified, use it, otherwise store the input as the whole value
        if isinstance(current_value, list):
            max_ind = len(current_value) - 1
            if var_ind > max_ind:
                current_value.extend((var_ind - max_ind) * [0])
        else:
            current_value = [current_value] + [0] * var_ind
        current_value[var_ind] = var_value[0]
        value_dict.set_val(var_name, current_value, units)
    else:
        if current_value is not None and isinstance(current_value[0], bool):
            # if a variable is defined as boolean but is read in as number, set as
            # boolean
            if var_value[0] == 1:
                var_value = ['True']
            elif var_value[0] == 0:
                var_value = ['False']
            else:
                ValueError(f'{var_name} allows 0 and 1 only, but it is {var_value[0]}')
        value_dict.set_val(var_name, var_value, units)
    return value_dict


def generate_aviary_names(legacy_code):
    """
    Create a dictionary that maps the specified Fortran code to Aviary variable names.
    Each Aviary variable will have a list of matching Fortran names.
    """
    alternate_names = {}
    for key in _MetaData.keys():
        historical_dict = _MetaData[key]['historical_name']
        if historical_dict and legacy_code in historical_dict:
            alt_name = _MetaData[key]['historical_name'][legacy_code]
            if isinstance(alt_name, str):
                alt_name = [alt_name]
            alternate_names[key] = alt_name
    return alternate_names


def update_name(alternate_names, var_name, verbosity=Verbosity.BRIEF):
    """update_name will convert a Fortran name to a list of equivalent Aviary names."""
    if '(' in var_name:  # some GASP lists are given as individual elements
        # get the target index
        var_ind = int(var_name.split('(')[1].split(')')[0])
        var_name = var_name.split('(')[0]  # remove the index formatting
    else:
        var_ind = None

    all_equivalent_names = []
    for key, list_of_names in alternate_names.items():
        if list_of_names is not None:
            for altname in list_of_names:
                altname = altname.lower()
                if altname.endswith(var_name.lower()):
                    all_equivalent_names.append(key)
                    continue
                elif var_ind is not None and altname.endswith(f'{var_name.lower()}({var_ind})'):
                    all_equivalent_names.append(key)
                    var_ind = None
                    continue

    # if there are no equivalent variable names, return the original name
    if len(all_equivalent_names) == 0:
        if verbosity >= Verbosity.VERBOSE:
            print('passing: ', var_name)
        all_equivalent_names = [var_name]

    return all_equivalent_names, var_ind


def update_gasp_options(vehicle_data, verbosity=Verbosity.BRIEF):
    """Handles variables that are affected by the values of others."""
    input_values: NamedValues = vehicle_data['input_values']

    for var_name in gasp_scaler_variables:
        update_gasp_scaler_variables(var_name, input_values)

    flap_types = [
        'plain',
        'split',
        'single_slotted',
        'double_slotted',
        'triple_slotted',
        'fowler',
        'double_slotted_fowler',
    ]

    design_type, design_units = input_values.get_item(Aircraft.Design.TYPE)
    if design_type[0] == 0:
        input_values.set_val(Aircraft.Design.TYPE, ['transport'], design_units)
    elif design_type[0] == 1:
        input_values.set_val(Aircraft.Design.TYPE, ['BWB'], design_units)

    ## PROBLEM TYPE ##
    # if multiple values of target_range are specified, use the one that
    # corresponds to the problem_type
    design_range, distance_units = input_values.get_item(Mission.Design.RANGE)
    try:
        problem_type = input_values.get_val(Settings.PROBLEM_TYPE)[0]
    except KeyError:
        problem_type = 'sizing'

    if isinstance(design_range, list):
        # if the design range target_range value is 0, set the problem_type to fallout
        if design_range[0] == 0:
            problem_type = 'fallout'
            input_values.set_val(Settings.PROBLEM_TYPE, [problem_type])
            design_range = 0
        if problem_type == 'sizing':
            design_range = design_range[0]
        elif problem_type == 'alternate':
            design_range = design_range[2]
        elif problem_type == 'fallout':
            design_range = 0
    else:
        if design_range == 0:
            input_values.set_val(Settings.PROBLEM_TYPE, ['fallout'])
    input_values.set_val(Mission.Design.RANGE, [design_range], distance_units)

    ## Passengers ##
    if Aircraft.CrewPayload.Design.NUM_PASSENGERS in input_values:
        num_passengers = input_values.get_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, 'unitless'
        )[0]
        num_passengers = int(num_passengers)
        input_values.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, [num_passengers], 'unitless'
        )

    if Aircraft.CrewPayload.Design.NUM_FIRST_CLASS in input_values:
        # In GASP, percentage of total number of passengers is given.
        # Convert it to the actual first class passengers.
        pct_first_class = input_values.get_val(
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 'unitless'
        )[0]
        num_first_class = int(pct_first_class * num_passengers)
        input_values.set_val(
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, [num_first_class], 'unitless'
        )
        num_tourist_class = num_passengers - num_first_class
        input_values.set_val(
            Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, [num_tourist_class], 'unitless'
        )

    ## Seats ##
    if Aircraft.Fuselage.NUM_AISLES in input_values:
        num_aisles = input_values.get_val(Aircraft.Fuselage.NUM_AISLES, 'unitless')[0]
        num_aisles = int(num_aisles)
        input_values.set_val(Aircraft.Fuselage.NUM_AISLES, [num_aisles], 'unitless')
        num_seat_abreast_tourist = input_values.get_val(
            Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 'unitless'
        )[0]
        num_seat_abreast_tourist = int(num_seat_abreast_tourist)
        input_values.set_val(
            Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, [num_seat_abreast_tourist]
        )
        try:
            num_seat_abreast_first = int(
                input_values.get_val(
                    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 'unitless'
                )[0]
            )
            input_values.set_val(
                Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, [num_seat_abreast_first]
            )
        except:
            pass
        try:
            num_seat_abreast_business = int(
                input_values.get_val(
                    Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 'unitless'
                )[0]
            )
            input_values.set_val(
                Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, [num_seat_abreast_business]
            )
        except:
            pass

    ## Cargo ##
    if (
        Aircraft.CrewPayload.CARGO_MASS in input_values
        and Aircraft.CrewPayload.Design.MAX_CARGO_MASS not in input_values
    ):
        # user has set cargo only: assume intention to set max only for backwards compatibility.
        cargo, units = input_values.get_item(Aircraft.CrewPayload.Design.CARGO_MASS)
        input_values.set_val(Aircraft.CrewPayload.Design.MAX_CARGO_MASS, cargo, units)
        input_values.set_val(Aircraft.CrewPayload.CARGO_MASS, 0, units)
        input_values.set_val(Aircraft.CrewPayload.Design.CARGO_MASS, 0, units)

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
        if strut_loc > 0:
            input_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, [False], 'unitless')
        elif strut_loc == 0:
            input_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, [True], 'unitless')

    if strut_loc < 0:
        input_values.set_val(Aircraft.Wing.HAS_FOLD, [True], 'unitless')
        input_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, [False], 'unitless')
        strut_loc = abs(strut_loc)

    if strut_loc < 1:
        input_values.set_val(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, [strut_loc], 'unitless'
        )
        input_values.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, [False], 'unitless')
    else:
        input_values.set_val(Aircraft.Strut.ATTACHMENT_LOCATION, [strut_loc], 'ft')
        input_values.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, [True], 'unitless')

    if input_values.get_val(Aircraft.Wing.HAS_FOLD)[0]:
        if not input_values.get_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION)[0]:
            input_values.set_val(
                Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, [True], 'unitless'
            )
        else:
            if input_values.get_val(Aircraft.Wing.FOLDED_SPAN, 'ft')[0] > 1:
                input_values.set_val(
                    Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                    [True],
                    'unitless',
                )
            else:
                input_values.set_val(
                    Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                    [False],
                    'unitless',
                )
    else:
        input_values.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, [True], 'unitless')
        input_values.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, [False], 'unitless')

    ## FLAPS ##
    flap_type = input_values.get_val(Aircraft.Wing.FLAP_TYPE)[0]
    if not isinstance(flap_type, str):
        flap_type = flap_types[flap_type - 1]
        input_values.set_val(Aircraft.Wing.FLAP_TYPE, [flap_type])
    flap_ind = flap_types.index(flap_type)
    if input_values.get_val(Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT)[0] <= 0:
        input_values.set_val(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT,
            [[0.62, 1.0, 0.733, 1.2, 1.32, 0.633, 0.678][flap_ind]],
        )
    if input_values.get_val(Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION, 'deg')[0] == 0:
        input_values.set_val(
            Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION,
            [[60, 60, 40, 55, 55, 30, 30][flap_ind]],
            'deg',
        )
    if input_values.get_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM)[0] == 0:
        input_values.set_val(
            Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
            [[0.9, 0.8, 1.18, 1.4, 1.6, 1.67, 2.25][flap_ind]],
        )
    if input_values.get_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM)[0] == 0:
        input_values.set_val(
            Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
            [[0.12, 0.23, 0.13, 0.23, 0.23, 0.1, 0.15][flap_ind]],
        )
    if Aircraft.Wing.NUM_FLAP_SEGMENTS in input_values:
        num_flap_segments = input_values.get_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, 'unitless')[0]
        num_flap_segments = int(num_flap_segments)
        input_values.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, [num_flap_segments], 'unitless')

    ## Fuel ##
    reserve_fuel_additional = input_values.get_val(
        Aircraft.Design.RESERVE_FUEL_ADDITIONAL, units='lbm'
    )[0]
    if reserve_fuel_additional <= 0:
        input_values.set_val(Aircraft.Design.RESERVE_FUEL_ADDITIONAL, [0], units='lbm')
        input_values.set_val(
            Aircraft.Design.RESERVE_FUEL_FRACTION,
            [-reserve_fuel_additional],
            units='unitless',
        )
    elif reserve_fuel_additional >= 10:
        input_values.set_val(Aircraft.Design.RESERVE_FUEL_FRACTION, [0], units='unitless')
    else:
        ValueError('"FRESF" is not valid between 0 and 10.')

    if Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR in input_values:
        if input_values.get_val(Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR)[0] > 4:
            if verbosity > Verbosity.BRIEF:
                print(
                    'When XLFMX > 4, it is landing flare initiation height (ft), '
                    'not landing flare load factor.'
                )
            input_values.delete(Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR)

    # if the value is negative, we are asking the code to calculate it
    # if it is positive, then we are going to use it as an override
    if input_values.get_val(Aircraft.Wing.FORM_FACTOR)[0] < 0:
        input_values.delete(Aircraft.Wing.FORM_FACTOR)
    if input_values.get_val(Aircraft.HorizontalTail.FORM_FACTOR)[0] < 0:
        input_values.delete(Aircraft.HorizontalTail.FORM_FACTOR)
    if input_values.get_val(Aircraft.VerticalTail.FORM_FACTOR)[0] < 0:
        input_values.delete(Aircraft.VerticalTail.FORM_FACTOR)
    if input_values.get_val(Aircraft.Fuselage.FORM_FACTOR)[0] < 0:
        input_values.delete(Aircraft.Fuselage.FORM_FACTOR)
    if input_values.get_val(Aircraft.Nacelle.FORM_FACTOR)[0] < 0:
        input_values.delete(Aircraft.Nacelle.FORM_FACTOR)
    if Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR in input_values:
        if input_values.get_val(Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR)[0] < 0:
            input_values.delete(Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR)

    # GASP-converted engine decks have uneven throttle ranges, which require the enabling
    # of global throttle range. This will result in extrapolation of the engine deck,
    # but provides closer matches to legacy results. To remove use of global throttle
    # (and therefore eliminate extrapolation), a T4 limit needs to be manually set for
    # the mission
    input_values.set_val(Aircraft.Engine.GLOBAL_THROTTLE, [True])

    # GEARBOX
    # Aviary has a reduction gearbox which is 1/gear ratio of GASP gearbox
    if Aircraft.Engine.Gearbox.GEAR_RATIO in input_values:
        ratios = input_values.get_val(Aircraft.Engine.Gearbox.GEAR_RATIO)
        ratios = [1 / val for val in ratios]
        input_values.set_val(Aircraft.Engine.Gearbox.GEAR_RATIO, ratios, units='unitless')

    # CARGO
    if Aircraft.CrewPayload.Design.MAX_CARGO_MASS in input_values:
        if input_values.get_val(Aircraft.CrewPayload.Design.MAX_CARGO_MASS, 'lbm')[0] >= 0:
            input_values.set_val(Aircraft.CrewPayload.Design.CARGO_MASS, [0.0], 'lbm')
            input_values.set_val(Aircraft.CrewPayload.CARGO_MASS, [0.0], 'lbm')

    # ENGINE
    if Aircraft.Engine.NUM_ENGINES in input_values:
        num_engines = input_values.get_val(Aircraft.Engine.NUM_ENGINES, 'unitless')[0]
        num_engines = int(num_engines)
    else:
        num_engines = 1
    input_values.set_val(Aircraft.Engine.NUM_ENGINES, [num_engines], 'unitless')
    if Aircraft.Design.TYPE in input_values:
        design_type = input_values.get_val(Aircraft.Design.TYPE, 'unitless')[0]
        if design_type == 'BWB':
            num_fuselage_engines = num_engines
            # assume all engines are fuselage engines
            input_values.set_val(
                Aircraft.Engine.NUM_FUSELAGE_ENGINES, [num_fuselage_engines], 'unitless'
            )
            # BWB engine sizing algorithm does not use reference diameter
            input_values.delete(Aircraft.Engine.REFERENCE_DIAMETER)
    else:
        input_values.set_val(Aircraft.Design.TYPE, 'transport')
    if Aircraft.Engine.TYPE in input_values:
        engine_type = input_values.get_val(Aircraft.Engine.TYPE, 'unitless')[0]
        if verbosity > Verbosity.BRIEF:
            print(
                f'Engine type {engine_type} was provided; currently only TURBOPROP(6) and '
                'TURBOJET(7) are supported by Aviary'
            )

    # FURNISHING
    if Aircraft.Furnishings.MASS in input_values:
        furnishing_mass_scaler = input_values.get_val(Aircraft.Furnishings.MASS, 'lbm')[0]
        if furnishing_mass_scaler < 0:
            furnishing_mass_scaler = abs(furnishing_mass_scaler)
            input_values.set_val(Aircraft.Furnishings.MASS_SCALER, [furnishing_mass_scaler], 'lbm')
            input_values.delete(Aircraft.Furnishings.MASS)

    unused_values = vehicle_data['unused_values']
    knac = unused_values.get_item('INGASP.KNAC')[0][0]
    if knac != 2:
        try:
            input_values.delete(Aircraft.Nacelle.AVG_DIAMETER)
            input_values.delete(Aircraft.Nacelle.AVG_LENGTH)
        except:
            pass

    # Variables required by GASP, but no default values are provided in GASP
    missing_vars = []
    if not Aircraft.Wing.ZERO_LIFT_ANGLE in input_values:
        missing_vars.append('ALPHL0')
    if not Aircraft.Wing.ASPECT_RATIO in input_values:
        missing_vars.append('AR')
    if not Aircraft.HorizontalTail.ASPECT_RATIO in input_values:
        missing_vars.append('ARHT')
    if not Aircraft.VerticalTail.ASPECT_RATIO in input_values:
        missing_vars.append('ARVT')
    if not Aircraft.Design.PART25_STRUCTURAL_CATEGORY in input_values:
        missing_vars.append('CATD')
    if not Aircraft.Fuselage.PRESSURE_DIFFERENTIAL in input_values:
        missing_vars.append('DELP')
    if not Mission.Taxi.DURATION in input_values:
        missing_vars.append('DELTT')
    if not Aircraft.Wing.FLAP_DEFLECTION_LANDING in input_values:
        missing_vars.append('DFLPLD')
    if not Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF in input_values:
        missing_vars.append('DFLPTO')
    if not Aircraft.Wing.SWEEP in input_values:
        missing_vars.append('DLMC4')
    if not Aircraft.Wing.INCIDENCE in input_values:
        missing_vars.append('EYEW')
    if not Aircraft.CrewPayload.Design.NUM_PASSENGERS in input_values:
        missing_vars.append('PAX')
    if not Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST in input_values:
        missing_vars.append('PS')
    if not Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST in input_values:
        missing_vars.append('SAB')
    if not Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION in input_values:
        missing_vars.append('SAH')
    if not Aircraft.Wing.TAPER_RATIO in input_values:
        missing_vars.append('SLM')
    if not Aircraft.HorizontalTail.TAPER_RATIO in input_values:
        missing_vars.append('SLMH')
    if not Aircraft.VerticalTail.TAPER_RATIO in input_values:
        missing_vars.append('SLMV')
    if not Aircraft.HorizontalTail.THICKNESS_TO_CHORD in input_values:
        missing_vars.append('TCHT')
    if not Aircraft.Wing.THICKNESS_TO_CHORD_ROOT in input_values:
        missing_vars.append('TCR')
    if not Aircraft.Wing.THICKNESS_TO_CHORD_TIP in input_values:
        missing_vars.append('TCT')
    if not Aircraft.VerticalTail.THICKNESS_TO_CHORD in input_values:
        missing_vars.append('TCVT')
    if not Aircraft.Nacelle.MASS_SPECIFIC in input_values:
        missing_vars.append('UWNAC')
    if not Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS in input_values:
        missing_vars.append('UWPAX')
    if not Aircraft.Design.MAX_STRUCTURAL_SPEED in input_values:
        missing_vars.append('VMLFSL')
    if not Aircraft.Fuselage.AISLE_WIDTH in input_values:
        missing_vars.append('WAS')
    if not Aircraft.Fuselage.SEAT_WIDTH in input_values:
        missing_vars.append('WS')
    if not Aircraft.LandingGear.MAIN_GEAR_LOCATION in input_values:
        missing_vars.append('YMG')
    if not Aircraft.Engine.WING_LOCATIONS in input_values:
        missing_vars.append('YP')
    if not Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION in input_values:
        missing_vars.append('SAH')
    if len(missing_vars) > 0:
        raise RuntimeError(
            f'The following variables are required but are not provided:\n {missing_vars}'
        )

    vehicle_data['input_values'] = input_values
    return vehicle_data


def update_flops_options(vehicle_data):
    """Handles variables that are affected by the values of others."""
    input_values: NamedValues = vehicle_data['input_values']

    for var_name in flops_scaler_variables:
        update_flops_scaler_variables(var_name, input_values)

    # TODO TWR should be checked for and a comment added that T/W ratio should be constrained to the
    # value found in the input file - TWR != Aircraft.Design.THRUST_RATIO!!!!
    # TWR <= 0 is not valid in Aviary (parametric variation)
    # if Aircraft.Design.THRUST_TO_WEIGHT_RATIO in input_values:
    #     if input_values.get_val(Aircraft.Design.THRUST_TO_WEIGHT_RATIO)[0] <= 0:
    #         input_values.delete(Aircraft.Design.THRUST_TO_WEIGHT_RATIO)

    # WSR
    # Additional mass fraction scaler set to zero to not add mass twice
    if Aircraft.Engine.ADDITIONAL_MASS_FRACTION in input_values:
        if input_values.get_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION)[0] >= 1:
            input_values.set_val(
                Aircraft.Engine.ADDITIONAL_MASS,
                input_values.get_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION),
                'lbm',
            )
            input_values.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, [0.0])

    # Miscellaneous propulsion mass trigger point 1 instead of 5
    if Aircraft.Propulsion.MISC_MASS_SCALER in input_values:
        if input_values.get_val(Aircraft.Propulsion.MISC_MASS_SCALER)[0] >= 1:
            input_values.set_val(
                Aircraft.Propulsion.TOTAL_MISC_MASS,
                input_values.get_val(Aircraft.Propulsion.MISC_MASS_SCALER),
                'lbm',
            )
            input_values.set_val(Aircraft.Propulsion.MISC_MASS_SCALER, [0.0])

    if Aircraft.Fuel.DENSITY in input_values:
        # Interpret value equivalently to FULDEN (FLOPS fuel density ratio relative to jet fuel 6.7 lbm/galUS) and convert to an absolute fuel density)
        input_values.set_val(
            Aircraft.Fuel.DENSITY,
            [6.7 * input_values.get_val(Aircraft.Fuel.DENSITY, 'lbm/galUS')[0]],
            'lbm/galUS',
        )
    # else: not required as jet fuel is assumed and default value in metadata is 6.7

    if Aircraft.Fuel.WING_FUEL_CAPACITY in input_values:
        if input_values.get_val(Aircraft.Fuel.WING_FUEL_CAPACITY, 'lbm')[0] < 50:
            # Interpret value equivalently to FWMAX = wing_fuel_fraction * fuel_density * 2/3
            FWMAX = input_values.get_val(Aircraft.Fuel.WING_FUEL_CAPACITY, 'lbm')[0]
            FULDEN = input_values.get_val(Aircraft.Fuel.DENSITY, 'lbm/ft**3')[0]
            input_values.set_val(
                Aircraft.Fuel.WING_FUEL_FRACTION, [FWMAX / (FULDEN * (2 / 3))], 'unitless'
            )
            input_values.delete(Aircraft.Fuel.WING_FUEL_CAPACITY)

    # Set detailed wing flag if model supports it
    if Aircraft.Wing.INPUT_STATION_DIST in input_values:
        input_values.set_val(Aircraft.Wing.DETAILED_WING, [True])

    vehicle_data['input_values'] = input_values
    return vehicle_data


def update_aviary_options(vehicle_data):
    """Special handling for variables that occurs for either legacy code."""
    input_values: NamedValues = vehicle_data['input_values']

    # if reference + scaled thrust both provided, set scale factor
    try:
        ref_thrust = input_values.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 'lbf')[0]
        ref_thrust = float(ref_thrust)
        input_values.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, [ref_thrust], 'lbf')
    except KeyError:
        pass

    vehicle_data['input_values'] = input_values
    return vehicle_data


def update_flops_scaler_variables(var_name, input_values: NamedValues):
    """
    The following parameters are used to modify or override
    internally computed weights and areas for various components as follows:
    < 0., negative of starting weight which will be modified
    as appropriate during optimization or parametric variation, lb or ft**2
    = 0., no weight for that component
    > 0. but < 5., scale factor applied to internally computed weight or area
    > 5., actual fixed weight for component, lb or ft**2
    Same rules also applied to various other FLOPS scaler parameters.
    """
    scaler_name = var_name + '_scaler'
    if scaler_name not in input_values:
        return
    scaler_value = input_values.get_val(scaler_name)[0]
    if scaler_value <= 0:
        input_values.delete(scaler_name)
    elif scaler_value < 5:
        return
    elif scaler_value > 5:
        if 'area' in var_name.lower():
            input_values.set_val(var_name, [scaler_value], 'ft**2')
        else:
            input_values.set_val(var_name, [scaler_value], 'lbm')
        input_values.delete(scaler_name)


def update_gasp_scaler_variables(var_name, input_values: NamedValues):
    """
    The following parameters are used to modify or override
    internally computed weights and areas for various components as follows:
    < 0., negative of starting weight which will be modified
    as appropriate during optimization or parametric variation, lb or ft**2
    = 0., no weight/area for that component
    > 0. but < 10., scale factor applied to internally computed weight
    > 10., actual fixed weight for component, lb or ft**2
    Same rules also applied to various other FLOPS scaler parameters.
    """
    scaler_name = var_name + '_scaler'
    if scaler_name not in input_values:
        return
    scaler_value = input_values.get_val(scaler_name)[0]
    if scaler_value <= 0:
        input_values.delete(scaler_name)
    elif scaler_value < 10:
        return
    elif scaler_value > 10:
        if 'area' in var_name.lower():
            input_values.set_val(var_name, [scaler_value], 'ft**2')
        else:
            input_values.set_val(var_name, [scaler_value], 'lbm')
        input_values.delete(scaler_name)


# list storing information on Aviary variables that are split from single
# FLOPS variables that use the same value-based branching behavior
flops_scaler_variables = [
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
    Aircraft.Nacelle.WETTED_AREA,
    Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
    Aircraft.VerticalTail.MASS,
    Aircraft.VerticalTail.WETTED_AREA,
    Aircraft.Wing.MASS,
    Aircraft.Wing.SHEAR_CONTROL_MASS,
    Aircraft.Wing.SURFACE_CONTROL_MASS,
    Aircraft.Wing.WETTED_AREA,
]

# GASP variables that use the same value-based branching behavior
gasp_scaler_variables = [
    Aircraft.Fuselage.WETTED_AREA,
]

initialization_guesses = {
    # initialization_guesses is a dictionary that contains values used to
    # initialize the trajectory
    'actual_takeoff_mass': 0,
    'rotation_mass': 0,
    'fuel_burn_per_passenger_mile': 0,
    'cruise_mass_final': 0,
    'flight_duration': 0,
    'time_to_climb': 0,
    'climb_range': 0,
    'reserves': 0,
}


def _setup_F2A_parser(parser):
    """
    Set up the subparser for the Fortran_to_aviary tool.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        'input_deck',
        type=str,
        nargs=1,
        help='Filename of vehicle input deck, including partial or complete path.',
    )
    parser.add_argument(
        '-o',
        '--out_file',
        default=None,
        help='Filename for converted input deck, including partial or complete path.',
    )
    parser.add_argument(
        '-l',
        '--legacy_code',
        type=LegacyCode,
        help='Name of the legacy code the deck originated from',
        choices=set(LegacyCode),
        required=True,
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Allow overwriting existing output files',
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        type=int,
        choices=Verbosity.values(),
        default=1,
        help='Set level of print statements',
    )


def _exec_F2A(args, user_args):
    # check if args.input_deck is a list, if so, use the first element
    if isinstance(args.input_deck, list):
        args.input_deck = args.input_deck[0]
    filepath = args.input_deck

    # convert verbosity from int to enum
    verbosity = Verbosity(args.verbosity)

    fortran_to_aviary(filepath, args.legacy_code, args.out_file, args.force, verbosity)
