import numpy as np
import openmdao.api as om
from pathlib import Path
import csv
import pkg_resources
import os

from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.variable_info.enums import ProblemType, EquationsOfMotion, LegacyCode
from aviary.variable_info.functions import add_aviary_output, add_aviary_input
from aviary.variable_info.variable_meta_data import _MetaData


class Null:
    '''
    This can be used to divert outputs, such as stdout, to improve performance
    '''

    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


def set_aviary_initial_values(model, inputs, meta_data=_MetaData):
    '''
    This function sorts through all the input
    variables to an Aviary model, and for those
    which are not options it sets the input
    value to be the value in the inputs, or
    to be the default if the value is not in the
    inputs.

    In the case when the value is not input nor
    present in the default, nothing is set.
    '''
    for key in meta_data:
        if ':' not in key or key.startswith('dynamic:'):
            continue
        if not meta_data[key]['option']:
            if key in inputs:
                val, units = inputs.get_item(key)
            else:
                val = meta_data[key]['default_value']
                units = meta_data[key]['units']

                if val is None:
                    # optional, but no default value
                    continue

            model.set_input_defaults(key, val=val, units=units)


def apply_all_values(aircraft_values: AviaryValues, prob):
    for var_name in get_keys(aircraft_values):
        var_data, var_units = aircraft_values.get_item(var_name)
        try:
            prob.set_val(var_name, val=var_data, units=var_units)
        except KeyError:
            pass
    return prob


def convert_strings_to_data(string_list):
    # convert_strings_to_data will convert a list of strings to usable data.
    # Strings that can't be converted to numbers will attempt to store as a logical,
    # otherwise they are passed as is
    value_list = [0]*len(string_list)
    for ii, dat in enumerate(string_list):
        dat = dat.strip('[]')
        try:
            # if the value is a number store it as a float or an int as appropriate
            # BUG this returns floats that can be converted to int (e.g. 1.0) as an int (1), even if the variable requires floats
            value_list[ii] = int(float(dat)) if float(
                dat).is_integer() else float(dat)
        except ValueError:
            # store value as a logical if it is a string that represents True or False
            if dat.lower() == 'true':
                value_list[ii] = True
            elif dat.lower() == 'false':
                value_list[ii] = False
            else:
                # if the value isn't a number or a logial, store it as a string
                value_list[ii] = dat
        except Exception as e:
            print('Exception', e)
    return value_list


def set_value(var_name, var_value, aviary_values: AviaryValues, units=None, is_array=False, meta_data=_MetaData):
    if var_name in aviary_values:
        current_value, current_units = aviary_values.get_item(var_name)
    else:
        current_value = meta_data[var_name]['default_value']
        current_units = meta_data[var_name]['units']

    if units == None:
        if current_units:
            units = current_units
        else:
            units = meta_data[var_name]['units']
        #     raise ValueError("You have specified a new variable without any units")

    if is_array:
        var_value = np.atleast_1d(var_value)
    elif len(var_value) == 1 and not isinstance(current_value, list):
        # if only a single value is provided, don't store it as a list
        var_value = var_value[0]

    # TODO handle enums in an automated method via checking metadata for enum type
    if var_name == 'problem_type':
        var_values = ProblemType[var_value]
    if var_name == 'settings:equations_of_motion':
        var_values = EquationsOfMotion(var_value)
    if var_name == 'settings:mass_method':
        var_values = LegacyCode(var_value)

    aviary_values.set_val(var_name, val=var_value, units=units, meta_data=meta_data)
    return aviary_values


def create_opts2vals(all_options: list, output_units: dict = {}):
    """
    create_opts2vals creates a component that converts options to outputs.

    Parameters
    ----------
    all_options : list of strings
        Each string is the name of an option in aviary_options.
    output_units : dict of units, optional
        This optional input allows the user to specify the units that will be used while
        adding the outputs. Only the outputs that shouldn't use their default units need
        to be specified. Each key should match one of the names in all_options, and each
        value must be a string representing a valid unit in openMDAO.

    Returns
    -------
    OptionsToValues : ExplicitComponent
        An explicit component that takes in an AviaryValues object that contains any
        options that need to be converted to outputs. There are no inputs to this
        component, only outputs. If the resulting component is added directly to a
        Group, the output variables will have the same name as the options they
        represent. If you need to rename them to prevent conflicting names in the
        group, running add_opts2vals will add the prefix "option:" to the name.
    """

    def configure_output(option_name: str, aviary_options: AviaryValues):
        option_data = aviary_options.get_item(option_name)
        out_units = output_units[option_name] if option_name in output_units.keys(
        ) else option_data[1]
        return {'val': option_data[0], 'units': out_units}

    class OptionsToValues(om.ExplicitComponent):
        def initialize(self):
            self.options.declare(
                'aviary_options', types=AviaryValues,
                desc='collection of Aircraft/Mission specific options'
            )

        def setup(self):
            for option_name in all_options:
                output_data = configure_output(
                    option_name, self.options['aviary_options'])
                add_aviary_output(self, option_name,
                                  val=output_data['val'], units=output_data['units'])

        def compute(self, inputs, outputs):
            aviary_options: AviaryValues = self.options['aviary_options']
            for option_name in all_options:
                output_data = configure_output(option_name, aviary_options)
                outputs[option_name] = aviary_options.get_val(
                    option_name, units=output_data['units'])

    return OptionsToValues


def add_opts2vals(Group: om.Group, OptionsToValues, aviary_options: AviaryValues):
    """
    Add the OptionsToValues component to the specified Group.

    Parameters
    ----------
    Group : Group
        The group or model the component should be added to.
    OptionsToValues : ExplicitComponent
        This is the explicit component that was created by create_opts2vals.
    aviary_options : AviaryValues
        aviary_options is an AviaryValues object that contains all of the options
        that need to be converted to outputs.

    Returns
    -------
    Opts2Vals : Group
        A group that wraps the OptionsToValues component in order to rename its
        variables with a prefix to keep them separate from any similarly named
        variables in the original group the component is being added to.
    """

    class Opts2Vals(om.Group):
        def initialize(self):
            self.options.declare(
                'aviary_options', types=AviaryValues,
                desc='collection of Aircraft/Mission specific options'
            )

        def setup(self):
            self.add_subsystem('options_to_values', OptionsToValues(
                aviary_options=aviary_options))

        def configure(self):
            all_output_data = self.options_to_values.list_outputs(out_stream=None)
            list_of_outputs = [(name, 'option:'+name) for name, data in all_output_data]
            self.promotes('options_to_values', list_of_outputs)

    Group.add_subsystem('opts2vals', Opts2Vals(
        aviary_options=aviary_options),
        promotes_outputs=['*'])

    return Group


def create_printcomp(all_inputs: list, input_units: dict = {}, meta_data=_MetaData):
    """
    Creates a component that prints the value of all inputs.

    Parameters
    ----------
    all_inputs : list of strings
        Each string is the name of a variable in the system
    input_units : dict of units, optional
        This optional input allows the user to specify the units that will be used while
        adding the inputs. Only the variables that shouldn't use their default units need
        to be specified. Each key should match one of the names in all_inputs, and each
        value must be a string representing a valid unit in openMDAO.

    Returns
    -------
    PrintComp : ExplicitComponent
        An explicit component that can be added to a group to print the current values
        of connected variables. There are no outputs from this component, only inputs.
    """

    def get_units(variable_name):
        if variable_name in input_units.keys():
            return input_units[variable_name]
        elif variable_name in meta_data:
            return meta_data[variable_name]['units']
        else:
            return None

    class PrintComp(om.ExplicitComponent):

        def setup(self):
            for variable_name in all_inputs:
                units = get_units(variable_name)
                if ':' in variable_name:
                    add_aviary_input(self, variable_name, units=units)
                else:
                    self.add_input(variable_name, units=units)

        def compute(self, inputs, outputs):
            print_string = ['v'*20]
            for variable_name in all_inputs:
                units = get_units(variable_name)
                print_string.append('{} {} {}'.format(
                    variable_name, inputs[variable_name], units))
            print_string.append('^'*20)
            print('\n'.join(print_string))

    return PrintComp


def promote_aircraft_and_mission_vars(group):
    external_outputs = []
    for comp in group.system_iter(recurse=False):

        # Skip all aviary systems.
        if comp.name == 'core_subsystems':
            continue

        out_names = [item for item in comp._var_allprocs_prom2abs_list['output']]
        in_names = [item for item in comp._var_allprocs_prom2abs_list['input']]

        external_outputs.extend(out_names)

        # Locally promote aircraft:* and mission:* only.
        promote_in = []
        promote_out = []

        for stem in ['mission:', 'aircraft:', 'dynamic:']:
            for name in out_names:
                if name.startswith(stem):
                    promote_out.append(f'{stem}*')
                    break

            for name in in_names:
                if name.startswith(stem):
                    promote_in.append(f'{stem}*')
                    break

        group.promotes(comp.name, outputs=promote_out, inputs=promote_in)

    return external_outputs


def get_path(path: [str, Path], verbose: bool = False) -> Path:
    """
    Convert a string or Path object to an absolute Path object, prioritizing different locations.

    This function attempts to find the existence of a path in the following order:
    1. As an absolute path.
    2. Relative to the current working directory.
    3. Relative to the Aviary package.

    If the path cannot be found in any of the locations, a FileNotFoundError is raised.

    Parameters
    ----------
    path : str or Path
        The input path, either as a string or a Path object.
    verbose : bool, optional
        If True, prints the final path being used. Default is False.

    Returns
    -------
    Path
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the path is not found in any of the prioritized locations.
    """

    # Store the original path for reference in error messages.
    original_path = path

    # If the input is a string, convert it to a Path object.
    if isinstance(path, str):
        path = Path(path)

    # Check if the path exists as an absolute path.
    if not path.exists():
        # If not, try finding the path relative to the current working directory.
        path = Path.cwd() / path

    # If the path still doesn't exist, attempt to find it relative to the Aviary package.
    if not path.exists():
        # Determine the path relative to the Aviary package.
        aviary_based_path = Path(
            pkg_resources.resource_filename('aviary', original_path))
        if verbose:
            print(
                f"Unable to locate '{original_path}' as an absolute or relative path. Trying Aviary package path: {aviary_based_path}")
        path = aviary_based_path

    # If the path still doesn't exist in any of the prioritized locations, raise an error.
    if not path.exists():
        raise FileNotFoundError(
            f'File not found in absolute path: {original_path}, relative path:{Path.cwd() / path}, or Aviary-based path: {Path(pkg_resources.resource_filename("aviary", original_path))}'
        )

    # If verbose is True, print the path being used.
    if verbose:
        print(f'Using {path} for file.')

    return path
