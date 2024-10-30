from typing import Union
from pathlib import Path
import importlib_resources
from contextlib import ExitStack
import atexit
import os

import openmdao.api as om
import numpy as np
from openmdao.utils.units import convert_units

from aviary.utils.aviary_values import AviaryValues, get_items
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


def get_aviary_resource_path(resource_name: str) -> str:
    """
    Get the file path of a resource in the Aviary package.

    Args:
        resource_name (str): The name of the resource.

    Returns:
        str: The file path of the resource.

    """
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    ref = importlib_resources.files('aviary') / resource_name
    path = file_manager.enter_context(
        importlib_resources.as_file(ref))
    return path


def set_aviary_initial_values(prob, aviary_inputs: AviaryValues):
    """
    Sets initial values for all inputs in the aviary inputs.

    This method is mostly used in tests and level 3 scripts.

    Parameters
    ----------
    prob : Problem
        OpenMDAO problem after setup.
    aviary_inputs : AviaryValues
        Instance of AviaryValues containing all initial values.
    """
    for (key, (val, units)) in get_items(aviary_inputs):
        try:
            prob.set_val(key, val, units)

        except:
            # Should be an option or an overridden output.
            continue


def set_aviary_input_defaults(model, inputs, aviary_inputs: AviaryValues,
                              meta_data=_MetaData):
    """
    This function sets the default values and units for any inputs prior to
    setup. This is needed to resolve ambiguities when inputs are promoted
    with the same name, but different units or values.

    This method is mostly used in tests and level 3 scripts.

    Parameters
    ----------
    model : System
        Top level aviary model.
    inputs : list
        List of varibles that are causing promotion problems. This needs to
        be crafted based on the openmdao exception messages.
    aviary_inputs : AviaryValues
        Instance of AviaryValues containing all initial values.
    meta_data : dict
        (Optional) Dictionary of aircraft metadata. Uses Aviary's built-in
        metadata by default.
    """
    for key in inputs:
        if key in aviary_inputs:
            val, units = aviary_inputs.get_item(key)
        else:
            val = meta_data[key]['default_value']
            units = meta_data[key]['units']

        model.set_input_defaults(key, val=val, units=units)


def convert_strings_to_data(string_list):
    """
    convert_strings_to_data will convert a list of strings to usable data.
    Strings that can't be converted to numbers will attempt to store as a logical,
    otherwise they are passed as is
    """
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


# TODO this function is only used in a single place (process_input_decks.py), and its
#      functionality can get handled in other places (convert_strings_to_data being able
#      to handle lists/arrays, and other special handling directly present in
#      process_input_decks.py)
def set_value(var_name, var_value, aviary_values: AviaryValues, units=None, is_array=False, meta_data=_MetaData):
    """
    Wrapper for AviaryValues.set_val(). Existing value/units of the provided variable name are used as defaults if
    they exist and not provided in this function. Special list handling provided: if 'is_array' is true, 'var_value' is
    always added to 'aviary_values' as a numpy array. Otherwise, if 'var_value' is a list or numpy array of length
    one and existing value in 'aviary_values' or default value in 'meta_data' is not a list or numpy array,
    individual value is pulled out of 'var_value' to be stored in 'aviary_values'.
    """
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
    elif len(var_value) == 1 and not isinstance(current_value, (list, np.ndarray)):
        # if only a single value is provided, don't store it as a list
        var_value = var_value[0]

    # TODO handle enums in an automated method via checking metadata for enum type
    if var_name == 'settings:problem_type':
        var_value = ProblemType(var_value)
    if var_name == 'settings:equations_of_motion':
        var_value = EquationsOfMotion(var_value)
    if var_name == 'settings:mass_method':
        var_value = LegacyCode(var_value)

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


def create_printcomp(all_inputs: list, input_units: dict = {}, meta_data=_MetaData, num_nodes=1):
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
                    try:
                        add_aviary_input(self, variable_name,
                                         units=units, shape=num_nodes)
                    except TypeError:
                        self.add_input(variable_name, units=units,
                                       shape=num_nodes, val=1.23456)
                else:
                    # using an arbitrary number that will stand out for unconnected variables
                    self.add_input(variable_name, units=units,
                                   shape=num_nodes, val=1.23456)

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
    """
    Promotes inputs and outputs in Aircraft and Mission hierarchy categories for provided group.
    """
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


# Python 3.10 adds the ability to specify multiple types using type hints like so:
# "str | Path" which is cleaner but Aviary still supports older versions


def get_model(file_name: str, verbose=False) -> Path:
    '''
    This function attempts to find the path to a file or folder in aviary/models
    If the path cannot be found in any of the locations, a FileNotFoundError is raised.

    Parameters
    ----------
    path : str or Path
        The input path, either as a string or a Path object.

    Returns
    -------
    aviary_path
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the path is not found.
    '''

    # Get the path to Aviary's models
    path = Path('models', file_name)
    aviary_path = Path(get_aviary_resource_path(str(path)))

    # If the file name was provided without a path, check in the subfolders
    if not aviary_path.exists():
        sub_dirs = [x[0] for x in os.walk(get_aviary_resource_path('models'))]
        for sub_dir in sub_dirs:
            temp_path = Path(sub_dir, file_name)
            if temp_path.exists():
                # only return the first matching file
                aviary_path = temp_path
                continue

    # If the path still doesn't exist, raise an error.
    if not aviary_path.exists():
        raise FileNotFoundError(
            f"File or Folder not found in Aviary's hangar"
        )
    if verbose:
        print('found', aviary_path, '\n')
    return aviary_path


def get_path(path: Union[str, Path], verbose: bool = False) -> Path:
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
        relative_path = Path.cwd() / path
        path = relative_path

    # If the path still doesn't exist, attempt to find it relative to the Aviary package.
    if not path.exists():
        # Determine the path relative to the Aviary package.
        aviary_based_path = Path(
            get_aviary_resource_path(original_path))
        if verbose:
            print(
                f"Unable to locate '{original_path}' as an absolute or relative path. Trying Aviary package path: {aviary_based_path}")
        path = aviary_based_path

    # If the path still doesn't exist, attempt to find it in the models directory.
    if not path.exists():
        try:
            hangar_based_path = get_model(original_path)
            if verbose:
                print(
                    f"Unable to locate '{aviary_based_path}' as an Aviary package path, checking built-in models")
            path = hangar_based_path
        except FileNotFoundError:
            pass

    # If the path still doesn't exist in any of the prioritized locations, raise an error.
    if not path.exists():
        raise FileNotFoundError(
            f'File not found in absolute path: {original_path}, relative path: '
            f'{relative_path}, or Aviary-based path: '
            f'{Path(get_aviary_resource_path(original_path))}'
        )

    # If verbose is True, print the path being used.
    if verbose:
        print(f'Using {path} for file.')

    return path


def wrapped_convert_units(val_unit_tuple, new_units):
    """
    Wrapper for OpenMDAO's convert_units function.

    Parameters
    ----------
    val_unit_tuple : tuple
        Tuple of the form (value, units) where value is a float and units is a
        string.
    new_units : str
        New units to convert to.

    Returns
    -------
    float
        Value converted to new units.
    """
    value, units = val_unit_tuple

    # can't convert units on None; return None
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return [convert_units(v, units, new_units) for v in value]
    else:
        return convert_units(value, units, new_units)
