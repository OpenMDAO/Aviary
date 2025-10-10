import atexit
from contextlib import ExitStack
import os
from pathlib import Path
from typing import Union
import warnings

import importlib_resources
import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues, get_items
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variable_meta_data import _MetaData


class Null:
    """This can be used to divert outputs, such as stdout, to improve performance."""

    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


def get_aviary_resource_path(resource_name: str) -> Path:
    """
    Get the file path of a resource in the Aviary package.

    Parameters
    ----------
        resource_name : str
            The name of the resource.

    Returns
    -------
        path : Path
            The file path of the resource.

    """
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    if resource_name:
        ref = importlib_resources.files('aviary') / resource_name
    else:
        ref = importlib_resources.files('aviary')
    path = file_manager.enter_context(importlib_resources.as_file(ref))
    return path


top_dir = Path(get_aviary_resource_path(''))


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
    for key, (val, units) in get_items(aviary_inputs):
        try:
            prob.set_val(key, val, units)

        except BaseException:
            # Should be an option or an overridden output.
            continue


def set_aviary_input_defaults(model, inputs, aviary_inputs: AviaryValues, meta_data=_MetaData):
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
        List of variables that are causing promotion problems. This needs to
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


def convert_strings_to_data(input_string):
    """
    convert_strings_to_data will convert a string or list of strings to usable data.
    Strings that can't be converted to numbers will attempt to store as a boolean,
    otherwise they are passed as is.
    """
    # pack input_string into a list if it is not
    # setup output list size
    if isinstance(input_string, list):
        islist = True
        value_list = [0] * len(input_string)
    else:
        islist = False
        input_string = [input_string]
        value_list = input_string

    for ii, dat in enumerate(input_string):
        dat = dat.strip('[]')
        try:
            # if the value is a number store it as a float or an int as appropriate
            # BUG this returns floats that can be converted to int (e.g. 1.0) as an
            # int (1), even if the variable requires floats
            value_list[ii] = int(dat) if '.' not in dat else float(dat)
        except ValueError:
            # store value as a boolean if it is a string that represents True or False
            if dat.lower() == 'true':
                value_list[ii] = True
            elif dat.lower() == 'false':
                value_list[ii] = False
            else:
                # if the value isn't a number or a boolean, store it as a string
                value_list[ii] = dat
    # unpack output value from list if it isn't supposed to be one
    if not islist:
        value_list = value_list[0]
    return value_list


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
        out_units = (
            output_units[option_name] if option_name in output_units.keys() else option_data[1]
        )
        return {'val': option_data[0], 'units': out_units}

    class OptionsToValues(om.ExplicitComponent):
        def initialize(self):
            self.options.declare(
                'aviary_options',
                types=AviaryValues,
                desc='collection of Aircraft/Mission specific options',
            )

        def setup(self):
            for option_name in all_options:
                output_data = configure_output(option_name, self.options['aviary_options'])
                add_aviary_output(
                    self,
                    option_name,
                    val=output_data['val'],
                    units=output_data['units'],
                )

        def compute(self, inputs, outputs):
            aviary_options: AviaryValues = self.options['aviary_options']
            for option_name in all_options:
                output_data = configure_output(option_name, aviary_options)
                # uses default value if not present
                if option_name in aviary_options:
                    outputs[option_name] = aviary_options.get_val(
                        option_name, units=output_data['units']
                    )

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
                'aviary_options',
                types=AviaryValues,
                desc='collection of Aircraft/Mission specific options',
            )

        def setup(self):
            self.add_subsystem('options_to_values', OptionsToValues(aviary_options=aviary_options))

        def configure(self):
            all_output_data = self.options_to_values.list_outputs(out_stream=None)
            list_of_outputs = [(name, 'option:' + name) for name, data in all_output_data]
            self.promotes('options_to_values', list_of_outputs)

    Group.add_subsystem(
        'opts2vals', Opts2Vals(aviary_options=aviary_options), promotes_outputs=['*']
    )

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
                        add_aviary_input(self, variable_name, units=units, shape=num_nodes)
                    except TypeError:
                        self.add_input(variable_name, units=units, shape=num_nodes, val=1.23456)
                else:
                    # using an arbitrary number that will stand out for unconnected
                    # variables
                    self.add_input(variable_name, units=units, shape=num_nodes, val=1.23456)

        def compute(self, inputs, outputs):
            print_string = ['v' * 20]
            for variable_name in all_inputs:
                units = get_units(variable_name)
                print_string.append('{} {} {}'.format(variable_name, inputs[variable_name], units))
            print_string.append('^' * 20)
            print('\n'.join(print_string))

    return PrintComp


def promote_aircraft_and_mission_vars(group):
    """Promotes inputs and outputs in Aircraft and Mission hierarchy categories for provided group."""
    external_outputs = []
    for comp in group.system_iter(recurse=False):
        # Skip all aviary systems.
        if comp.name == 'core_subsystems':
            continue

        try:
            resolver = comp._resolver
            out_names = [item for item in resolver.prom_iter(iotype='output')]
            in_names = [item for item in resolver.prom_iter(iotype='input')]

        except AttributeError:
            # This is an older version of OpenMDAO
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


def get_path(path: Union[str, Path], verbosity=Verbosity.BRIEF) -> Path:
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
    verbosity : Verbosity, optional
        Sets level of printouts for this function.

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
        if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
            print(
                f"Unable to locate '{original_path}' as an absolute or relative path. "
                'Trying Aviary package path.'
            )
        # Determine the path relative to the Aviary package.
        aviary_based_path = Path(get_aviary_resource_path(original_path))

        path = aviary_based_path

    # If the path still doesn't exist, attempt to find it in the models directory.
    if not path.exists():
        if verbosity > Verbosity.BRIEF:
            print(
                f"Unable to locate '{aviary_based_path}' as an Aviary package path, "
                'checking built-in models'
            )
        try:
            hangar_based_path = get_model(original_path)
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

    # Print the path being used.
    if verbosity > Verbosity.BRIEF:
        print(f'Found {path}')

    return path


def get_model(file_name: str, verbosity=Verbosity.BRIEF) -> Path:
    """
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
    """
    # Get the path to Aviary's models
    aviary_path = Path(get_aviary_resource_path(str(Path('models', file_name))))
    # Check if provided path is valid
    if aviary_path.exists():
        return aviary_path
    # otherwise check models folder contents
    else:
        from glob import glob

        contents = glob(str(get_aviary_resource_path('models') / '**'), recursive=True)
        close_match = None
        for item in contents:
            item = Path(item)
            # check if full filepath, file name with extension, or just file (or folder) name
            # matches target
            if aviary_path == item or aviary_path.name == item.name:
                return item
            elif aviary_path.stem == item.stem:
                close_match = item

    if close_match is not None:
        # Probably requested the wrong file extension.
        return close_match

    # If the path doesn't exist, raise an error.
    raise FileNotFoundError("File or Folder not found in Aviary's hangar")


def sigmoidX(x, x0, mu=1.0):
    """
    Sigmoid used to smoothly transition between piecewise functions.

    Parameters
    ----------
    x: float or array
        independent variable
    x0: float
        the center of symmetry. When x = x0, sigmoidX = 1/2.
    mu: float
        steepness parameter.

    Returns
    -------
    float or array
        smoothed value from input parameter x.
    """
    if mu == 0:
        raise ValueError('mu must be non-zero')

    if isinstance(x, np.ndarray):
        if np.isrealobj(x):
            dtype = float
        else:
            dtype = complex
        n_size = x.size
        y = np.zeros(n_size, dtype=dtype)
        # avoid overflow in squared term, underflow seems to be ok
        calc_idx = np.where((x.real - x0) / mu > -320)
        y[calc_idx] = 1 / (1 + np.exp(-(x[calc_idx] - x0) / mu))
    else:
        if isinstance(x, float):
            dtype = float
        else:
            dtype = complex
        y = 0
        if (x - x0) * mu > -320:
            y = 1 / (1 + np.exp(-(x - x0) / mu))
    if dtype == float:
        y = y.real
    return y


def dSigmoidXdx(x, x0, mu=1.0):
    """
    Derivative of sigmoid function.

    Parameters
    ----------
    x: float or array
        independent variable
    x0: float
        the center of symmetry. When x = x0, sigmoidX = 1/2.
    mu: float
        steepness parameter.

    Returns
    -------
    float or array
        smoothed derivative value from input parameter x.
    """
    if mu == 0:
        raise ValueError('mu must be non-zero')

    if isinstance(x, np.ndarray):
        if np.isrealobj(x):
            dtype = float
        else:
            dtype = complex
        n_size = x.size
        y = np.zeros(n_size, dtype=dtype)
        term = np.zeros(n_size, dtype=dtype)
        term2 = np.zeros(n_size, dtype=dtype)
        # avoid overflow in squared term, underflow seems to be ok
        calc_idx = np.where((x.real - x0) / mu > -320)
        term[calc_idx] = np.exp(-(x[calc_idx] - x0) / mu)
        term2[calc_idx] = (1 + term[calc_idx]) * (1 + term[calc_idx])
        y[calc_idx] = term[calc_idx] / mu / term2[calc_idx]
    else:
        y = 0
        if (x - x0) * mu > -320:
            term = np.exp(-(x - x0) / mu)
            term2 = (1 + term) * (1 + term)
            y = term / mu / term2
    if dtype == float:
        y = y.real
    return y


def smooth_min(x, b, mu=100.0):
    """
    Smooth approximation of the min function using the log-sum-exp trick.

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true minimum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of min(x, b).
    """
    sum_log_exp = np.log(np.exp(np.multiply(-mu, x)) + np.exp(np.multiply(-mu, b)))
    rv = -(1 / mu) * sum_log_exp
    return rv


def d_smooth_min(x, b, mu=100.0):
    """
    Derivative of function smooth_min(x)

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true minimum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of derivative of min(x, b).
    """
    d_sum_log_exp = np.exp(np.multiply(-mu, x)) / (
        np.exp(np.multiply(-mu, x)) + np.exp(np.multiply(-mu, b))
    )
    return d_sum_log_exp


def smooth_max(x, b, mu=10.0):
    """
    Smooth approximation of the min function using the log-sum-exp trick.

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true maximum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of max(x, b).
    """
    mu_x = mu * x
    mu_b = mu * b
    m = np.maximum(mu_x, mu_b)
    sum_log_exp = (m + np.log(np.exp(mu_x - m) + np.exp(mu_b - m))) / mu

    return sum_log_exp


def d_smooth_max(x, b, mu=10.0):
    """
    Derivative of function smooth_min(x)

    Parameters:
    x (float or array-like): First value.
    b (float or array-like): Second value.
    mu (float): The smoothing factor. Higher values make it closer to the true minimum. Try between 75 and 275.

    Returns:
    float or array-like: The smooth approximation of derivative of min(x, b).
    """
    mu_x = mu * x
    mu_b = mu * b
    m = np.maximum(mu_x, mu_b)
    numerator = np.exp(mu_x - m)
    denominator = np.exp(mu_x - m) + np.exp(mu_b - m)
    d_sum_log_exp = mu * numerator / denominator
    return d_sum_log_exp


def sin_int4(val):
    """Define a smooth, differentialbe approximation to the 'int' function."""
    return sin_int(sin_int(sin_int(sin_int(val)))) - 0.5


def dydx_sin_int4(val):
    """Define the derivative (dy/dx) of sin_int4, at x = val."""
    y0 = sin_int(val)
    y1 = sin_int(y0)
    y2 = sin_int(y1)

    dydx3 = dydx_sin_int(y2)
    dydx2 = dydx_sin_int(y1)
    dydx1 = dydx_sin_int(y0)
    dydx0 = dydx_sin_int(val)

    dydx = dydx3 * dydx2 * dydx1 * dydx0

    return dydx


# 'int' function can be approximated by recursively applying this sin function
# which makes a smooth, differentialbe function (is there a good one?)
def sin_int(val):
    """
    Define one step in approximating the 'int' function with a smooth,
    differentialbe function.
    """
    int_val = val - np.sin(2 * np.pi * (val + 0.5)) / (2 * np.pi)

    return int_val


def dydx_sin_int(val):
    """Define the derivative (dy/dx) of sin_int, at x = val."""
    dydx = 1.0 - np.cos(2 * np.pi * (val + 0.5))

    return dydx


def smooth_int_tanh(x, mu=10.0):
    """
    Smooth approximation of int(x) using tanh.
    """
    f = np.floor(x)
    frac = x - f
    t = np.tanh(mu * (frac - 0.5))
    s = 0.5 * (t + 1)
    y = f + s
    return y


def d_smooth_int_tanh(x, mu=10.0):
    """
    Smooth approximation of int(x) using tanh.
    Returns (y, dy_dx).
    """
    f = np.floor(x)
    frac = x - f
    t = np.tanh(mu * (frac - 0.5))
    dy_dx = 0.5 * mu * (1 - t**2)
    return dy_dx
