import inspect
import subprocess
import tempfile
import os
import numpy as np


"""
# DocTAPE #
------------
Documentation Testing and Automated Placement of Expressions

A collection of utility functions (and wrappers for Glue) that are useful
for automating the process of building and testing documentation to ensure that
documentation doesn't get stale.

expected_error is an execption that can be used in try/except blocks to allow desired errors to
pass while still raising unexpected errors.

gramatical_list combines the elements of a list into a string with proper punctuation
check_value is a simple function for comparing two values
check_contains confirms that all the elements of one iterable are contained in the other
check_args gets the signature of a function and compares it to the arguments you are expecting
run_command_no_file_error executes a CLI command but won't fail if a FileNotFoundError is raised
get_attribute_name gets the name of an object's attribute based on it's value
get_all_keys recursively get all of the keys from a dict of dicts
get_value recursively get a value from a dict of dicts
glue_variable Glue a variable for later use in markdown cells of notebooks (can auto format for code)
glue_keys recursively glue all of the keys from a dict of dicts
"""


class expected_error(Exception):
    ...


def gramatical_list(list_of_strings: list, cc='and', add_accents=False) -> str:
    """
    Combines the elements of a list into a string with proper punctuation

    Parameters
    ----------
    list_of_strings : list
        A list of strings (or elements with a string representation)
    cc : str, optional
        The coordinating conjunction to use with the list (default is `and`)
    add_accents : bool, optional
        Whether or not to wrap each element with ` characters (default is False)

    Returns
    -------
    str
        A string that combines the elements of the list into a string with proper punctuation
    """
    list_of_strings = ['`'+str(s)+'`' if add_accents else str(s)
                       for s in list_of_strings]
    if len(list_of_strings) == 1:
        return list_of_strings[0]
    elif len(list_of_strings) == 2:
        return list_of_strings[0]+' '+cc+' '+list_of_strings[1]
    else:
        return ', '.join(list_of_strings[:-1]+[cc+' '+list_of_strings[-1]])


def get_previous_line(n=1) -> str:
    """
    returns the previous n line(s) of code as a string

    Parameters
    ----------
    n : int
        The number of lines to return (default is 1)

    Returns
    -------
    str
        A string that contains the previous line of code or a
        list that contains the previous n lines of code
    """
    pframe = inspect.currentframe().f_back  # get the previous frame that called this function
    # get the lines of code as a list of strings
    lines, first_line = inspect.getsourcelines(pframe)
    # get the line number of the line that called this function
    lineno = pframe.f_lineno - first_line if first_line else pframe.f_lineno - 1
    # get the previous lines
    return lines[lineno-n:lineno] if n > 1 else lines[lineno-1].strip()


def get_variable_name(*variables) -> str:
    """
    returns the name of the variable passed to the function as a string

    Parameters
    ----------
    variables : any
        The variable(s) of interest

    Returns
    -------
    str
        A string that contains the name of variable passed to this function
        (or list of strings, if multiple arguments are passed)
    """
    pframe = inspect.currentframe().f_back  # get the previous frame that called this function
    # get the lines of code as a list of strings
    lines, first_line = inspect.getsourcelines(pframe)
    # get the line number that called this function
    lineno = pframe.f_lineno - first_line if first_line else pframe.f_lineno - 1
    # extract the argument and remove all whitespace
    arg: str = ''.join(lines[lineno].split(
        'get_variable_name(')[1].split(')')[0].split())
    if ',' in arg:
        return arg.split(',')
    else:
        return arg


def check_value(val1, val2, error_type=ValueError):
    """
    Compares two values and raises a ValueError if they are not equal.

    This method checks whether the provided values are equal. For primitive data types
    such as strings, integers, floats, lists, tuples, dictionaries, and sets, it uses
    the equality operator. For other types, it uses identity comparison.

    Parameters
    ----------
    val1 : any
        The first value to be compared.
    val2 : any
        The second value to be compared.
    error_type : Exception, optional
        The exception to raise (default is ValueError)

    Raises
    ------
    ValueError
        If the values are not equal (or not the same object for non-primitive types).
    """
    if isinstance(val1, (str, int, float, list, tuple, dict, set, np.ndarray, type({}.keys()))):
        if val1 != val2:
            raise error_type(f"{val1} is not equal to {val2}")
    else:
        if val1 is not val2:
            raise error_type(f"{val1} is not {val2}")


def check_contains(expected_values, actual_values, error_string="{var} not in {actual_values}", error_type=RuntimeError):
    """
    Checks that all of the expected_values exist in actual_values
    (It does not check for missing values)

    Parameters
    ----------
    expected_values : any iterable
        This can also be a single value, in which case it will be wrapped into a list
    actual_values : any iterable
    error_string : str, optional
        The string to display as the error message,
        kwarg substitutions will be made using .format() for "var" and "actual_values"
    error_type : Exception, optional
        The exception to raise (default is RuntimeError)

    Raises
    ------
    RuntimeError
        If a value in expected_values is not present in actual_values
    """
    # if a single expected item is provided, wrap it
    if not hasattr(expected_values, '__class_getitem__'):
        expected_values = [expected_values]
    for var in expected_values:
        if var not in actual_values:
            raise error_type(error_string.format(var=var, actual_values=actual_values))


def check_args(func, expected_args: tuple[list, dict, str], args_to_ignore: tuple[list, tuple] = ['self'], exact=True, error_type=ValueError):
    """
    Checks that the expected arguments are valid for a given function.

    This method verifies that the provided `expected_args` match the actual arguments
    of the given function `func`. If `exact` is True, the method checks for an exact
    match. If `exact` is False, it only checks that the provided `expected_args` are
    included in the actual arguments (it won't fail if the function has additional arguments).

    Parameters
    ----------
    func : function
        The function whose arguments are being checked.
    expected_args : list, dict, or str
        The expected arguments. If a dict, the values will be compared to the default values.
        If a string, it will be treated as a single argument of interest. (exact will be set to False)
    args_to_ignore : list or tuple, optional
        Arguments to ignore during the check (default is ['self']).
    exact : bool, optional
        Whether to check for an exact match of arguments (default is True).
    error_type : Exception, optional
        The exception to raise (default is ValueError)

    Raises
    ------
    ValueError
        If the expected arguments do not match the actual arguments of the function.
    """
    if isinstance(expected_args, str):
        expected_args = [expected_args]
        exact = False
    params = inspect.signature(func).parameters
    available_args = {
        arg: params[arg].default for arg in params if arg not in args_to_ignore}
    if exact:
        if isinstance(expected_args, dict):
            check_value(available_args, expected_args)
        else:
            check_value(sorted(available_args), sorted(expected_args))
    else:
        for arg in expected_args:
            if arg not in available_args:
                raise error_type(f'{arg} is not a valid argument for {func.__name__}')
            elif isinstance(expected_args, dict) and expected_args[arg] != available_args[arg]:
                raise error_type(
                    f"the default value of {arg} is {available_args[arg]}, not {expected_args[arg]}")


def run_command_no_file_error(command: str):
    """
    Executes a CLI command and handles FileNotFoundError separately.

    This method runs a given command in a temporary directory and captures the output.
    If the command returns a non-zero exit code, it checks the error message. If the
    error is a FileNotFoundError, it prints the error name. For other errors, it prints
    the full error message.

    Parameters
    ----------
    command : str
        The CLI command to be executed.

    Raises
    ------
    CalledProcessError
        If the command returns a non-zero exit code (except for FileNotFoundError).
    """
    with tempfile.TemporaryDirectory() as tempdir:
        rc = subprocess.run(command.split(), cwd=tempdir, capture_output=True, text=True)
        if rc.returncode:
            err = rc.stderr.split('\n')[-2].split(':')[0]
            if err == 'FileNotFoundError':
                print(err)
            else:
                print(rc.stderr)
                rc.check_returncode()


def get_attribute_name(object: object, attribute, error_type=AttributeError) -> str:
    """
    Gets the name of an object's attribute based on it's value

    This is intended for use with Enums and other objects that have unique values.
    This method will return the name of the first attribute that has a value that
    matches the value provided.

    Parameters
    ----------
    object : any
        The object whose attributes will be searched
    attribute : any
        The value of interest
    error_type : Exception, optional
        The exception to raise (default is AttributeError)

    Returns
    -------
    name : str
        The name of the attribute

    Raises
    ------
    AttributeError
        If the object has no attributes with the provided value.
    """
    for name, val in object.__dict__.items():
        if val == attribute:
            return name

    raise error_type(
        f"`{object.__name__}` object has no attribute with a value of `{attribute}`")


def get_all_keys(dict_of_dicts: dict, track_layers=False, all_keys=None) -> list:
    """
    Recursively get all of the keys from a dict of dicts
    Note: this will not add duplicates of keys, but will
    continue deeper even if a key is duplicated

    Parameters
    ----------
    dict_of_dicts : dict
        The dictionary who's keys will be gathered
    track_layers : Bool
        Whether or not to track where keys inside the dict of dicts
        came from. This will get every key, by ensuring that all keys
        have a unique name by tracking the path it took to get there.
    all_keys : list
        A list of the keys that have been found so far

    Returns
    -------
    all_keys : list
        A list of all the keys in the dict_of_dicts
    """
    if all_keys is None:
        all_keys = []
    for key, val in dict_of_dicts.items():
        if track_layers is True:
            current_layer = ''
        elif track_layers:
            current_layer = track_layers
        if track_layers and current_layer:
            key = current_layer+'.'+key
        if key not in all_keys:
            all_keys.append(key)
        if isinstance(val, dict):
            if track_layers:
                current_layer = key
            else:
                current_layer = False
            all_keys = get_all_keys(val, track_layers=current_layer, all_keys=all_keys)
    return all_keys


def get_value(dict_of_dicts: dict, comlpete_key: str):
    """
    Recursively get a value from a dict of dicts

    Parameters
    ----------
    dict_of_dicts : dict
    complete_key : str
        A string that contains the full path through the dict_of_dicts
        (i.e. dictkey1.dictkey2.keyofinterest)

    Returns
    -------
    val : any
        The value found
    """

    for key in comlpete_key.split('.'):
        dict_of_dicts = dict_of_dicts[key]
    return dict_of_dicts


def glue_variable(name: str, val=None, md_code=False, display=True):
    """
    Glue a variable for later use in markdown cells of notebooks

    Note:
    glue_variable(get_variable_name(Aircraft.APU.MASS))
    can be used to glue the name of the variable (Aircraft.APU.MASS)
    not the value of the variable ('aircraft:apu:mass')

    Parameters
    ----------
    name : str
        The name the value will be glued to
    val : any
        The value to be displayed in the markdown cell (default is the value of name)
    md_code : Bool
        Whether to wrap the value in markdown code formatting (e.g. `code`)
    """
    # local import so myst isn't required unless glue is being used
    from myst_nb import glue
    from IPython.display import Markdown
    if val is None:
        val = name
    if md_code:
        val = Markdown('`'+val+'`')
    else:
        val = Markdown(val)
    glue(name, val, display)


def glue_keys(dict_of_dicts: dict, display=True) -> list:
    """
    Recursively glue all of the keys from a dict of dicts

    Parameters
    ----------
    dict_of_dicts : dict
        The dictionary who's keys will be glued

    Returns
    -------
    all_keys : list
        A list of all the keys that were glued
    """
    all_keys = get_all_keys(dict_of_dicts)
    for key in all_keys:
        glue_variable(key, md_code=True, display=display)
    return all_keys
