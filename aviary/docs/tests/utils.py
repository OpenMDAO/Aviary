import inspect
import subprocess
import tempfile
import os
import numpy as np


def check_value(val1, val2):
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

    Raises
    ------
    ValueError
        If the values are not equal (or not the same object for non-primitive types).
    """
    if isinstance(val1, (str, int, float, list, tuple, dict, set, np.ndarray)):
        if val1 != val2:
            raise ValueError(f"{val1} is not equal to {val2}")
    else:
        if val1 is not val2:
            raise ValueError(f"{val1} is not {val2}")


def check_contains(expected_values, actual_values, error_string="{var} not in {actual_values}", error_type=RuntimeError):
    """
    Checks that all of the expected_values exist in actual_values
    (It does not check for missing values)

    Args:
        expected_values : any iterable
            This can also be a single value, in which case it will be wrapped into a list
        actual_values : any iterable
        error_string : str
            The string to display as the error message,
            kwarg subsitutions will be made using .format() for "var" and "actual_values"
        error_type : Exception
            The exception to raise (default is RuntimeError)

    Raises:
        RuntimeError
            If a value in expected_values is not present in actual_values
    """
    # if a single expected item is provided, wrap it
    if not hasattr(expected_values, '__class_getitem__'):
        expected_values = [expected_values]
    for var in expected_values:
        if var not in actual_values:
            raise error_type(error_string.format(var=var, actual_values=actual_values))


def check_args(func, expected_args: list | dict | str, args_to_ignore: list | tuple = ['self'], exact=True):
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
                raise ValueError(f'{arg} is not a valid argument for {func.__name__}')
            elif isinstance(expected_args, dict) and expected_args[arg] != available_args[arg]:
                raise ValueError(
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
        os.chdir(tempdir)
        rc = subprocess.run(command.split(), capture_output=True, text=True)
        if rc.returncode:
            err = rc.stderr.split('\n')[-2].split(':')[0]
            if err == 'FileNotFoundError':
                print(err)
            else:
                print(rc.stderr)
                rc.check_returncode()
