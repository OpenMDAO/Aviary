import inspect
import subprocess
import tempfile
import os


def check_value(val1, val2):
    if isinstance(val1, (str, int, float, list, tuple, dict, set)):
        if val1 != val2:
            raise ValueError(f"{val1} is not equal to {val2}")
    else:
        if val1 is not val2:
            raise ValueError(f"{val1} is not {val2}")


def check_args(func, expected_args: list | dict | str, args_to_ignore: list | tuple = ['self'], exact=True):
    '''
    Checks that the expected_args are valid for func, if exact is True, this will fail if the
    expected_args do not exactly match the available args. If exact is False, this will only
    fail if an arg provided in expected_args is not included in the available_args.

    If expected_args is provided as a dict instead of a list, the values for each argument
    will be compared to the default value of the function.
    If expected_args is a string, it will be interpreted as a single argument of interest, and
    exact will be set to False.
    '''
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
    '''This will test a CLI command, but won't fail for a FileNotFoundError'''
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
