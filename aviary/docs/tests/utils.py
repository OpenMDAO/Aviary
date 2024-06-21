import inspect
import subprocess
import tempfile
import os


def check_value(val1, val2):
    if isinstance(val1, (str, int, float, list, tuple, dict)):
        if val1 != val2:
            raise ValueError(f"{val1} is not equal to {val2}")
    else:
        if val1 is not val2:
            raise ValueError(f"{val1} is not {val2}")


def check_args(func, expected_args: list, args_to_ignore=['self'], exact=True):
    '''
    Checks that the expected_args are valid for func, if exact is True, this will fail if the
    expected_args do not exactly match the available args. If exact is False, this will only
    fail if an arg provided in expected args is not included in the available_args
    '''
    available_args = [arg for arg in inspect.getfullargspec(
        func)[0] if arg not in args_to_ignore]
    if exact:
        available_args.sort()
        expected_args.sort()
        check_value(available_args, expected_args)
    else:
        for arg in expected_args:
            if arg not in available_args:
                raise ValueError(f'{arg} is not a valid argument for {func.__name__}')


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
