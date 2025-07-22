"""
This script is designed to test run scripts located in a specified directory.
It uses Python's unittest framework to dynamically generate test cases for each
script that begins with 'run_' and ends with '.py'.
"""

import os
import subprocess
import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.general_utils import set_pyoptsparse_opt
from parameterized import parameterized

# TODO: Address any issue that requires a skip.
SKIP_EXAMPLES = {
    'run_multimission_example.py': 'Broken due to OpenMDAO changes',
    'run_OAS_wing_mass_example.py': 'Timeout when running via this script',
    'run_NPSS_example.py': 'Cannot be run without NPSS install',
    'run_level3_example.py': 'Currently broken, awaiting refresh',
}

# TODO: temporary fix, waiting on https://github.com/OpenMDAO/OpenMDAO/issues/3510
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')


def find_examples():
    """
    Find and return a list of run scripts in the specified directory.

    Returns
    -------
    list
        A list of pathlib.Path objects pointing to the run scripts.
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.')

    run_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith('run_') and file.endswith('.py'):
                run_files.append(Path(root) / file)
    return run_files


def example_name(testcase_func, param_num, param):
    """
    Returns a formatted case name for unit testing with decorator @parameterized.expand().
    It is intended to be used when expand() is called with a list of strings
    representing test case names.

    Parameters
    ----------
    testcase_func : Any
        This parameter is ignored.
    param_num : Any
        This parameter is ignored.
    param : param
        The param object containing the case name to be formatted.
    """
    return 'test_example_' + param.args[0].name.replace('.py', '')


@use_tempdirs
class RunScriptTest(unittest.TestCase):
    """
    A test case class that uses unittest to run and test scripts with a timeout.

    Attributes
    ----------
    base_directory : str
        The base directory where the run scripts are located.
    run_files : list
        A list of paths to run scripts found in the base directory.

    Methods
    -------
    setUpClass()
        Class method to find all run scripts before tests are run.
    find_run_files(base_dir)
        Finds and returns all run scripts in the specified directory.
    run_script(script_path)
        Attempts to run a script with a timeout and handles errors.
    test_run_scripts()
        Generates a test for each run script with a timeout.
    """

    def run_script(self, script_path, max_allowable_time=500):
        """
        Attempt to run a script with a 500-second timeout and handle errors.

        Parameters
        ----------
        script_path : pathlib.Path
            The path to the script to be run.

        Raises
        ------
        Exception
            Any exception other than ImportError or TimeoutExpired that occurs while running the script.
        """
        with open(os.devnull, 'w') as devnull:
            proc = subprocess.Popen(['python', script_path], stdout=devnull, stderr=subprocess.PIPE)
        proc.wait(timeout=max_allowable_time)
        (stdout, stderr) = proc.communicate()

        if proc.returncode != 0:
            if 'ImportError' in str(stderr):
                self.skipTest(f'Skipped {script_path.name} due to ImportError')
            else:
                raise Exception(f'Error running {script_path.name}:\n{stderr.decode("utf-8")}')

    @parameterized.expand(find_examples(), name_func=example_name)
    def test_run_scripts(self, example_path):
        """Test each run script to ensure it executes without error."""
        if example_path.name in SKIP_EXAMPLES:
            reason = SKIP_EXAMPLES[example_path.name]
            self.skipTest(f'Skipped {example_path.name}: {reason}.')

        self.run_script(example_path)


if __name__ == '__main__':
    unittest.main()
