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



if __name__ == '__main__':
    unittest.main()
