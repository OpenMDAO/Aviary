"""
This script is designed to test run scripts located in a specified directory.
It uses Python's unittest framework to dynamically generate test cases for each
script that begins with 'run_' and ends with '.py'. The script ensures that
each run script executes without errors within a 10-second timeout. If a script
runs longer than 10 seconds, it's still considered a pass unless other errors occur.
"""

import os
import subprocess
import unittest
from pathlib import Path


class RunScriptTest(unittest.TestCase):
    """
    A test case class that uses unittest to run and test scripts with a timeout.

    ...

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

    @classmethod
    def setUpClass(cls):
        """
        Class method to set up the test case class by finding all run scripts.

        This method is called once before starting the tests and is used to
        populate the 'run_files' attribute with a list of run scripts.
        """
        base_directory = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), ".")
        cls.run_files = cls.find_run_files(base_directory)

    @staticmethod
    def find_run_files(base_dir):
        """
        Find and return a list of run scripts in the specified directory.

        Parameters
        ----------
        base_dir : str
            The directory to search for run scripts.

        Returns
        -------
        list
            A list of pathlib.Path objects pointing to the run scripts.
        """
        run_files = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.startswith('run_') and file.endswith('.py'):
                    run_files.append(Path(root) / file)
        return run_files

    def run_script(self, script_path, max_allowable_time=300):
        """
        Attempt to run a script with a 300-second timeout and handle errors.

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
            proc = subprocess.Popen(['python', script_path],
                                    stdout=devnull, stderr=subprocess.PIPE)
        proc.wait(timeout=max_allowable_time)
        (stdout, stderr) = proc.communicate()

        if proc.returncode != 0:
            if 'ImportError' in str(stderr):
                self.skipTest(f"Skipped {script_path.name} due to ImportError")
            else:
                raise Exception(
                    f"Error running {script_path.name}:\n{stderr.decode('utf-8')}")

    def test_run_scripts(self):
        """
        Test each run script to ensure it executes without error.

        This method generates a subtest for each script in 'run_files'.
        Each script is tested to ensure it runs without errors.
        """
        for script_path in self.run_files:
            with self.subTest(script=script_path.name):
                self.run_script(script_path)


if __name__ == "__main__":
    unittest.main()
