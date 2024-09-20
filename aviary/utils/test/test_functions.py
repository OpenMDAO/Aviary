import unittest
import os
import shutil
from pathlib import Path

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission
from aviary.utils.functions import add_opts2vals, create_opts2vals, get_path


class TestOpts2Vals(unittest.TestCase):
    """
    Test the functionality of create_opts2vals function.
    """

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS,
                             val=180, units='unitless')
        self.options.set_val(Mission.Design.CRUISE_ALTITUDE, val=35, units='kft')

    def test_default_units(self):
        tol = 1e-4
        self.prob = om.Problem()
        OptionsToValues = create_opts2vals(
            [Aircraft.CrewPayload.NUM_PASSENGERS,
             Mission.Design.CRUISE_ALTITUDE,])
        add_opts2vals(self.prob.model, OptionsToValues, self.options)
        self.prob.setup()
        self.prob.run_model()

        assert_near_equal(
            self.prob['option:'+Aircraft.CrewPayload.NUM_PASSENGERS], 180, tol)
        assert_near_equal(self.prob['option:'+Mission.Design.CRUISE_ALTITUDE], 35, tol)

    def test_specified_units(self):
        tol = 1e-4
        self.prob = om.Problem()
        OptionsToValues = create_opts2vals(
            [Aircraft.CrewPayload.NUM_PASSENGERS,
             Mission.Design.CRUISE_ALTITUDE,],
            output_units={Mission.Design.CRUISE_ALTITUDE: 'm'})
        add_opts2vals(self.prob.model, OptionsToValues, self.options)
        self.prob.setup()
        self.prob.run_model()

        altitude_in_meters = om.convert_units(35000, 'ft', 'm')
        assert_near_equal(
            self.prob['option:'+Aircraft.CrewPayload.NUM_PASSENGERS], 180, tol)
        assert_near_equal(
            self.prob['option:'+Mission.Design.CRUISE_ALTITUDE], altitude_in_meters, tol)


class TestGetPath(unittest.TestCase):
    """
    Test get_path function from string and Path object for absolute and relative path
    """

    def setUp(self):
        self.current_dir = Path.cwd()
        process_id = os.getpid()
        # Create a test file in current directory
        self.relative_test_file = f'testfile_{process_id}.txt'
        with open(self.relative_test_file, 'w') as f:
            f.write('test')
        # Create a test file in a sub-directory
        self.sub_directory = self.current_dir / f'tmp_dir_{process_id}'
        self.sub_directory.mkdir(exist_ok=True)
        self.absolute_test_file = self.sub_directory / f'testfile2_{process_id}.txt'
        with open(self.absolute_test_file, 'w') as f:
            f.write('test')

    def tearDown(self):
        # Cleanup the created test files and directories after each test
        if Path(self.relative_test_file).exists():
            Path(self.relative_test_file).unlink()
        if Path(self.absolute_test_file).exists():
            Path(self.absolute_test_file).unlink()
        shutil.rmtree(self.sub_directory, ignore_errors=True)

    def test_path_from_string_absolute(self):
        result = get_path(str(self.absolute_test_file))
        self.assertEqual(result, self.absolute_test_file)

    def test_path_from_string_relative(self):
        result = get_path(self.relative_test_file)
        self.assertEqual(result, Path(self.relative_test_file)
                         )  # Comparing relative paths

    def test_path_from_path_object_absolute(self):
        result = get_path(self.absolute_test_file)
        self.assertEqual(result, self.absolute_test_file)

    def test_path_from_path_object_relative(self):
        result = get_path(Path(self.relative_test_file))
        self.assertEqual(result, Path(self.relative_test_file)
                         )  # Comparing relative paths

    def test_non_existent_path(self):
        with self.assertRaises(FileNotFoundError):
            get_path('nonexistentfile.txt')


if __name__ == "__main__":
    unittest.main()
