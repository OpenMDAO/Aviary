import os
import shutil
import unittest
from pathlib import Path

from aviary.api import top_dir
from aviary.utils.functions import convert_strings_to_data, get_path


class TestGetPath(unittest.TestCase):
    """Test get_path function from string and Path object for absolute and relative path."""

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
        self.assertEqual(result, Path(self.relative_test_file))  # Comparing relative paths

    def test_path_from_path_object_absolute(self):
        result = get_path(self.absolute_test_file)
        self.assertEqual(result, self.absolute_test_file)

    def test_path_from_path_object_relative(self):
        result = get_path(Path(self.relative_test_file))
        self.assertEqual(result, Path(self.relative_test_file))  # Comparing relative paths

    def test_non_existent_path(self):
        with self.assertRaises(FileNotFoundError):
            get_path('nonexistentfile.txt')


class TestTopDir(unittest.TestCase):
    def test_top_dir(self):
        result = Path(__file__).parent.parent.parent
        self.assertEqual(result, top_dir)


class TestConvertStrings2Data(unittest.TestCase):
    def is_list_of_given_type(self, lst, givenType):
        """Test if the list of data is of given type."""
        return isinstance(lst, list) and all(isinstance(i, givenType) for i in lst)

    def test_read_float(self):
        data_list = ['1.0']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [1.0])
        self.assertTrue(self.is_list_of_given_type(var_values, float))

        data_list = ['100_000.0']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [100000.0])
        self.assertTrue(self.is_list_of_given_type(var_values, float))

        data_list = ['1.0', '0.285']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [1.0, 0.285])
        self.assertTrue(self.is_list_of_given_type(var_values, float))

        data_list = ['180']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [180])
        self.assertTrue(self.is_list_of_given_type(var_values, int))

        data_list = ['0', '180']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [0, 180])
        self.assertTrue(self.is_list_of_given_type(var_values, int))

        data_list = ['False']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [False])
        self.assertTrue(self.is_list_of_given_type(var_values, bool))

        data_list = ['True']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [True])
        self.assertTrue(self.is_list_of_given_type(var_values, bool))

        data_list = ['false', 'true']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, [False, True])
        self.assertTrue(self.is_list_of_given_type(var_values, bool))

        data_list = ['this_is_a_string']
        var_values = convert_strings_to_data(data_list)
        self.assertEqual(var_values, ['this_is_a_string'])
        self.assertTrue(self.is_list_of_given_type(var_values, str))


if __name__ == '__main__':
    unittest.main()
