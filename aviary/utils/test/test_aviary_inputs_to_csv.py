import os
import unittest
import importlib.util

from aviary.utils.aviary_inputs_to_csv import save_to_csv_file
from aviary.utils.functions import get_path

from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class PythonModelToCSV(unittest.TestCase):
    def find_all_inputs(self, directory_path):
        """
        This function collects all specified 'inputs' from python modules in the specified directory (recursively)
        The output is a dict containing the names of the modules storing 'inputs', and the values of the 'inputs'
        """
        # variable storing all inputs and their original module names
        collected_inputs = {}

        python_inputs = {
            'large_single_aisle_1_FLOPS_data.py',
            'large_single_aisle_2_FLOPS_data.py',
            'large_single_aisle_2_altwt_FLOPS_data.py',
            'large_single_aisle_2_detailwing_FLOPS_data.py',
            'multi_engine_single_aisle_data.py',
        }

        # recursively walk through all the files in the specified directory
        # for subdir, dirs, files in os.walk(directory_path):
        #    for file in files:
        # check for appropriate python files and generate the spec for the loader to use
        #        if file.endswith('.py') and not file.startswith('__'):
        #            module_name = file[:-3]
        #            file_path = os.path.join(subdir, file)

        for file in python_inputs:
            module_name = file[:-3]
            file_path = get_path(file)
            spec = importlib.util.spec_from_file_location(module_name, file_path)

            # import the module from the spec and execute it to load its material
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # check if the module has a variable called 'inputs' & add it to the output of the function
                if hasattr(module, 'inputs'):
                    collected_inputs[module_name] = module.inputs
                else:
                    # Probably can remove this to clean up terminal line spam?
                    print('No inputs directory found in ' + file)

        return collected_inputs

    def compare_files(self, filepath, validation_data, skip_list=['#']):
        """
        Compares the converted file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want
        to skip. This is useful for skipping data that Aviary might need but
        Fortran-based tools do not.
        """
        filename = filepath.split('.')[0] + '.csv'

        filename_path = get_path(filename)
        validation_data_path = get_path(validation_data)

        # Open the converted and validation files
        with open(filename, 'r') as f_in, open(validation_data_path, 'r') as expected:
            for line in f_in:
                if any(s in line for s in skip_list):
                    # expected.readline()
                    continue
                if line.startswith('\n'):
                    continue

                expected_line = ''
                # Skip any lines that are empty, have a comment indicator to start, or have members in the 'skip_list'
                while (
                    expected_line.__len__() == 0
                    or expected_line.find('#') != -1
                    or any(s in expected_line for s in skip_list)
                ):
                    expected_line = ''.join(expected.readline().split())
                line_no_whitespace = ''.join(line.split())

                # Assert that the lines are equal
                try:
                    self.assertEqual(line_no_whitespace.count(expected_line), 1)

                except Exception:
                    exc_string = (
                        f'Error: {filename}\nFound: {line_no_whitespace}\nExpected: {expected_line}'
                    )
                    raise Exception(exc_string)

    def test_python_models_to_csv(self):
        """
        This is the test portion of the class that provides function arguments to 'find_all_inputs' and 'compare_functions'.
        The program checks for all 'inputs' in aviary/models/aircraft, then compares them to the specified CSVs below.
        """

        base_path = 'aviary/models/aircraft/'

        # Need to find a way to make this flexible and not hardcode existing CSVs in here
        existing_csvs = {
            'large_single_aisle_1_FLOPS_data_test.csv': base_path
            + 'large_single_aisle_1/large_single_aisle_1_FLOPS_data.csv',
            'large_single_aisle_2_FLOPS_data_test.csv': base_path
            + 'large_single_aisle_2/large_single_aisle_2_FLOPS_data.csv',
            'large_single_aisle_2_altwt_FLOPS_data_test.csv': base_path
            + 'large_single_aisle_2/large_single_aisle_2_altwt_FLOPS_data.csv',
            'large_single_aisle_2_detailwing_FLOPS_data_test.csv': base_path
            + 'large_single_aisle_2/large_single_aisle_2_detailwing_FLOPS_data.csv',
            'multi_engine_single_aisle_data_test.csv': base_path
            + 'multi_engine_single_aisle/multi_engine_single_aisle_data.csv',
        }

        # Finding all 'inputs' through this function call
        python_file_inputs = self.find_all_inputs('aviary/models/aircraft')
        # Specifying the string tags to ignore in the line-by-line comparison of the CSVs
        excluded_lines = ['#', 'engine:data_file']

        for mod_name, input_data in python_file_inputs.items():
            csv_filename = mod_name + '_test.csv'
            # Function call to generate CSVs from the AviaryInputs
            save_to_csv_file(csv_filename, input_data)

            if csv_filename in existing_csvs:
                self.compare_files(csv_filename, existing_csvs.get(csv_filename), excluded_lines)


if __name__ == '__main__':
    unittest.main()
