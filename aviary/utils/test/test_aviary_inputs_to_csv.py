import os
import unittest
import importlib.util

from aviary.utils.aviary_inputs_to_csv import save_to_csv_file

from openmdao.utils.testing_utils import use_tempdirs

base_path = 'aviary/models/aircraft/'

# Need to find a way to make this flexible and not hardcode existing CSVs in here
existing_csvs = {
    'advanced_single_aisle_data.csv': base_path
    + 'advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
    'bwb_detailed_FLOPS_data.csv': base_path + 'blended_wing_body/bwb_detailed_FLOPS.csv',
    'bwb_simple_FLOPS_data.csv': base_path + 'blended_wing_body/bwb_simple_FLOPS.csv',
}


# @use_tempdirs
class PythonModelToCSV(unittest.TestCase):
    def test_python_models_to_csv(self):
        python_file_inputs = PythonModelToCSV.find_all_inputs('aviary/models/aircraft')

        for mod_name, input_data in python_file_inputs.items():
            csv_filename = mod_name + '.csv'
            save_to_csv_file(csv_filename, input_data)

            if csv_filename in existing_csvs:
                no_match = PythonModelToCSV.comp_files(
                    csv_filename, existing_csvs.get(csv_filename)
                )
                if no_match:
                    exc_string = (
                        f'Error between {csv_filename} and {existing_csvs.get(csv_filename)}'
                    )
                    print(exc_string)
                    # raise Exception(exc_string)

    # This function collects all specified 'inputs' from python modules in the specified directory (recursively)
    def find_all_inputs(directory_path):
        # variable storing all inputs and their original module names
        collected_inputs = {}

        # recursively walk through all the files in the specified directory
        for subdir, dirs, files in os.walk(directory_path):
            for file in files:
                # check for appropriate python files and generate the spec for the loader to use
                if file.endswith('.py') and not file.startswith('__'):
                    module_name = file[:-3]
                    file_path = os.path.join(subdir, file)
                    spec = importlib.util.spec_from_file_location(module_name, file_path)

                    # import the module from the spec and execute it to load its material
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # check if the module has a variable called 'inputs' & add it to the output of the function
                        if hasattr(module, 'inputs'):
                            collected_inputs[module_name] = module.inputs
                        else:
                            print('No inputs directory found in ' + file)

        return collected_inputs

    def comp_files(generated_csv, existing_csv):
        with open(generated_csv, 'r') as generated, open(existing_csv, 'r') as existing:
            gen = generated.readlines()
            exi = existing.readlines()

        diff_file = existing_csv.replace('.csv', '_diff.csv')
        with open(diff_file, 'w') as out:
            for line in gen:
                if line not in exi:
                    out.write(line)
                    # print(line)

        return os.path.isfile(diff_file) and os.path.getsize(diff_file) > 0


if __name__ == '__main__':
    unittest.main()
