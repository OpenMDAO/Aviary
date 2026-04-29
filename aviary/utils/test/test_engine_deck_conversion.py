import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.engine_deck_conversion import EngineDeckType, convert_engine_deck
from aviary.utils.functions import get_path


@use_tempdirs
class TestEngineDeckConversion(unittest.TestCase):
    """Test engine deck conversion utility by comparing against previously converted engine deck files."""

    def prepare_and_run(self, filename, output_file=None, data_format=EngineDeckType.GASP):
        # Specify the input file
        input_file = get_path('utils/test/data/' + filename)

        # Specify the output file
        if not output_file:
            filename = input_file.stem + '.csv'
            output_file = Path.cwd() / Path('TEST_' + filename)
        else:
            output_file = str(Path(output_file))

        # Execute the conversion
        convert_engine_deck(input_file, output_file, data_format, True)

    def compare_files(self, filepath, skip_list=['# created']):
        """
        Compares the converted file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want to skip. This is
        useful for skipping data that Aviary might need but Fortran-based tools do not.
        """
        filename = filepath.split('.')[0] + '.csv'

        validation_data = get_path('models/engines/' + filename)

        # Open the converted and validation files
        with open('TEST_' + filename, 'r') as f_in, open(validation_data, 'r') as expected:
            for line in f_in:
                if any(s in line for s in skip_list):
                    expected.readline()
                    continue
                # Remove whitespace and compare
                expected_line = ''.join(expected.readline().split())
                line_no_whitespace = ''.join(line.split())

                # Assert that the lines are equal
                try:
                    self.assertEqual(line_no_whitespace.count(expected_line), 1)

                except Exception:
                    exc_string = f'Error: TEST_{filename}\nFound: {line_no_whitespace}\nExpected: {expected_line}'
                    raise Exception(exc_string)

    def test_TF_conversion_FLOPS(self):
        filename = 'turbofan_22k.txt'

        self.prepare_and_run(filename, data_format=EngineDeckType.FLOPS)
        self.compare_files(filename)

    def test_TF_conversion_GASP(self):
        filename = 'turbofan_23k_1.eng'

        self.prepare_and_run(filename, data_format=EngineDeckType.GASP)
        self.compare_files(filename)

    def test_TP_conversion(self):
        filename = 'turboshaft_4465hp.eng'

        self.prepare_and_run(filename, data_format=EngineDeckType.GASP_TS)
        self.compare_files(filename)

    def test_bad_filenames(self):
        with self.subTest('give_bad_filename'):
            input_file = 'fake_engine_20k.txt'
            try:
                convert_engine_deck(input_file, 'output.csv', EngineDeckType.FLOPS)
            except FileNotFoundError as e:
                self.assertEqual(
                    str(e),
                    'Engine deck file not found: fake_engine_20k.txt. Please check that the file path is correct and the file exists.',
                )
            else:
                raise AssertionError('Did not raise an error for nonexistent file.')

        with self.subTest('give_folder_path'):
            input_file = 'models/engines'
            try:
                convert_engine_deck(input_file, 'output.csv', EngineDeckType.FLOPS)
            except ValueError as e:
                self.assertEqual(
                    str(e),
                    'Path is not a file: models/engines. Please provide a path to a valid engine deck file.',
                )
            else:
                raise AssertionError('Did not raise an error for providing folder instead of file.')


if __name__ == '__main__':
    unittest.main()
    # test = TestEngineDeckConversion()
    # test.test_TP_conversion()
