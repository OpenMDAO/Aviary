import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.functions import get_path
from aviary.utils.propeller_map_conversion import PropMapType, convert_propeller_map


@use_tempdirs
class TestPropellerMapConversion(unittest.TestCase):
    """Test GASP propeller data file conversion utility by comparing against already converted data files."""

    def prepare_and_run(self, filename, output_file=None, data_format=PropMapType.GASP):
        # Specify the input file
        input_file = filepath = get_path('models/engines/propellers/' + filename)

        # Specify the output file
        if not output_file:
            filename = filepath.stem + '.csv'
            output_file = Path.cwd() / Path('TEST_' + filename)
        else:
            output_file = str(Path(output_file))

        # Specify the legacy code and propeller map
        data_format = data_format

        # Execute the conversion
        convert_propeller_map(input_file, output_file, round_data=True)

    def compare_files(self, filepath, skip_list=[]):
        """
        Compares the converted file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want
        to skip. This is useful for skipping data that Aviary might need but
        Fortran-based tools do not.
        """
        filename = filepath.split('.')[0] + '.csv'

        validation_data = get_path('models/engines/propellers/' + filename)

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
                    exc_string = (
                        f'Error: {filename}\nFound: {line_no_whitespace}\nExpected: {expected_line}'
                    )
                    raise Exception(exc_string)

    def test_propfan_conversion(self):
        filename = 'PropFan.map'
        self.prepare_and_run(filename, data_format=PropMapType.GASP)
        self.compare_files(filename, skip_list=['# created'])

    def test_GA_conversion(self):
        filename = 'general_aviation.map'
        self.prepare_and_run(filename, data_format=PropMapType.GASP)
        self.compare_files(filename, skip_list=['# created'])


if __name__ == '__main__':
    unittest.main()
    # test = TestPropellerMapConversion()
    # test.test_PM_conversion()
