import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.functions import get_path
from aviary.utils.propeller_map_conversion import PropMapType, _exec_PMC


class DummyArgs(object):
    def __init__(self):
        self.input_file = None
        self.output_file = None
        self.data_format = None


@use_tempdirs
class TestPropellerMapConversion(unittest.TestCase):
    """
    Test GASP propeller data file conversion utility by comparing against already converted data files.
    """

    def prepare_and_run(self, filename, output_file=None, data_format=PropMapType.GASP):
        args = DummyArgs()

        # Specify the input file
        args.input_file = filepath = get_path('models/propellers/'+filename)

        # Specify the output file
        if not output_file:
            filename = filepath.stem+'.prop'
            args.output_file = Path.cwd() / Path('TEST_'+filename)
        else:
            args.output_file = str(Path(output_file))

        # Specify the legacy code and propeller map
        args.data_format = data_format

        # Execute the conversion
        _exec_PMC(args, None)

    def compare_files(self, filepath, skip_list=[]):
        """
        Compares the converted file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want
        to skip. This is useful for skipping data that Aviary might need but
        Fortran-based tools do not.
        """
        filename = filepath.split('.')[0]+'.prop'

        validation_data = get_path('models/propellers/'+filename)

        # Open the converted and validation files
        with open('TEST_'+filename, 'r') as f_in, open(validation_data, 'r') as expected:
            for line in f_in:
                if any(s in line for s in skip_list):
                    break
                # Remove whitespace and compare
                expected_line = ''.join(expected.readline().split())
                line_no_whitespace = ''.join(line.split())

                # Assert that the lines are equal
                try:
                    self.assertEqual(line_no_whitespace.count(expected_line), 1)

                except:
                    exc_string = f'Error:  {filename}\nFound: {line_no_whitespace}\nExpected:  {expected_line}'
                    raise Exception(exc_string)

    def test_PM_conversion(self):
        filename = 'PropFan.map'
        self.prepare_and_run(filename, data_format=PropMapType.GASP)
        self.compare_files(filename, skip_list=['# created'])


if __name__ == "__main__":
    unittest.main()
