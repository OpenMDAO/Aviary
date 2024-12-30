import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.engine_deck_conversion import EngineDeckType, _exec_EDC
from aviary.utils.functions import get_path


class DummyArgs(object):
    def __init__(self):
        self.input_file = None
        self.output_file = None
        self.data_format = None


@use_tempdirs
class TestEngineDeckConversion(unittest.TestCase):
    """
    Test engine deck conversion utility by comparing against previously converted engine deck files
    """

    def prepare_and_run(self, filename, output_file=None, data_format=EngineDeckType.GASP):
        args = DummyArgs()

        # Specify the input file
        args.input_file = filepath = get_path('models/engines/'+filename)

        # Specify the output file
        if not output_file:
            filename = filepath.stem+'.deck'
            args.output_file = Path.cwd() / Path('TEST_'+filename)
        else:
            args.output_file = str(Path(output_file))

        # Specify the legacy code and engine type
        args.data_format = data_format

        # Execute the conversion
        _exec_EDC(args, None)
        return args

    def compare_files(self, filepath, skip_list=[]):
        """
        Compares the converted file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want
        to skip. This is useful for skipping data that Aviary might need but
        Fortran-based tools do not.
        """
        filename = filepath.split('.')[0]+'.deck'

        validation_data = get_path('models/engines/'+filename)

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

                except Exception as error:
                    exc_string = (
                        f'Error: {filename}\n'
                        f'Found: {line_no_whitespace}\n'
                        f'Expected: {expected_line}'
                    )
                    raise Exception(exc_string)

    # TODO currently untested!!
    # def test_TF_conversion(self):
    #     return

    def test_TP_conversion(self):
        filename = 'turboshaft_4465hp.eng'

        args = self.prepare_and_run(filename, data_format=EngineDeckType.GASP_TS)
        self.compare_files(filename, skip_list=['# created'])


if __name__ == "__main__":
    unittest.main()
