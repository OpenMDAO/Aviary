import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.fortran_to_aviary import fortran_to_aviary
from aviary.utils.functions import get_path
from aviary.variable_info.enums import LegacyCode


@use_tempdirs
class TestFortranToAviary(unittest.TestCase):
    """Test fortran_to_aviary legacy code input file conversion utility by comparing against already converted input files."""

    def prepare_and_run(self, filepath, out_file=None, legacy_code=LegacyCode.GASP):
        # Specify the output file
        filename = filepath.split('.')[0] + '.csv'
        if not out_file:
            out_file = Path.cwd() / Path('TEST_' + filename)
        else:
            out_file = Path(out_file)
        legacy_code = legacy_code

        # Execute the conversion
        fortran_to_aviary(filepath, legacy_code, out_file, force=True, verbosity=0)

    def compare_files(self, filepath, skip_list=['# created ']):
        """
        Compares the converted file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want
        to skip. This is useful for skipping data that Aviary might need but
        Fortran-based tools do not.
        """
        filename = filepath.split('.')[0] + '.csv'

        validation_data = get_path(filename)

        # Open the converted and validation files
        with open('TEST_' + filename, 'r') as f_in, open(validation_data, 'r') as expected:
            for line in f_in:
                if any(s in line for s in skip_list):
                    # expected.readline()
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

    def test_large_single_aisle(self):
        filepath = 'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.dat'
        comparison_filepath = 'utils/test/data/converter_test_large_single_aisle_1_GASP.csv'

        self.prepare_and_run(
            filepath,
            out_file=Path.cwd() / Path('TEST_' + comparison_filepath),
        )
        self.compare_files(comparison_filepath)

    def test_small_single_aisle(self):
        filepath = 'models/aircraft/small_single_aisle/small_single_aisle_GASP.dat'
        comparison_filepath = 'utils/test/data/converter_test_small_single_aisle_GASP.csv'

        self.prepare_and_run(
            filepath,
            out_file=Path.cwd() / Path('TEST_' + comparison_filepath),
        )
        self.compare_files(comparison_filepath)

    def test_diff_configuration(self):
        filepath = 'utils/test/data/configuration_test_data_GASP.dat'
        comparison_filepath = 'utils/test/data/converter_test_configuration_GASP.csv'

        self.prepare_and_run(filepath, out_file=Path.cwd() / Path('TEST_' + comparison_filepath))
        self.compare_files(comparison_filepath)

    def test_bwb_gasp(self):
        filepath = 'models/aircraft/blended_wing_body/generic_BWB_GASP.dat'
        comparison_filepath = 'utils/test/data/converter_test_BWB_GASP.csv'

        self.prepare_and_run(
            filepath,
            out_file=Path.cwd() / Path('TEST_' + comparison_filepath),
        )
        self.compare_files(comparison_filepath)

    def test_advanced_single_aisle(self):
        # Note: The csv comparison file N3CC_generic_low_speed_polars_FLOPSinp.csv was generated using the fortran-to-Aviary converter
        # and was not evaluated for comparison to the original. Thus, until this file is evaluated, this test is purely a regression
        # test.

        filepath = 'models/aircraft/advanced_single_aisle/N3CC_generic_low_speed_polars_FLOPS.txt'
        comparison_filepath = 'utils/test/data/converter_test_N3CC_FLOPS.csv'
        self.prepare_and_run(
            filepath,
            out_file=Path.cwd() / Path('TEST_' + comparison_filepath),
            legacy_code=LegacyCode.FLOPS,
        )
        self.compare_files(comparison_filepath)


if __name__ == '__main__':
    unittest.main()

    # test = TestFortranToAviary()
    # test.test_small_single_aisle()
