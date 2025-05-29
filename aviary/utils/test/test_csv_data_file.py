import os
import unittest
import warnings

from openmdao.utils.assert_utils import assert_near_equal, assert_warning
from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.csv_data_file import read_data_file, write_data_file
from aviary.utils.functions import get_path
from aviary.utils.named_values import NamedValues, get_items, get_keys
from aviary.utils.process_input_decks import parse_inputs
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variable_meta_data import CoreMetaData, add_meta_data


@use_tempdirs
class TestAviaryCSV(unittest.TestCase):
    """Test read/write CSV files and compare CSV files."""

    def setUp(self):
        self.filename = get_path('utils/test/data/csv_test.csv')
        # what will get written to the csv
        self.data = NamedValues(
            {
                'aircraft:wing:span': ([15.24, 118, 90, 171], 'ft'),
                'aircraft:crew_and_payload:num_passengers': ([125, 28, 0.355, 44], 'unitless'),
                'fake_var': ([0.932, 1023.54, 0, -13], 'lbm'),
            }
        )
        self.comments = ['comment 1', 'comment 2# more comment 2', 'inline comment']

        # what the csv should look like after writing
        self.expected_contents = [
            '# comment 1',
            '# comment 2# more comment 2',
            '# inline comment',
            '',
            'aircraft:wing:span (ft), aircraft:crew_and_payload:num_passengers, fake_var (lbm)',
            '                  15.24,                                      125,          0.932',
            '                    118,                                       28,        1023.54',
            '                     90,                                    0.355,              0',
            '                    171,                                       44,            -13',
        ]

    def test_write_data_file(self):
        write_data_file('write.csv', self.data, self.comments)
        read_contents = []
        with open('write.csv') as file:
            for line in file:
                read_contents.append(line.strip('\n'))
        if read_contents != self.expected_contents:
            raise ValueError('Contents written to csv do not match expected values')

    def test_read_data_file(self):
        self._compare_csv_results(*read_data_file(self.filename, save_comments=True))

    def test_read_metadata_csv(self):
        # catch warnings as errors
        warnings.filterwarnings('error')

        try:
            data, comments = read_data_file(self.filename, CoreMetaData)
        except UserWarning:
            # disable warnings as errors behavior for future tests
            warnings.resetwarnings()
            # run read_data_file() without catching warnings as errors so it completes
            data, comments = read_data_file(self.filename, CoreMetaData, save_comments=True)
            if 'fake_var' in get_keys(data):
                raise RuntimeError('fake_var should be skipped when reading csv')
            self._compare_csv_results(data, comments)
        else:
            # disable warnings as errors behavior for future tests
            warnings.resetwarnings()
            raise RuntimeError('File should raise warning of skipped header data')

        # add fake_vars to metadata and try again - should work identically to test_read_data_file()
        add_meta_data(key='fake_var', meta_data=CoreMetaData, units='lbm')
        self._compare_csv_results(*read_data_file(self.filename, CoreMetaData, save_comments=True))

    def test_aliases_csv(self):
        aliases = {'Real Var': 'Fake Var'}
        data = read_data_file(self.filename, aliases=aliases)
        if 'fake_var' in get_keys(data):
            raise RuntimeError("'fake_var' should be converted to 'Real Var'")
        if 'Real Var' not in get_keys(data):
            raise RuntimeError("'Real Var' is not in data read from csv")

    @use_tempdirs
    def test_parse_input(self):
        aircraft_values = get_option_defaults(engine=False)
        # create a temporary csv file for testing non-existing variable name
        file_name = 'aircraft_for_invalid_var.csv'
        with open(file_name, 'w') as file:
            file.write('test_string,0\n')  # be ignored
            file.write('aircraft:wing:mass_scalar,1,unitless\n')  # raise a warning
            file.write('aircraft:anti_icing:mass,551,lbm\n')  # a good variable
        vehicle_deck = get_path(file_name)

        msg = "Variable 'aircraft:wing:mass_scalar' is not in meta_data nor in 'guess_names'. It will be ignored."
        with assert_warning(UserWarning, msg):
            parse_inputs(vehicle_deck, aircraft_values)

        # remove the temporary csv file
        os.remove(file_name)

    def _compare_csv_results(self, data, comments):
        expected_data = self.data

        if comments != self.comments:
            raise ValueError(
                f'Comments read from {self.filename.name} do not match expected values'
            )
        for item in get_items(data):
            key = item[0]
            val = item[1][0]
            units = item[1][1]
            if key not in expected_data:
                raise ValueError(f'{key} not found in {self.filename.name}')
            expected_units = expected_data.get_item(key)[1]
            assert_near_equal(val, expected_data.get_val(key, expected_units))
            if units != expected_units:
                raise ValueError(
                    f'Units {units} read from {self.filename.name} do not '
                    f'match expected units of {expected_units}'
                )


if __name__ == '__main__':
    unittest.main()
