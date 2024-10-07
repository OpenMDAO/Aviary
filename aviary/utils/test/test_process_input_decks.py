import unittest
from openmdao.utils.testing_utils import use_tempdirs

from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.functions import get_path


@use_tempdirs
class TestCreateVehicle(unittest.TestCase):
    """
    Test creation and modification of aircraft from CSV file.
    """

    def test_load_aircraft_csv(self):
        # Test loading a standard aircraft CSV file.
        file_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'
        aircraft_values, initialization_guesses = create_vehicle(get_path(file_path))
        self.assertIsNotNone(aircraft_values)
        self.assertIsNotNone(initialization_guesses)

    def test_load_modified_aircraft_csv(self):
        # Test loading a modified aircraft CSV file with an additional blank line.
        original_file_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'
        modified_file_path = 'modified_aircraft.csv'

        # Copy and modify the original CSV
        with open(get_path(original_file_path), 'r') as original_file, open(modified_file_path, 'w') as modified_file:
            content = original_file.readlines()
            half_way_point = len(content) // 2

            # Write first half of the file
            modified_file.writelines(content[:half_way_point])

            # Insert a few blank lines
            modified_file.write(',,,,\n')

            # Write second half of the file
            modified_file.writelines(content[half_way_point:])

        # Test create_vehicle with the modified file
        aircraft_values, initialization_guesses = create_vehicle(modified_file_path)
        self.assertIsNotNone(aircraft_values)
        self.assertIsNotNone(initialization_guesses)


if __name__ == '__main__':
    unittest.main()
