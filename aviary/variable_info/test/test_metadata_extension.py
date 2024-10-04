import unittest

from aviary.examples.variable_meta_data_extension import ExtendedMetaData
from aviary.examples.variables_extension import Aircraft


class MetaDataExtensionTest(unittest.TestCase):
    """
    Test the use of extended meta data
    """

    def test_metadata_extension(self):

        aircraft_variable = ExtendedMetaData[Aircraft.LandingGear.MAIN_GEAR_OLEO_DIAMETER]

        self.assertEqual(Aircraft.LandingGear.MAIN_GEAR_OLEO_DIAMETER,
                         'aircraft:landing_gear:main_gear_oleo_diameter')
        self.assertEqual(aircraft_variable['units'], 'ft')
        self.assertEqual(aircraft_variable['desc'], 'Main gear oleo diameter')
        self.assertEqual(aircraft_variable['default_value'], 0.0)
        self.assertEqual(aircraft_variable['option'], False)
        self.assertEqual(aircraft_variable['types'], None)


if __name__ == '__main__':
    unittest.main()
