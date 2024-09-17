import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.examples.variable_meta_data_extension import ExtendedMetaData
from aviary.examples.variables_extension import Aircraft
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_value
from aviary.utils.functions import get_path


class MetaDataExtensionTest(unittest.TestCase):
    """
    Test set_value with units for extended MetaData.
    """

    def test_set_value_metadata_extension(self):
        option_defaults = AviaryValues()

        filename = get_path(
            'models/engines/turbofan_23k_1.deck')
        option_defaults.set_val(Aircraft.Engine.DATA_FILE, filename)
        option_defaults.set_val(Aircraft.Wing.AERO_CENTER, val=5, units='ft',
                                meta_data=ExtendedMetaData)

        option_defaults = set_value(Aircraft.Wing.AERO_CENTER, [
                                    1.0], option_defaults, 'ft', ExtendedMetaData)

        check_val = option_defaults.get_val(Aircraft.Wing.AERO_CENTER, units='inch')

        assert_near_equal(check_val, 12, 1e-9)


if __name__ == '__main__':
    unittest.main()
