import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.flops_based.characteristic_lengths import CharacteristicLengths
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft


class CharacteristicLengthsTest(unittest.TestCase):
    """Test characteristic length and fineness ratio calculations"""

    def setUp(self):
        self.prob = om.Problem()

    def test_case_multiengine(self):
        # test with multiple engine types
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 2, 3]))
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 7)

        prob.model.add_subsystem(
            'char_lengths',
            CharacteristicLengths(aviary_options=aviary_options),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        input_list = [
            (Aircraft.Canard.AREA, 'ft**2'),
            (Aircraft.Canard.ASPECT_RATIO, 'unitless'),
            (Aircraft.Canard.THICKNESS_TO_CHORD, 'unitless'),
            # (Aircraft.Fuselage.AVG_DIAMETER, 'ft'),
            (Aircraft.Fuselage.LENGTH, 'ft'),
            (Aircraft.HorizontalTail.AREA, 'ft**2'),
            (Aircraft.HorizontalTail.ASPECT_RATIO, 'unitless'),
            (Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 'unitless'),
            (Aircraft.VerticalTail.AREA, 'ft**2'),
            (Aircraft.VerticalTail.ASPECT_RATIO, 'unitless'),
            (Aircraft.VerticalTail.THICKNESS_TO_CHORD, 'unitless'),
            (Aircraft.Wing.AREA, 'ft**2'),
            (Aircraft.Wing.ASPECT_RATIO, 'unitless'),
            (Aircraft.Wing.GLOVE_AND_BAT, 'ft**2'),
            (Aircraft.Wing.TAPER_RATIO, 'unitless'),
            (Aircraft.Wing.THICKNESS_TO_CHORD, 'unitless')
        ]
        for var, units in input_list:
            prob.set_val(var, aviary_options.get_val(var, units))

        # this is another component's output
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, val=12.75)

        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, val=np.array([6, 4.25, 9.6]))
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH, val=np.array([8.4, 5.75, 10]))

        prob.run_model()

        length = prob.get_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH)
        fineness = prob.get_val(Aircraft.Nacelle.FINENESS)

        expected_length = np.array([8.4, 5.75, 10.])
        expected_fineness = np.array([1.4, 1.352941176470, 1.041666666667])

        assert_near_equal(length, expected_length, tolerance=1e-10)
        assert_near_equal(fineness, expected_fineness, tolerance=1e-10)

        # getting nan for undefined partials?
        # don't see nan anymore.
        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
    # test = CharacteristicLengthsTest()
    # test.setUp()
    # test.test_case_multiengine()
