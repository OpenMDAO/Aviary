import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.flops_based.characteristic_lengths import (
    BWBWingCharacteristicLength,
    WingCharacteristicLength,
    OtherCharacteristicLengths,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft


class CharacteristicLengthsTest(unittest.TestCase):
    """Test characteristic length and fineness ratio calculations."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case_multiengine(self):
        # test with multiple engine types
        prob = self.prob

        aviary_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')

        aviary_options_wing = {
            Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION: aviary_inputs.get_val(
                Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION
            ),
        }

        prob.model.add_subsystem(
            'wing_char_length',
            WingCharacteristicLength(**aviary_options_wing),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        aviary_options_others = {
            Aircraft.Engine.NUM_ENGINES: np.array([2, 2, 3]),
        }

        prob.model.add_subsystem(
            'other_char_lengths',
            OtherCharacteristicLengths(**aviary_options_others),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
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
            (Aircraft.Wing.THICKNESS_TO_CHORD, 'unitless'),
        ]
        for var, units in input_list:
            prob.set_val(var, aviary_inputs.get_val(var, units))

        # this is another component's output
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, val=12.75)

        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, val=np.array([6, 4.25, 9.6]))
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH, val=np.array([8.4, 5.75, 10]))

        prob.run_model()

        length = prob.get_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH)
        fineness = prob.get_val(Aircraft.Nacelle.FINENESS)

        expected_length = np.array([8.4, 5.75, 10.0])
        expected_fineness = np.array([1.4, 1.352941176470, 1.041666666667])

        assert_near_equal(length, expected_length, tolerance=1e-10)
        assert_near_equal(fineness, expected_fineness, tolerance=1e-10)

        # getting nan for undefined partials?
        # don't see nan anymore.
        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBWingCharacteristicLengthsTest(unittest.TestCase):
    """Test characteristic length and fineness ratio calculations for BWB."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        prob.model.add_subsystem(
            'cl', BWBWingCharacteristicLength(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.AREA, val=8668.64638424)
        prob.set_val(Aircraft.Wing.SPAN, val=238.08)
        prob.run_model()

        out1 = prob.get_val(Aircraft.Wing.CHARACTERISTIC_LENGTH)
        exp1 = 36.410645095113139
        assert_near_equal(out1, exp1, tolerance=1e-9)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
