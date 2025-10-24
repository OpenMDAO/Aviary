import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.flops_based.nacelle import Nacelles
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class NacelleTest(unittest.TestCase):
    """Test nacelle wetted area computation."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case_multiengine(self):
        # test with multiple engine types
        prob = self.prob

        options = {
            Aircraft.Engine.NUM_ENGINES: np.array([2, 2, 3]),
        }

        prob.model.add_subsystem(
            'nacelles', Nacelles(**options), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, val=np.array([6, 4.25, 9.6]))
        prob.set_val(Aircraft.Nacelle.AVG_LENGTH, val=np.array([8.4, 5.75, 10]))
        prob.set_val(Aircraft.Nacelle.WETTED_AREA_SCALER, val=np.array([1.0, 0.92, 1.4]))

        prob.run_model()

        wetted_area = prob.get_val(Aircraft.Nacelle.WETTED_AREA)
        total_wetted_area = prob.get_val(Aircraft.Nacelle.TOTAL_WETTED_AREA)

        expected_wetted_area = np.array([141.12, 62.951, 376.32])
        expected_total_wetted_area = sum(expected_wetted_area * np.array([2, 2, 3]))

        assert_near_equal(wetted_area, expected_wetted_area, tolerance=1e-10)
        assert_near_equal(total_wetted_area, expected_total_wetted_area, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
