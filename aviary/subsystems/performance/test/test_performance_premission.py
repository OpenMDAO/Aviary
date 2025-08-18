import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from aviary.variable_info.variables import Aircraft, Mission
from aviary.subsystems.performance.performance_premission import PerformancePremission


class PerformancePremissionTest(unittest.TestCase):
    """Test computation of parameters in performance premission."""

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem('perf_premission', PerformancePremission(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        # arbitrary numbers in roughly correct order of magnitude for testing
        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 32745, 'lbf')
        prob.set_val(Aircraft.Wing.AREA, 1823, 'ft**2')
        prob.set_val(Mission.Design.GROSS_MASS, 203154, 'lbm')

        prob.run_model()

        TW = prob.get_val(Aircraft.Design.THRUST_TO_WEIGHT_RATIO)
        WS = prob.get_val(Aircraft.Design.WING_LOADING)

        TW_expected = 0.161183141853
        WS_expected = 111.4393856281

        assert_near_equal(TW, TW_expected, 1e-12)
        assert_near_equal(WS, WS_expected, 1e-12)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
