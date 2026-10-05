import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.performance.performance_premission import PerformancePremission
from aviary.variable_info.variables import Aircraft


class PerformancePremissionTest(unittest.TestCase):
    """Test computation of parameters in performance premission."""

    def test_case(self):
        prob = om.Problem()

        options = {
            Mission.GRAVITY: (9.80665, 'm/s**2'),
        }
        prob.model.add_subsystem(
            'perf_premission', PerformancePremission(**options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        # arbitrary numbers in roughly correct order of magnitude for testing
        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 32745, 'lbf')
        prob.set_val(Aircraft.Wing.AREA, 1823, 'ft**2')
        prob.set_val(Aircraft.Design.GROSS_MASS, 203154, 'lbm')

        prob.run_model()

        # lower tol because of small discrepency between gravity value used by OM and Aviary causes impersision in unit conversion
        expected_values = {
            Aircraft.Design.THRUST_TO_WEIGHT_RATIO: (0.161183141853, 'unitless'),
            Aircraft.Design.WING_LOADING: (111.4393856281, 'lbf/ft**2'),
        }

        with self.subTest(check='value'):
            for var_name, (expected, units) in expected_values.items():
                with self.subTest(var=var_name):
                    actual = prob.get_val(var_name, units=units)
                    assert_near_equal(actual, expected, 1e-8)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
