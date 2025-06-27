import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.engine_controls import TransportEngineCtrlsMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


class BasicTransportEngineCtrlsTest(unittest.TestCase):
    """Test the BasicTransportEngineCtrls component."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit='AdvancedSingleAisle'), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'engine_ctrls',
            TransportEngineCtrlsMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST],
            output_keys=Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
            atol=2e-12,
            excludes=['size_prop.*'],
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class BasicTransportEngineCtrlsTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.engine_controls as control

        control.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.engine_controls as control

        control.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'engine_ctrls',
            TransportEngineCtrlsMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('LargeSingleAisle1FLOPS', preprocess=True)

        prob.setup(force_alloc_complex=True)
        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 50000.0, 'lbf')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
