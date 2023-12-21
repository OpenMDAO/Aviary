import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.engine_controls import \
    TransportEngineCtrlsMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class BasicTransportEngineCtrlsTest(unittest.TestCase):
    '''
    Test the BasicTransportEngineCtrls component.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit='N3CC'),
                          name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)

        prob = self.prob

        prob.model.add_subsystem(
            'engine_ctrls',
            TransportEngineCtrlsMass(aviary_options=flops_inputs),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST],
            output_keys=Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
            atol=2e-12,
            excludes=['size_prop.*'])

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
