import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.engine_oil import (
    AltEngineOilMass, TransportEngineOilMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class TransportEngineOilMassTest(unittest.TestCase):
    '''
    Tests transport/GA engine oil mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'engine_oil',
            TransportEngineOilMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER,
                        Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST],
            output_keys=[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS],
            version=Version.TRANSPORT,
            tol=4.0e-3)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltEngineOilMassTest(unittest.TestCase):
    '''
    Tests alternate engine oil mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'engine_oil',
            AltEngineOilMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Propulsion.ENGINE_OIL_MASS_SCALER],
            output_keys=[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS],
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
