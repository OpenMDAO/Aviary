import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.nacelle import NacelleMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class NacelleMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "nacelle",
            NacelleMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=[Aircraft.Nacelle.AVG_DIAMETER,
                        Aircraft.Nacelle.AVG_LENGTH,
                        Aircraft.Nacelle.MASS_SCALER,
                        Aircraft.Engine.SCALED_SLS_THRUST],
            output_keys=Aircraft.Nacelle.MASS)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
