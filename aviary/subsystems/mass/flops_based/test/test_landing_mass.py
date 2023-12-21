import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.landing_mass import LandingMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class LandingMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit='LargeSingleAisle1FLOPS'),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob
        prob.model.add_subsystem(
            "landing_mass",
            LandingMass(aviary_options=get_flops_inputs(case_name)),
            promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Mission.Design.GROSS_MASS,
                        Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO],
            output_keys=Aircraft.Design.TOUCHDOWN_MASS)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
