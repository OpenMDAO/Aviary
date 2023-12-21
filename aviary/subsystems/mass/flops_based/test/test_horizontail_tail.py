import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.horizontal_tail import (AltHorizontalTailMass,
                                                                HorizontalTailMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class ExplicitHorizontalTailMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "horizontal_tail",
            HorizontalTailMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.HorizontalTail.AREA,
                        Aircraft.HorizontalTail.TAPER_RATIO,
                        Mission.Design.GROSS_MASS,
                        Aircraft.HorizontalTail.MASS_SCALER],
            output_keys=Aircraft.HorizontalTail.MASS,
            version=Version.TRANSPORT,
            tol=2.0e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class ExplicitAltHorizontalTailMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "horizontal_tail",
            AltHorizontalTailMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.HorizontalTail.AREA,
                        Aircraft.HorizontalTail.MASS_SCALER],
            output_keys=Aircraft.HorizontalTail.MASS,
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
