import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fuselage import (AltFuselageMass,
                                                         TransportFuselageMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class FuselageMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "fuselage",
            TransportFuselageMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.LENGTH,
                        Aircraft.Fuselage.AVG_DIAMETER,
                        Aircraft.Fuselage.MASS_SCALER],
            output_keys=Aircraft.Fuselage.MASS,
            version=Version.TRANSPORT,
            atol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltFuselageMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "fuselage",
            AltFuselageMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.MASS_SCALER,
                        Aircraft.Fuselage.WETTED_AREA,
                        Aircraft.Fuselage.MAX_HEIGHT,
                        Aircraft.Fuselage.MAX_WIDTH],
            output_keys=Aircraft.Fuselage.MASS,
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
