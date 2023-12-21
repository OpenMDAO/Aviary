import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.electrical import (AltElectricalMass,
                                                           ElectricalMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class ElectricMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "electric_test",
            ElectricalMass(aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_outputs=[
                Aircraft.Electrical.MASS,
            ],
            promotes_inputs=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Electrical.MASS_SCALER
            ]
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=[Aircraft.Fuselage.LENGTH,
                        Aircraft.Fuselage.MAX_WIDTH,
                        Aircraft.Electrical.MASS_SCALER],
            output_keys=Aircraft.Electrical.MASS,
            version=Version.TRANSPORT)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltElectricMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "electric_test",
            AltElectricalMass(aviary_options=get_flops_inputs(
                case_name, preprocess=True)),
            promotes_outputs=[
                Aircraft.Electrical.MASS,
            ],
            promotes_inputs=[
                Aircraft.Electrical.MASS_SCALER
            ]
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self.prob,
            case_name,
            input_keys=Aircraft.Electrical.MASS_SCALER,
            output_keys=Aircraft.Electrical.MASS,
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
