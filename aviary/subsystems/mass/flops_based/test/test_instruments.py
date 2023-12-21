import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.instruments import TransportInstrumentMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class TransportInstrumentsMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "instruments_tests",
            TransportInstrumentMass(
                aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_outputs=[
                Aircraft.Instruments.MASS,
            ],
            promotes_inputs=[
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Instruments.MASS_SCALER
            ]
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.PLANFORM_AREA,
                        Aircraft.Instruments.MASS_SCALER],
            output_keys=Aircraft.Instruments.MASS,
            tol=1e-3)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
