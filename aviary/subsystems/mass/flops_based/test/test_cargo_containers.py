import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.cargo_containers import \
    TransportCargoContainersMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class CargoContainerMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "cargo_containers",
            TransportCargoContainersMass(
                aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.CrewPayload.CARGO_CONTAINER_MASS_SCALER,
                        Aircraft.CrewPayload.CARGO_MASS,
                        Aircraft.CrewPayload.BAGGAGE_MASS],
            output_keys=Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
