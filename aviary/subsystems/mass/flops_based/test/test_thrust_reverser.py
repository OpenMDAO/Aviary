import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.thrust_reverser import ThrustReverserMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class ThrustReverserMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=['LargeSingleAisle1FLOPS', 'N3CC']),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "thrust_rev",
            ThrustReverserMass(aviary_options=get_flops_inputs(case_name)),
            promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Engine.THRUST_REVERSERS_MASS_SCALER,
                        Aircraft.Engine.SCALED_SLS_THRUST],
            output_keys=[Aircraft.Engine.THRUST_REVERSERS_MASS,
                         Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS])

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
