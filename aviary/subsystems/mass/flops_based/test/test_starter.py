import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.starter import TransportStarterMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class TransportStarterMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit='N3CC'),
                          name_func=print_case)
    def test_case_1(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "starter_test",
            TransportStarterMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Nacelle.AVG_DIAMETER],
            output_keys=Aircraft.Propulsion.TOTAL_STARTER_MASS)

    def test_case_2(self):
        # test with more than 4 engines
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([5]))
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 5)
        aviary_options.set_val(Mission.Constraints.MAX_MACH, 0.785)

        prob.model.add_subsystem(
            "starter_test",
            TransportStarterMass(aviary_options=aviary_options),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94]), 'ft')

        prob.run_model()

        mass = prob.get_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 'lbm')
        expected_mass = 1555.38298314

        assert_near_equal(mass, expected_mass, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
