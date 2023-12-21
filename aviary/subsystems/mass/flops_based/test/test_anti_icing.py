import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.anti_icing import AntiIcingMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class AntiIcingMassTest(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "anti_icing",
            AntiIcingMass(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.AntiIcing.MASS_SCALER,
                        Aircraft.Fuselage.MAX_WIDTH,
                        Aircraft.Nacelle.AVG_DIAMETER,
                        Aircraft.Wing.SPAN,
                        Aircraft.Wing.SWEEP],
            output_keys=Aircraft.AntiIcing.MASS,
            tol=3.0e-3)

    def test_case_2(self):
        # test with more than four engines
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([5]))
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 5)

        prob.model.add_subsystem(
            "anti_icing",
            AntiIcingMass(aviary_options=aviary_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.33, 'ft')
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94]), 'ft')
        prob.set_val(Aircraft.Wing.SPAN, 117.83, 'ft')
        prob.set_val(Aircraft.Wing.SWEEP, 25.0, 'deg')

        prob.run_model()

        mass = prob.get_val(Aircraft.AntiIcing.MASS)
        expected_mass = 305.14673602

        assert_near_equal(mass, expected_mass, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
