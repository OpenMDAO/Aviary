import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.starter import TransportStarterMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission


class TransportStarterMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit='AdvancedSingleAisle'), name_func=print_case)
    def test_case_1(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Engine.NUM_ENGINES: inputs.get_val(Aircraft.Engine.NUM_ENGINES),
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'starter_test',
            TransportStarterMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Nacelle.AVG_DIAMETER],
            output_keys=Aircraft.Propulsion.TOTAL_STARTER_MASS,
        )

    def test_case_2(self):
        # test with more than 4 engines
        prob = self.prob

        options = {
            Aircraft.Engine.NUM_ENGINES: np.array([5]),
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: 5,
            Mission.Constraints.MAX_MACH: 0.785,
        }

        prob.model.add_subsystem(
            'starter_test',
            TransportStarterMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94]), 'ft')

        prob.run_model()

        mass = prob.get_val(Aircraft.Propulsion.TOTAL_STARTER_MASS, 'lbm')
        expected_mass = 1555.38298314

        assert_near_equal(mass, expected_mass, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportStarterMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.starter as starter

        starter.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.starter as starter

        starter.GRAV_ENGLISH_LBM = 1.0

    def test_case_2(self):
        prob = om.Problem()

        options = {
            Aircraft.Engine.NUM_ENGINES: np.array([5]),
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: 5,
            Mission.Constraints.MAX_MACH: 0.785,
        }

        prob.model.add_subsystem(
            'starter_test',
            TransportStarterMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94]), 'ft')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
