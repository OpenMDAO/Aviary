import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.anti_icing import AntiIcingMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


class AntiIcingMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.AntiIcing.MASS_SCALER,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.SWEEP,
                Aircraft.Engine.SCALED_SLS_THRUST,
            ],
            output_keys=Aircraft.AntiIcing.MASS,
            tol=3.0e-3,
        )

    def test_case_2(self):
        # test with more than four engines
        prob = self.prob

        options = get_flops_options('LargeSingleAisle1FLOPS')
        options[Aircraft.Engine.NUM_ENGINES] = np.array([5])
        options[Aircraft.Propulsion.TOTAL_NUM_ENGINES] = 5

        prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = options

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.33, 'ft')
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94]), 'ft')
        prob.set_val(Aircraft.Wing.SPAN, 117.83, 'ft')
        prob.set_val(Aircraft.Wing.SWEEP, 25.0, 'deg')
        prob.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST,
            np.array(
                [
                    28928.1,
                ]
            ),
            'lbf',
        )

        prob.run_model()

        mass = prob.get_val(Aircraft.AntiIcing.MASS)
        expected_mass = 305.14673602

        assert_near_equal(mass, expected_mass, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_case_3(self):
        # test with multiple engine types
        prob = self.prob

        options = get_flops_options('LargeSingleAisle1FLOPS')
        options[Aircraft.Engine.NUM_ENGINES] = np.array([2, 2, 4])
        options[Aircraft.Propulsion.TOTAL_NUM_ENGINES] = 8

        prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = options
        prob.model_options[Aircraft.Engine.REFERENCE_SLS_THRUST] = np.array(
            [28928.1, 28928.1, 28928.1]
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.33, 'ft')
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94, 8, 5]), 'ft')
        prob.set_val(Aircraft.Wing.SPAN, 117.83, 'ft')
        prob.set_val(Aircraft.Wing.SWEEP, 25.0, 'deg')
        prob.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST, np.array([28928.1, 28928.1, 28928.1]), 'lbf'
        )

        prob.run_model()

        mass = prob.get_val(Aircraft.AntiIcing.MASS)
        expected_mass = 352.5412182

        assert_near_equal(mass, expected_mass, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs', compact_print=False)
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AntiIcingMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.anti_icing as antiicing

        antiicing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.anti_icing as antiicing

        antiicing.GRAV_ENGLISH_LBM = 1.0

    def test_case_2(self):
        prob = om.Problem()

        options = get_flops_options('AdvancedSingleAisle')
        options[Aircraft.Engine.NUM_ENGINES] = np.array([5])
        options[Aircraft.Propulsion.TOTAL_NUM_ENGINES] = 5

        prob.model.add_subsystem(
            'anti_icing',
            AntiIcingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = options

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.AntiIcing.MASS_SCALER, 1.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 12.33, 'ft')
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, np.array([7.94]), 'ft')
        prob.set_val(Aircraft.Wing.SPAN, 117.83, 'ft')
        prob.set_val(Aircraft.Wing.SWEEP, 25.0, 'deg')
        prob.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST,
            np.array(
                [
                    28928.1,
                ]
            ),
            'lbf',
        )

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
