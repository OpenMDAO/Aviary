import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.unusable_fuel import (
    AltUnusableFuelMass,
    TransportUnusableFuelMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


class TransportUnusableFuelMassTest(unittest.TestCase):
    """Tests transport/GA unusable fuel mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'unusable_fuel',
            TransportUnusableFuelMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.TOTAL_CAPACITY,
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                Aircraft.Wing.AREA,
            ],
            output_keys=[  # Aircraft.Fuel.TOTAL_VOLUME,
                Aircraft.Fuel.UNUSABLE_FUEL_MASS
            ],
            version=Version.TRANSPORT,
            tol=5e-4,
            excludes=['size_prop.*'],
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportUnusableFuelMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.unusable_fuel as ufuel

        ufuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.unusable_fuel as ufuel

        ufuel.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'unusable_fuel',
            TransportUnusableFuelMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 30000.0, 'lbm')
        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 40000.0, 'lbf')
        prob.set_val(Aircraft.Wing.AREA, 1000.0, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AltUnusableFuelMassTest(unittest.TestCase):
    """Tests alternate unusable fuel mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'unusable_fuel', AltUnusableFuelMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, Aircraft.Fuel.TOTAL_CAPACITY],
            output_keys=[  # Aircraft.Fuel.TOTAL_VOLUME,
                Aircraft.Fuel.UNUSABLE_FUEL_MASS
            ],
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltUnusableFuelMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.unusable_fuel as ufuel

        ufuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.unusable_fuel as ufuel

        ufuel.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'unusable_fuel', AltUnusableFuelMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuel.TOTAL_CAPACITY, 30000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
