import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.unusable_fuel import (
    AltUnusableFuelMass,
    TransportUnusableFuelMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
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
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.MAX_CAPACITY_MASS,
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                Aircraft.Wing.AREA,
            ],
            output_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS],
            version=Version.TRANSPORT_and_BWB,
            tol=5e-4,
            excludes=['size_prop.*'],
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
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
            self,
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, Aircraft.Fuel.MAX_CAPACITY_MASS],
            output_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS],
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
