import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.unusable_fuel import (
    AltUnusableFuelMass, TransportUnusableFuelMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft

# TODO: update non-transport tests to remove these variables
_wing_ref_area = 'aircraft:wing:dimensions:area'
_engine_count_factor = 'aircraft:propulsion:control:engine_count_factor'
_max_scaled_thrust = 'aircraft:propulsion:dimensions:max_scaled_thrust'
_total_fuel_capacity = 'aircraft:fuel:mass:total_capacity'
_density_ratio = 'aircraft:fuel:mass:density_ratio'
_tank_count = 'aircraft:fuel:control:tank_count'
_total_fuel_vol = 'TBD:total_fuel_vol:'
_unusable_fuel_mass = 'TBD:unusable_fuel'


class TransportUnusableFuelMassTest(unittest.TestCase):
    '''
    Tests transport/GA unusable fuel mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob
        flops_inputs = get_flops_inputs(case_name)

        prob.model.add_subsystem(
            'unusable_fuel',
            TransportUnusableFuelMass(aviary_options=flops_inputs),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
                        Aircraft.Fuel.DENSITY_RATIO,
                        Aircraft.Fuel.TOTAL_CAPACITY,
                        Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                        Aircraft.Wing.AREA],
            output_keys=[  # Aircraft.Fuel.TOTAL_VOLUME,
                Aircraft.Fuel.UNUSABLE_FUEL_MASS],
            version=Version.TRANSPORT,
            tol=5e-4,
            excludes=['size_prop.*'])

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltUnusableFuelMassTest(unittest.TestCase):
    '''
    Tests alternate unusable fuel mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'unusable_fuel',
            AltUnusableFuelMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
                        Aircraft.Fuel.TOTAL_CAPACITY],
            output_keys=[  # Aircraft.Fuel.TOTAL_VOLUME,
                Aircraft.Fuel.UNUSABLE_FUEL_MASS],
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
