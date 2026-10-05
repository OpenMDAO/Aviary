import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.landing_gear import AltLandingGearMass, LandingGearMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    print_case,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class LandingGearMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'landing_gear',
            LandingGearMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            self.prob,
            case_name,
            input_keys=[
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                Aircraft.Design.TOUCHDOWN_MASS_MAX,
            ],
            output_keys=[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_MASS],
            version=Version.TRANSPORT_and_BWB,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_alt_gravity(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'landing_gear',
            LandingGearMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = {Mission.GRAVITY: (35, 'ft/s**2')}

        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 106.94, units='inch'
        )
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER, 0.8846)
        prob.model.set_input_defaults(
            Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 74.86, units='inch'
        )
        prob.model.set_input_defaults(Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER, 0.8846)
        prob.model.set_input_defaults(Aircraft.Design.TOUCHDOWN_MASS_MAX, 108976.4, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)

        prob.run_model()

        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: (5101.09161083, 'lbm'),
            Aircraft.LandingGear.NOSE_GEAR_MASS: (681.43604869, 'lbm'),
        }

        tol = 1e-10
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class AltLandingGearMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'landing_gear_alt',
            AltLandingGearMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            self.prob,
            case_name,
            input_keys=[
                Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.MAIN_GEAR_MASS_SCALER,
                Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH,
                Aircraft.LandingGear.NOSE_GEAR_MASS_SCALER,
                Aircraft.Design.GROSS_MASS,
            ],
            output_keys=[Aircraft.LandingGear.MAIN_GEAR_MASS, Aircraft.LandingGear.NOSE_GEAR_MASS],
            version=Version.ALTERNATE,
            atol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_case_alt_gravity(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'landing_gear_alt',
            AltLandingGearMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.LandingGear.MAIN_GEAR_OLEO_LENGTH, 100.0, 'inch')
        prob.set_val(Aircraft.LandingGear.NOSE_GEAR_OLEO_LENGTH, 75.0, 'inch')
        prob.set_val(Aircraft.Design.GROSS_MASS, 100000.0, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
