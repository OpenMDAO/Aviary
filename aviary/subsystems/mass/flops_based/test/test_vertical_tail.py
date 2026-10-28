import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.vertical_tail import AltVerticalTailMass, VerticalTailMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class VerticalTailMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'vertical_tail',
            VerticalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.VerticalTail.AREA,
                Aircraft.VerticalTail.TAPER_RATIO,
                Aircraft.Design.GROSS_MASS,
                Aircraft.VerticalTail.MASS_SCALER,
            ],
            output_keys=Aircraft.VerticalTail.MASS,
            version=Version.TRANSPORT_and_BWB,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_alt_gravity(self):
        prob = self.prob

        prob.model.add_subsystem(
            'vertical_tail',
            VerticalTailMass(**{Mission.GRAVITY: (30, 'ft/s**2')}),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Design.GROSS_MASS, 100000, 'lbm')
        prob.set_val(Aircraft.VerticalTail.AREA, 250, 'ft**2')
        prob.set_val(Aircraft.VerticalTail.TAPER_RATIO, 0.33, 'unitless')
        prob.set_val(Aircraft.VerticalTail.MASS_SCALER, 1.0, 'unitless')

        prob.run_model()

        with self.subTest(var='vertical_tail_mass'):
            actual = prob.get_val(Aircraft.VerticalTail.MASS, units='lbm')
            assert_near_equal(actual, 898.1766089260108, 1e-10)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class AltVerticalTailMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'vertical_tail',
            AltVerticalTailMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[Aircraft.VerticalTail.AREA, Aircraft.VerticalTail.MASS_SCALER],
            output_keys=Aircraft.VerticalTail.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
