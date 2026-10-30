import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fin import FinMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS', 'BWB300FLOPS']


@use_tempdirs
class FinMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def _make_prob(self, gravity=9.80665, gravity_units='m/s**2'):
        prob = self.prob

        prob.model.add_subsystem(
            'fin',
            FinMass(**{Aircraft.Fins.NUM_FINS: 1, Mission.GRAVITY: (gravity, gravity_units)}),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Design.GROSS_MASS, 100000, 'lbm')
        prob.set_val(Aircraft.Fins.AREA, 250.00, 'ft**2')
        prob.set_val(Aircraft.Fins.TAPER_RATIO, 0.3300, 'unitless')
        prob.set_val(Aircraft.Fins.MASS_SCALER, 1.0, 'unitless')

        return prob

    def test_case1(self):
        prob = self._make_prob()
        prob.run_model()

        with self.subTest(var='fin_mass'):
            actual = prob.get_val(Aircraft.Fins.MASS, units='lbm')
            assert_near_equal(actual, 917.228, 1.0e-3)

        with self.subTest('check_partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    def test_alt_gravity(self):
        prob = self._make_prob(gravity=30, gravity_units='ft/s**2')
        prob.run_model()

        with self.subTest(var='fin_mass'):
            actual = prob.get_val(Aircraft.Fins.MASS, units='lbm')
            assert_near_equal(actual, 898.1766089260107, 1e-10)

        with self.subTest('check_partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    def test_IO(self):
        self._make_prob()
        assert_match_varnames(self.prob.model)

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_bwb_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'fin',
            FinMass(),
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
                Aircraft.Design.GROSS_MASS,
                Aircraft.Fins.TAPER_RATIO,
                Aircraft.Fins.AREA,
            ],
            output_keys=[
                Aircraft.Fins.MASS,
            ],
            list_inputs=False,
            list_outputs=False,
            version=Version.BWB,
            rtol=1e-10,
        )


if __name__ == '__main__':
    unittest.main()
