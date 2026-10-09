import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.flops_based.canard import CanardMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class CanardMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        # No test cases with canards. Use dummy vars. See issue #1091
        prob = self.prob

        prob.model.add_subsystem(
            'canard',
            CanardMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Design.GROSS_MASS, 100000, 'lbm')
        prob.set_val(Aircraft.Canard.AREA, 250.00, 'ft**2')
        prob.set_val(Aircraft.Canard.TAPER_RATIO, 0.330, 'unitless')
        prob.set_val(Aircraft.Canard.MASS_SCALER, 1.0, 'unitless')

        prob.run_model()

        with self.subTest(var='canard_mass'):
            actual = prob.get_val(Aircraft.Canard.MASS, units='lbm')
            assert_near_equal(actual, 1099.75, 1.0e-3)

        with self.subTest('check_partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_alt_gravity(self):
        prob = self.prob

        prob.model.add_subsystem(
            'canard',
            CanardMass(**{Mission.GRAVITY: (30, 'ft/s**2')}),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Design.GROSS_MASS, 100000, 'lbm')
        prob.set_val(Aircraft.Canard.AREA, 250.00, 'ft**2')
        prob.set_val(Aircraft.Canard.TAPER_RATIO, 0.330, 'unitless')
        prob.set_val(Aircraft.Canard.MASS_SCALER, 1.0, 'unitless')
        prob.set_val(Aircraft.Canard.MASS, 1099.75, 'lbm')

        prob.run_model()

        with self.subTest(var='canard_mass'):
            actual = prob.get_val(Aircraft.Canard.MASS, units='lbm')
            assert_near_equal(actual, 1084.46884262, 1e-10)

        with self.subTest('check_partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
