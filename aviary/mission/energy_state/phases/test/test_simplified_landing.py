import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.energy_state.phases.simplified_landing import LandingCalc, LandingGroup
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class LandingCalcTest(unittest.TestCase):
    """Test computation in LandingCalc class (the simplified landing)."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'landing',
            LandingCalc(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.FINAL_MASS, val=152800.0, units='lbm')
        self.prob.model.set_input_defaults(Dynamic.Atmosphere.DENSITY, val=1.225, units='kg/m**3')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless'
        )

    def test_case_1(self):
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-5

        expected_values = {
            Mission.Landing.GROUND_DISTANCE: (6403.64963504, 'ft'),
            Mission.Landing.INITIAL_VELOCITY: (136.22914933, 'kn'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_case_alt_gravity(self):
        self.prob.model_options['*'] = {Mission.GRAVITY: (10, 'm/s**2')}

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-5

        expected_values = {
            Mission.Landing.GROUND_DISTANCE: (6480.61482263, 'ft'),
            Mission.Landing.INITIAL_VELOCITY: (137.56580613, 'kn'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class LandingGroupTest(unittest.TestCase):
    """Test the computation of LandingGroup."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'land',
            LandingGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Dynamic.Mission.VELOCITY, val=100, units='knot')
        self.prob.model.set_input_defaults(Mission.FINAL_MASS, val=152800.0, units='lbm')
        self.prob.model.set_input_defaults(Mission.Landing.INITIAL_ALTITUDE, val=35, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.0, units='ft**2')
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case_1(self):
        self.prob.run_model()

        tol = 1e-5

        expected_values = {
            Mission.Landing.GROUND_DISTANCE: (6407.65299289, 'ft'),
            Mission.Landing.INITIAL_VELOCITY: (136.29923391, 'kn'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(
            out_stream=None, excludes=['*.standard_atmosphere'], method='cs'
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
    # test = LandingCalcTest()
    # test.setUp()
    # test.test_case_alt_gravity()
