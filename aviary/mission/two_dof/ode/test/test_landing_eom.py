import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.mission.two_dof.ode.landing_eom import (
    GlideConditionComponent,
    LandingAltitudeComponent,
    LandingGroundRollComponent,
)
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LandingAltTestCase(unittest.TestCase):
    """Test computation of initial altitude in LandingAltitudeComponent component."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', LandingAltitudeComponent(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.Landing.OBSTACLE_HEIGHT, 50, units='ft')
        self.prob.model.set_input_defaults(Mission.Landing.AIRPORT_ALTITUDE, 0, units='ft')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Mission.Landing.OBSTACLE_HEIGHT], 50, tol
        )  # not actual GASP value, but intuitively correct

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class GlideTestCase(unittest.TestCase):
    """Test computation of initial velocity and stall velocity in GlideConditionComponent component."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', GlideConditionComponent(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, RHO_SEA_LEVEL_ENGLISH, units='slug/ft**3'
        )  # value from online calculator

        self.prob.model.set_input_defaults(Mission.Landing.MAXIMUM_SINK_RATE, 900, units='ft/min')

        self.prob.model.set_input_defaults('mass', 165279, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Mission.Landing.GLIDE_TO_STALL_RATIO, 1.3, units='unitless'
        )
        self.prob.model.set_input_defaults('CL_max', 2.9533, units='unitless')
        self.prob.model.set_input_defaults(
            Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR, 1.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Landing.TOUCHDOWN_SINK_RATE, 5, units='ft/s')
        self.prob.model.set_input_defaults(Mission.Landing.INITIAL_ALTITUDE, val=50.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Landing.BRAKING_DELAY, val=1.0, units='s')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        # note: actual GASP values differ slightly
        # GASP values: INITIAL_VELOCITY=142.74, STALL_VELOCITY=109.73, TAS_touchdown=126.27,
        #              density_ratio=.998739, wing_loading_land=120.61, theta=3.57,
        #              glide_distance=802, tr_distance=167, delay_distance=213, flare_alt=20.8
        expected_values = {
            (Mission.Landing.INITIAL_VELOCITY, 'kn'): 142.783,
            (Mission.Landing.STALL_VELOCITY, 'kn'): 109.8331,
            ('TAS_touchdown', 'kn'): 126.3081,
            ('density_ratio', 'unitless'): 1.0,
            ('wing_loading_land', 'lbf/ft**2'): 120.61519375,
            ('theta', 'deg'): 3.56857,
            ('glide_distance', 'ft'): 801.7444,
            ('tr_distance', 'ft'): 166.5303,
            ('delay_distance', 'ft'): 213.184,
            ('flare_alt', 'ft'): 20.73407,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)


class GlideTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.two_dof.ode.landing_eom as landing

        landing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.two_dof.ode.landing_eom as landing

        landing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('group', GlideConditionComponent(), promotes=['*'])
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.DENSITY, RHO_SEA_LEVEL_ENGLISH, units='slug/ft**3'
        )
        prob.model.set_input_defaults(Mission.Landing.MAXIMUM_SINK_RATE, 900, units='ft/min')
        prob.model.set_input_defaults('mass', 165279, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        prob.model.set_input_defaults(Mission.Landing.GLIDE_TO_STALL_RATIO, 1.3, units='unitless')
        prob.model.set_input_defaults('CL_max', 2.9533, units='unitless')
        prob.model.set_input_defaults(
            Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR, 1.15, units='unitless'
        )
        prob.model.set_input_defaults(Mission.Landing.TOUCHDOWN_SINK_RATE, 5, units='ft/s')
        prob.model.set_input_defaults(Mission.Landing.INITIAL_ALTITUDE, val=50.0, units='ft')
        prob.model.set_input_defaults(Mission.Landing.BRAKING_DELAY, val=1.0, units='s')
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)


class GroundRollTestCase(unittest.TestCase):
    """
    Test the computation of groundroll distance and average acceleration/deceleration in
    LandingGroundRollComponent component.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', LandingGroundRollComponent(), promotes=['*'])

        self.prob.model.set_input_defaults('touchdown_CD', val=0.07344)
        self.prob.model.set_input_defaults('touchdown_CL', val=1.18694)
        self.prob.model.set_input_defaults(
            Mission.Landing.STALL_VELOCITY, val=109.73, units='kn'
        )  # note: EAS in GASP, although at this altitude they are nearly identical
        self.prob.model.set_input_defaults('TAS_touchdown', val=126.27, units='kn')
        self.prob.model.set_input_defaults('thrust_idle', val=1276, units='lbf')
        self.prob.model.set_input_defaults(
            'density_ratio', val=0.998739, units='unitless'
        )  # note: calculated from GASP glide speed values
        self.prob.model.set_input_defaults('wing_loading_land', val=120.61, units='lbf/ft**2')
        self.prob.model.set_input_defaults('glide_distance', val=802, units='ft')
        self.prob.model.set_input_defaults('tr_distance', val=167, units='ft')
        self.prob.model.set_input_defaults('delay_distance', val=213, units='ft')
        self.prob.model.set_input_defaults('CL_max', 2.9533, units='unitless')
        self.prob.model.set_input_defaults('mass', 165279, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        # note: actual GASP values differ: ground_roll_distance=1798, GROUND_DISTANCE=2980,
        #       average_acceleration=0.3932
        expected_values = {
            'ground_roll_distance': 2406.43116212,
            Mission.Landing.GROUND_DISTANCE: 3588.43116212,
            'average_acceleration': 0.29308129,
        }

        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-12, rtol=1e-12)


class GroundRollTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.two_dof.ode.landing_eom as landing

        landing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.two_dof.ode.landing_eom as landing

        landing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', LandingGroundRollComponent(), promotes=['*'])

        self.prob.model.set_input_defaults('touchdown_CD', val=0.07344)
        self.prob.model.set_input_defaults('touchdown_CL', val=1.18694)
        self.prob.model.set_input_defaults(
            Mission.Landing.STALL_VELOCITY, val=109.73, units='kn'
        )  # note: EAS in GASP, although at this altitude they are nearly identical
        self.prob.model.set_input_defaults('TAS_touchdown', val=126.27, units='kn')
        self.prob.model.set_input_defaults('thrust_idle', val=1276, units='lbf')
        self.prob.model.set_input_defaults(
            'density_ratio', val=0.998739, units='unitless'
        )  # note: calculated from GASP glide speed values
        self.prob.model.set_input_defaults('wing_loading_land', val=120.61, units='lbf/ft**2')
        self.prob.model.set_input_defaults('glide_distance', val=802, units='ft')
        self.prob.model.set_input_defaults('tr_distance', val=167, units='ft')
        self.prob.model.set_input_defaults('delay_distance', val=213, units='ft')
        self.prob.model.set_input_defaults('CL_max', 2.9533, units='unitless')
        self.prob.model.set_input_defaults('mass', 165279, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
