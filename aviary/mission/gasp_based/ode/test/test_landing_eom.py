import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.ode.landing_eom import (
    GlideConditionComponent, LandingAltitudeComponent,
    LandingGroundRollComponent)
from aviary.variable_info.variables import Aircraft, Mission


class LandingAltTestCase(unittest.TestCase):
    """
    Test computation of initial altitude in LandingAltitudeComponent component
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", LandingAltitudeComponent(), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Mission.Landing.OBSTACLE_HEIGHT, 50, units="ft")
        self.prob.model.set_input_defaults(
            Mission.Landing.AIRPORT_ALTITUDE, 0, units="ft")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Mission.Landing.OBSTACLE_HEIGHT], 50, tol
        )  # not actual GASP value, but intuitively correct

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class GlideTestCase(unittest.TestCase):
    """
    Test computation of initial velocity and stall velocity in GlideConditionComponent component
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", GlideConditionComponent(), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            "rho_app", RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3"
        )  # value from online calculator

        self.prob.model.set_input_defaults(
            Mission.Landing.MAXIMUM_SINK_RATE, 900, units="ft/min")

        self.prob.model.set_input_defaults("mass", 165279, units="lbm")
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units="ft**2")
        self.prob.model.set_input_defaults(
            Mission.Landing.GLIDE_TO_STALL_RATIO, 1.3, units="unitless")
        self.prob.model.set_input_defaults("CL_max", 2.9533, units="unitless")
        self.prob.model.set_input_defaults(
            Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR, 1.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Mission.Landing.TOUCHDOWN_SINK_RATE, 5, units="ft/s")
        self.prob.model.set_input_defaults(
            Mission.Landing.INITIAL_ALTITUDE, val=50.0, units="ft")
        self.prob.model.set_input_defaults(
            Mission.Landing.BRAKING_DELAY, val=1.0, units="s")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob.get_val(Mission.Landing.INITIAL_VELOCITY,
                              units="kn"), 142.783, tol
        )  # note: actual GASP value is: 142.74
        assert_near_equal(
            self.prob.get_val(Mission.Landing.STALL_VELOCITY,
                              units="kn"), 109.8331, tol
        )  # note: EAS in GASP, although at this altitude they are nearly identical. actual GASP value is 109.73
        assert_near_equal(
            self.prob.get_val("TAS_touchdown", units="kn"), 126.3081, tol
        )  # note: actual GASP value is: 126.27
        assert_near_equal(
            self.prob.get_val("density_ratio", units="unitless"), 1.0, tol
        )  # note: calculated from GASP glide speed values as: .998739
        assert_near_equal(
            self.prob.get_val("wing_loading_land", units="lbf/ft**2"), 120.61519375, tol
        )  # note: actual GASP value is: 120.61
        assert_near_equal(
            self.prob.get_val("theta", units="deg"), 3.56857, tol
        )  # note: actual GASP value is: 3.57
        assert_near_equal(
            self.prob.get_val("glide_distance", units="ft"), 801.7444, tol
        )  # note: actual GASP value is: 802
        assert_near_equal(
            self.prob.get_val("tr_distance", units="ft"), 166.5303, tol
        )  # note: actual GASP value is: 167
        assert_near_equal(
            self.prob.get_val("delay_distance", units="ft"), 213.184, tol
        )  # note: actual GASP value is: 213
        assert_near_equal(
            self.prob.get_val("flare_alt", units="ft"), 20.73407, tol
        )  # note: actual GASP value is: 20.8

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-12)


class GlideTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.gasp_based.ode.landing_eom as landing
        landing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.landing_eom as landing
        landing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "group", GlideConditionComponent(), promotes=["*"])
        prob.model.set_input_defaults(
            "rho_app", RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3")
        prob.model.set_input_defaults(
            Mission.Landing.MAXIMUM_SINK_RATE, 900, units="ft/min")
        prob.model.set_input_defaults("mass", 165279, units="lbm")
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units="ft**2")
        prob.model.set_input_defaults(
            Mission.Landing.GLIDE_TO_STALL_RATIO, 1.3, units="unitless")
        prob.model.set_input_defaults("CL_max", 2.9533, units="unitless")
        prob.model.set_input_defaults(
            Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR, 1.15, units="unitless")
        prob.model.set_input_defaults(
            Mission.Landing.TOUCHDOWN_SINK_RATE, 5, units="ft/s")
        prob.model.set_input_defaults(
            Mission.Landing.INITIAL_ALTITUDE, val=50.0, units="ft")
        prob.model.set_input_defaults(
            Mission.Landing.BRAKING_DELAY, val=1.0, units="s")
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-12)


class GroundRollTestCase(unittest.TestCase):
    """
    Test the computation of groundroll distance and average acceleration/deceleration in 
    LandingGroundRollComponent component.
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", LandingGroundRollComponent(), promotes=["*"]
        )

        self.prob.model.set_input_defaults("touchdown_CD", val=0.07344)
        self.prob.model.set_input_defaults("touchdown_CL", val=1.18694)
        self.prob.model.set_input_defaults(
            Mission.Landing.STALL_VELOCITY, val=109.73, units="kn"
        )  # note: EAS in GASP, although at this altitude they are nearly identical
        self.prob.model.set_input_defaults("TAS_touchdown", val=126.27, units="kn")
        self.prob.model.set_input_defaults("thrust_idle", val=1276, units="lbf")
        self.prob.model.set_input_defaults(
            "density_ratio", val=0.998739, units="unitless"
        )  # note: calculated from GASP glide speed values
        self.prob.model.set_input_defaults(
            "wing_loading_land", val=120.61, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults("glide_distance", val=802, units="ft")
        self.prob.model.set_input_defaults("tr_distance", val=167, units="ft")
        self.prob.model.set_input_defaults("delay_distance", val=213, units="ft")
        self.prob.model.set_input_defaults("CL_max", 2.9533, units="unitless")
        self.prob.model.set_input_defaults("mass", 165279, units="lbm")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob["ground_roll_distance"], 2406.43116212, tol
        )  # actual GASP value is: 1798
        assert_near_equal(
            self.prob[Mission.Landing.GROUND_DISTANCE], 3588.43116212, tol
        )  # actual GASP value is: 2980
        assert_near_equal(
            self.prob["average_acceleration"], 0.29308129, tol
        )  # actual GASP value is: 0.3932

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=5e-12, rtol=1e-12)


class GroundRollTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.gasp_based.ode.landing_eom as landing
        landing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.landing_eom as landing
        landing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", LandingGroundRollComponent(), promotes=["*"]
        )

        self.prob.model.set_input_defaults("touchdown_CD", val=0.07344)
        self.prob.model.set_input_defaults("touchdown_CL", val=1.18694)
        self.prob.model.set_input_defaults(
            Mission.Landing.STALL_VELOCITY, val=109.73, units="kn"
        )  # note: EAS in GASP, although at this altitude they are nearly identical
        self.prob.model.set_input_defaults("TAS_touchdown", val=126.27, units="kn")
        self.prob.model.set_input_defaults("thrust_idle", val=1276, units="lbf")
        self.prob.model.set_input_defaults(
            "density_ratio", val=0.998739, units="unitless"
        )  # note: calculated from GASP glide speed values
        self.prob.model.set_input_defaults(
            "wing_loading_land", val=120.61, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults("glide_distance", val=802, units="ft")
        self.prob.model.set_input_defaults("tr_distance", val=167, units="ft")
        self.prob.model.set_input_defaults("delay_distance", val=213, units="ft")
        self.prob.model.set_input_defaults("CL_max", 2.9533, units="unitless")
        self.prob.model.set_input_defaults("mass", 165279, units="lbm")

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=5e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
