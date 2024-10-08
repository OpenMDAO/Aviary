import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary import constants
from aviary.mission.flops_based.phases.simplified_landing import (LandingCalc,
                                                                  LandingGroup)
from aviary.variable_info.variables import Aircraft, Mission, Dynamic


class LandingCalcTest(unittest.TestCase):
    """
    Test computation in LandingCalc class (the simplified landing)
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "land",
            LandingCalc(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Landing.TOUCHDOWN_MASS, val=152800.0, units="lbm"
        )  # check (this is the design landing mass)
        self.prob.model.set_input_defaults(
            Dynamic.Mission.DENSITY, val=constants.RHO_SEA_LEVEL_METRIC, units="kg/m**3"
        )  # not exact value but should be close enough
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.0, units="ft**2"
        )  # check (this is the reference wing area)
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless')  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(
            self.prob[Mission.Landing.GROUND_DISTANCE], 6403.64963504, tol
        )  # not actual value
        # not actual value
        assert_near_equal(
            self.prob[Mission.Landing.INITIAL_VELOCITY], 136.22914933, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class LandingCalcTest2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.flops_based.phases.simplified_landing as landing
        landing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.flops_based.phases.simplified_landing as landing
        landing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "land",
            LandingCalc(),
            promotes=["*"],
        )
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class LandingGroupTest(unittest.TestCase):
    """
    Test the computation of LandingGroup
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "land",
            LandingGroup(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Landing.TOUCHDOWN_MASS, val=152800.0, units="lbm"
        )  # check (this is the design landing mass)
        self.prob.model.set_input_defaults(
            Mission.Landing.INITIAL_ALTITUDE, val=35, units="ft"
        )  # confirm initial altitude should be 35 ft.
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.0, units="ft**2"
        )  # check (this is the reference wing area)
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=3, units='unitless')  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(
            self.prob[Mission.Landing.GROUND_DISTANCE], 6407.19354429, tol
        )  # not actual value
        # not actual value
        assert_near_equal(
            self.prob[Mission.Landing.INITIAL_VELOCITY], 136.22914933, tol)

        partial_data = self.prob.check_partials(
            out_stream=None, excludes=['*.standard_atmosphere'], method="cs"
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
