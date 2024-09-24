"""
Test file to test the outputs, derivatives, and IO of each sample component/group.
The name of this file needs to start with 'test' so that the testflo command will
find and run the file.
"""

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary import constants
from aviary.mission.flops_based.phases.simplified_takeoff import (
    FinalTakeoffConditions, StallSpeed, TakeoffGroup)
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class StallSpeedTest(unittest.TestCase):
    """
    Test computation in StallSpeed class
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "comp",
            StallSpeed(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults("mass", val=181200.0, units="lbm")  # check
        self.prob.model.set_input_defaults(
            Dynamic.Mission.DENSITY, val=constants.RHO_SEA_LEVEL_METRIC, units="kg/m**3"
        )  # check
        self.prob.model.set_input_defaults(
            "planform_area", val=1370.0, units="ft**2"
        )  # check (this is the reference wing area)
        self.prob.model.set_input_defaults(
            'Cl_max', val=2.0000, units='unitless')  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(self.prob["v_stall"], 71.90002053, tol)  # not actual value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(
            partial_data, atol=1e-12, rtol=1e-12
        )  # check the partial derivatives


class StallSpeedTest2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.flops_based.phases.simplified_takeoff as takeoff
        takeoff.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.flops_based.phases.simplified_takeoff as takeoff
        takeoff.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "comp",
            StallSpeed(),
            promotes=["*"],
        )
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class FinalConditionsTest(unittest.TestCase):
    """
    Test final conditions computation in FinalTakeoffConditions class
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "comp", FinalTakeoffConditions(num_engines=2), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            "v_stall", val=100, units="m/s"
        )  # not actual value
        self.prob.model.set_input_defaults(
            Mission.Summary.GROSS_MASS, val=181200.0, units="lbm"
        )  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.FUEL_SIMPLE, val=577, units="lbm"
        )  # check
        self.prob.model.set_input_defaults(
            Dynamic.Mission.DENSITY,
            val=constants.RHO_SEA_LEVEL_ENGLISH,
            units="slug/ft**3",
        )  # check
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.0, units="ft**2"
        )  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_COEFFICIENT_MAX, val=2.0000, units='unitless')  # check
        self.prob.model.set_input_defaults(
            Mission.Design.THRUST_TAKEOFF_PER_ENG, val=28928.0, units="lbf")  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_OVER_DRAG, val=17.354, units='unitless')  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(
            self.prob[Mission.Takeoff.GROUND_DISTANCE], 6637.65417226, tol
        )  # ft (not actual value)
        # m/s (not actual value)
        assert_near_equal(self.prob[Mission.Takeoff.FINAL_VELOCITY], 123.09, tol)
        assert_near_equal(
            self.prob[Mission.Takeoff.FINAL_MASS], 180623.0, tol
        )  # lbm (not actual value)
        assert_near_equal(
            self.prob[Mission.Takeoff.FINAL_ALTITUDE], 35, tol)  # ft

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class FinalConditionsTest2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.flops_based.phases.simplified_takeoff as takeoff
        takeoff.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.flops_based.phases.simplified_takeoff as takeoff
        takeoff.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", FinalTakeoffConditions(num_engines=2), promotes=["*"]
        )
        # default value v_stall = 0.1 will worsen the output
        prob.model.set_input_defaults("v_stall", val=100, units="m/s")
        # default value GROSS_MASS = 150000 will worsen the output
        prob.model.set_input_defaults(
            Mission.Summary.GROSS_MASS, val=181200.0, units="lbm")
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class TakeoffGroupTest(unittest.TestCase):
    """
    Test computation in TakeoffGroup
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group_example", TakeoffGroup(num_engines=2), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Mission.Summary.GROSS_MASS, val=181200.0, units="lbm"
        )  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.FUEL_SIMPLE, val=577, units="lbm"
        )  # check
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.0, units="ft**2"
        )  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_COEFFICIENT_MAX, val=2.0000, units='unitless')  # check
        self.prob.model.set_input_defaults(
            Mission.Design.THRUST_TAKEOFF_PER_ENG, val=28928.0, units="lbf")  # check
        self.prob.model.set_input_defaults(
            Mission.Takeoff.LIFT_OVER_DRAG, val=17.354, units='unitless')  # check
        self.prob.model.set_input_defaults(
            Dynamic.Mission.ALTITUDE, val=0, units="ft")  # check

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-5

        assert_near_equal(
            self.prob[Mission.Takeoff.GROUND_DISTANCE], 6637.58951504, tol
        )  # ft (not actual value)
        assert_near_equal(
            self.prob[Mission.Takeoff.FINAL_VELOCITY], 88.49655173, tol
        )  # m/s (not actual value)
        assert_near_equal(
            self.prob[Mission.Takeoff.FINAL_MASS], 180623.0, tol
        )  # lbm (not actual value)
        assert_near_equal(
            self.prob[Mission.Takeoff.FINAL_ALTITUDE], 35, tol)  # ft

        partial_data = self.prob.check_partials(
            out_stream=None, excludes=['*.standard_atmosphere'], method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
