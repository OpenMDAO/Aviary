import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.flight_path_eom import FlightPathEOM
from aviary.variable_info.variables import Dynamic


class FlightPathEOMTestCase(unittest.TestCase):
    def setUp(self):
        self.ground_roll = False
        self.prob = om.Problem()
        self.fp = self.prob.model.add_subsystem(
            "group", FlightPathEOM(num_nodes=2, ground_roll=self.ground_roll), promotes=["*"]
        )
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        # ground_roll = False (the aircraft is not confined to the ground)

        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array(
                [-27.10027, -27.10027]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE_RATE], np.array(
                [0.5403023, 0.5403023]), tol)
        assert_near_equal(
            self.prob["normal_force"], np.array(
                [-0.0174524, -0.0174524]), tol)
        assert_near_equal(
            self.prob["fuselage_pitch"], np.array(
                [58.2958, 58.2958]), tol)
        assert_near_equal(
            self.prob["load_factor"], np.array(
                [1.883117, 1.883117]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE], np.array(
                [0.841471, 0.841471]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE], np.array(
                [15.36423, 15.36423]), tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_case2(self):
        """
        ground_roll = True (the aircraft is confined to the ground)
        """
        self.fp.options["ground_roll"] = True
        self.prob.setup(force_alloc_complex=True)

        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array(
                [-27.09537, -27.09537]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE_RATE], np.array(
                [0.5403023, 0.5403023]), tol)
        assert_near_equal(
            self.prob["normal_force"], np.array(
                [-0.0, -0.0]), tol)
        assert_near_equal(
            self.prob["fuselage_pitch"], np.array(
                [57.29578, 57.29578]), tol)
        assert_near_equal(
            self.prob["load_factor"], np.array(
                [1.850816, 1.850816]), tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class FlightPathEOMTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.gasp_based.ode.flight_path_eom as fp
        fp.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.flight_path_eom as fp
        fp.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        """
        ground_roll = False (the aircraft is not confined to the ground)
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            "group", FlightPathEOM(num_nodes=2, ground_roll=False), promotes=["*"]
        )
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_case2(self):
        """
        ground_roll = True (the aircraft is confined to the ground)
        """
        prob = om.Problem()
        prob.model.add_subsystem(
            "group", FlightPathEOM(num_nodes=2, ground_roll=True), promotes=["*"]
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.setup(force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
