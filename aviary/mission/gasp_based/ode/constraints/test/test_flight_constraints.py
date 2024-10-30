import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.gasp_based.ode.constraints.flight_constraints import \
    FlightConstraints
from aviary.variable_info.variables import Aircraft, Dynamic


class FlightConstraintTestCase(unittest.TestCase):
    """
    Test minimum TAS computation
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", FlightConstraints(num_nodes=2), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Dynamic.Mission.MASS, np.array([174878.0, 174878.0]), units="lbm"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units="ft**2")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.DENSITY, 0.0023081 * np.ones(2), units="slug/ft**3"
        )
        self.prob.model.set_input_defaults(
            "CL_max", 1.2596 * np.ones(2), units="unitless")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, 7.76 * np.ones(2), units="deg")
        self.prob.model.set_input_defaults(Aircraft.Wing.INCIDENCE, 0.0, units="deg")
        self.prob.model.set_input_defaults("alpha", 5.19 * np.ones(2), units="deg")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, 252 * np.ones(2), units="kn"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob["theta"], np.array([0.2260201, 0.2260201]), tol
        )  # from GASP
        assert_near_equal(
            self.prob["TAS_violation"], np.array([-99.39848181, -99.39848181]), tol
        )  # note: output value isn't in GASP
        assert_near_equal(
            self.prob["TAS_min"], np.array([325.9296, 325.9296]), tol
        )

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=3e-11, rtol=1e-12)


class FlightConstraintTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.gasp_based.ode.constraints.flight_constraints as constraints
        constraints.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.constraints.flight_constraints as constraints
        constraints.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "group", FlightConstraints(num_nodes=2), promotes=["*"]
        )
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
