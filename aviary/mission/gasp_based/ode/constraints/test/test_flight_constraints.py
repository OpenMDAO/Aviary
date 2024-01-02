import unittest
import os

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.gasp_based.ode.constraints.flight_constraints import \
    FlightConstraints
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft, Dynamic


class FlightConstraintTestCase(unittest.TestCase):
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
            "rho", 0.0023081 * np.ones(2), units="slug/ft**3"
        )
        self.prob.model.set_input_defaults(
            "CL_max", 1.2596 * np.ones(2), units="unitless")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, 7.76 * np.ones(2), units="deg")
        self.prob.model.set_input_defaults(Aircraft.Wing.INCIDENCE, 0.0, units="deg")
        self.prob.model.set_input_defaults("alpha", 5.19 * np.ones(2), units="deg")
        self.prob.model.set_input_defaults("TAS", 252 * np.ones(2), units="kn")

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

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=3e-11, rtol=1e-12)

    @skipIfMissingXDSM('climb_specs/constraints.json')
    def test_flight_constraints_spec(self):
        subsystem = self.prob.model
        assert_match_spec(subsystem, "climb_specs/constraints.json")


if __name__ == "__main__":
    unittest.main()
