import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.gasp_based.ode.constraints.speed_constraints import \
    SpeedConstraints
from aviary.variable_info.variables import Dynamic


class SpeedConstraintTestCase1(unittest.TestCase):
    """
    Test speed constraint at MACH = 0.6 with targeted MACH at 0.8
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            SpeedConstraints(num_nodes=3, EAS_target=229, mach_cruise=0.8),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults("EAS", np.array([229, 229, 229]), units="kn")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.MACH, np.array([0.6, 0.6, 0.6]), units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob["speed_constraint"],
            np.array([[0, -45.8], [0, -45.8], [0, -45.8]]),
            tol,
        )

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class SpeedConstraintTestCase2(unittest.TestCase):
    """
    Test speed constraint at MACH = 0.9 with targeted Mach at 0.8
    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            SpeedConstraints(num_nodes=3, EAS_target=229, mach_cruise=0.8),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults("EAS", np.array([229, 229, 229]), units="kn")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.MACH, np.array([0.9, 0.9, 0.9]), units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob["speed_constraint"],
            np.array([[0, 22.9], [0, 22.9], [0, 22.9]]),
            tol,
        )

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
