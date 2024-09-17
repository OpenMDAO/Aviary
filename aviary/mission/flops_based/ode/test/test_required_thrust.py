import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.flops_based.ode.required_thrust import RequiredThrust
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.variable_info.variables import Dynamic


class RequiredThrustTest(unittest.TestCase):
    """
    Test required thrust
    """

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            "req_thrust", RequiredThrust(num_nodes=2), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DRAG, np.array([47447.13138523, 44343.01567596]), units="N"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.MASS, np.array([106292, 106292]), units="lbm"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.ALTITUDE_RATE, np.array([1.72, 11.91]), units="m/s"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY_RATE, np.array([5.23, 2.7]), units="m/s**2"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, np.array([160.99, 166.68]), units="m/s"
        )

        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):

        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob["thrust_required"],
            np.array([304653.8, 208303.1]), tol
        )

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model, exclude_outputs={'thrust_required'})


if __name__ == "__main__":
    unittest.main()
