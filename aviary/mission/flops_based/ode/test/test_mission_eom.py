import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.flops_based.ode.mission_EOM import MissionEOM
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.variable_info.variables import Dynamic


class MissionEOMTest(unittest.TestCase):
    """
    Test energy-method equations of motion
    """

    def setUp(self):
        self.prob = prob = om.Problem()
        prob.model.add_subsystem(
            "mission", MissionEOM(num_nodes=3), promotes=["*"]
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.MASS, np.array([81796.1389890711, 74616.9849763798, 65193.7423491884]), units="kg"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.DRAG, np.array([9978.32211087097, 8769.90342254821, 7235.03338269778]), units="lbf"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.ALTITUDE_RATE, np.array([29.8463233754212, -5.69941245767868E-09, -4.32644785970493]), units="ft/s"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY_RATE, np.array([0.558739800813549, 3.33665416459715E-17, -0.38372209277242]), units="m/s**2"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, np.array([164.029012458452, 232.775306059091, 117.638805929526]), units="m/s"
        )
        prob.model.set_input_defaults(
            Dynamic.Mission.THRUST_MAX_TOTAL, np.array([40799.6009633346, 11500.32, 42308.2709683461]), units="lbf"
        )
        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        """
        test on mission EOM using data from validation_cases/validation_data/flops_data/full_mission_test_data.py
        """

        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(self.prob.get_val(Dynamic.Mission.ALTITUDE_RATE_MAX, units='ft/min'),
                          np.array([3679.0525544843, 760.55416759, 6557.07891846677]), tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-12)

    def test_IO(self):
        assert_match_varnames(self.prob.model, exclude_outputs={'thrust_required'})


if __name__ == "__main__":
    unittest.main()
