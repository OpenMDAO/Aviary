import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.ascent_ode import AscentODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic


class AscentODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options))

        self.prob.model = AscentODE(num_nodes=2,
                                    aviary_options=aviary_options,
                                    core_subsystems=default_mission_subsystems)

    def test_ascent_partials(self):
        # Test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Mission.VELOCITY, [100, 100], units="kn")
        self.prob.set_val("t_curr", [1, 2], units="s")
        self.prob.set_val("interference_independent_of_shielded_area", 1.89927266)
        self.prob.set_val("drag_loss_due_to_shielded_wing_area", 68.02065834)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array(
                [641174.75, 641174.75]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE], np.array(
                [2260.644, 2260.644]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE], np.array(
                [0.0, 0.0]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE_RATE], np.array(
                [168.781, 168.781]), tol)
        assert_near_equal(
            self.prob["alpha_rate"], np.array(
                [0.0, 0.0]), tol)
        assert_near_equal(
            self.prob["normal_force"], np.array(
                [0.0, 0.0]), tol)
        assert_near_equal(
            self.prob["fuselage_pitch"], np.array(
                [0.0, 0.0]), tol)
        assert_near_equal(
            self.prob["load_factor"], np.array(
                [11850.494, 11850.494]), tol)

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
