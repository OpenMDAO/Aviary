import unittest
import os

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.utils.test_utils.IO_test_util import (assert_match_spec,
                                                  check_prob_outputs,
                                                  skipIfMissingXDSM)
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


class ClimbODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.sys = self.prob.model = ClimbODE(
            num_nodes=1,
            EAS_target=250,
            mach_cruise=0.8,
            aviary_options=get_option_defaults(),
            core_subsystems=default_mission_subsystems
        )

    def test_start_of_climb(self):
        """Test against GASP start of climb at 250 kts EAS, check partials"""
        self.sys.options["EAS_target"] = 250

        self.prob.setup(check=False, force_alloc_complex=True)

        throttle_climb = 0.956
        self.prob.set_val(
            Dynamic.Mission.THROTTLE, throttle_climb, units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, 1000, units="ft")
        self.prob.set_val(Dynamic.Mission.MASS, 174845, units="lbm")
        self.prob.set_val("EAS", 250, units="kn")
        # slightly greater than zero to help check partials
        self.prob.set_val(Aircraft.Wing.INCIDENCE, 0.0000001, units="deg")

        self.prob.run_model()

        testvals = {
            "alpha": 5.19,
            "CL": 0.5975,
            "CD": 0.0307,
            Dynamic.Mission.ALTITUDE_RATE: 3186 / 60,
            # TAS (kts -> ft/s) * cos(gamma)
            "distance_rate": (254 * 1.68781) * np.cos(np.deg2rad(7.12)),
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -13505,  # lbm/h
            "theta": np.deg2rad(12.31),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: np.deg2rad(7.12),
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-1)  # TODO tighten

        self.prob.setup(check=False, force_alloc_complex=True)
        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_end_of_climb(self):
        """Test against GASP at 270 kts EAS and at cruise Mach."""
        self.sys.options["num_nodes"] = 2
        self.sys.options["EAS_target"] = 270

        self.prob.setup(check=False, force_alloc_complex=True)

        throttle_climb = 0.956
        self.prob.set_val(
            Dynamic.Mission.THROTTLE, np.array([
                throttle_climb, throttle_climb]), units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, np.array([11000, 37000]), units="ft")
        self.prob.set_val(Dynamic.Mission.MASS, np.array([174149, 171592]), units="lbm")
        self.prob.set_val("EAS", np.array([270, 270]), units="kn")

        self.prob.run_model()

        testvals = {
            "alpha": [4.08, 4.05],
            "CL": [0.5119, 0.6113],
            "CD": [0.0270, 0.0326],
            Dynamic.Mission.ALTITUDE_RATE: [3054 / 60, 453 / 60],
            # TAS (kts -> ft/s) * cos(gamma)
            "distance_rate": [
                (319 * 1.68781) * np.cos(np.deg2rad(5.42)),
                (459 * 1.68781) * np.cos(np.deg2rad(0.56)),
            ],
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [-11422, -6039],
            "theta": np.deg2rad([9.5, 4.61]),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: np.deg2rad([5.42, 0.56]),
            Dynamic.Mission.THRUST_TOTAL: [25610, 10790],
        }
        check_prob_outputs(self.prob, testvals, 1e-1)  # TODO tighten

    @skipIfMissingXDSM('statics_specs/climb1.json')
    def test_climb1_spec(self):
        """Test climb1 phase spec"""
        assert_match_spec(self.sys, "statics_specs/climb1.json")

    @skipIfMissingXDSM('statics_specs/climb2.json')
    def test_climb2_spec(self):
        """Test climb2 phase spec"""
        assert_match_spec(self.sys, "statics_specs/climb2.json")


if __name__ == "__main__":
    unittest.main()
