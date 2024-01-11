import unittest
import os

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from packaging import version

from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.IO_test_util import (assert_match_spec,
                                                  check_prob_outputs,
                                                  skipIfMissingXDSM)
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


class DescentODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.sys = self.prob.model = DescentODE(num_nodes=1,
                                                mach_cruise=0.8,
                                                aviary_options=get_option_defaults(),
                                                core_subsystems=default_mission_subsystems)

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"), "Skipping due to OpenMDAO version being too low (<3.26)")
    def test_high_alt(self):
        """Test descent above 10k ft with Mach under and over the EAS limit"""
        self.sys.options["num_nodes"] = 2
        self.sys.options["input_speed_type"] = SpeedType.MACH
        self.sys.options["EAS_limit"] = 350

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(
            Dynamic.Mission.THROTTLE, np.array([
                0, 0]), units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, np.array([36500, 14500]), units="ft")
        self.prob.set_val(Dynamic.Mission.MASS, np.array([147661, 147572]), units="lbm")

        self.prob.run_model()

        testvals = {
            "alpha": np.array([3.2, 1.21]),
            "CL": np.array([0.5123, 0.2583]),
            "CD": np.array([0.0279, 0.0197]),
            Dynamic.Mission.ALTITUDE_RATE: np.array([-2385, -3076]) / 60,
            # TAS (ft/s) * cos(gamma)
            "distance_rate": [
                (459 * 1.68781) * np.cos(np.deg2rad(-2.94)),
                (437 * 1.68781) * np.cos(np.deg2rad(-3.98)),
            ],
            # lbm/h
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: np.array([-452, -996]),
            "EAS": np.array([249, 350]) * 1.68781,  # kts -> ft/s
            Dynamic.Mission.MACH: np.array([0.8, 0.696]),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: np.deg2rad([-2.94, -3.98]),
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-1)  # TODO tighten

        partial_data = self.prob.check_partials(
            method="cs", out_stream=None, excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_low_alt(self):
        """Test descent below 10k ft"""
        self.sys.options["input_speed_type"] = SpeedType.EAS
        self.sys.options["EAS_limit"] = 350

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Mission.THROTTLE, 0, units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, 1500, units="ft")
        self.prob.set_val(Dynamic.Mission.MASS, 147410, units="lbm")
        self.prob.set_val("EAS", 250, units="kn")

        self.prob.run_model()

        testvals = {
            "alpha": 4.21,
            "CL": 0.5063,
            "CD": 0.0271,
            Dynamic.Mission.ALTITUDE_RATE: -1158 / 60,
            # TAS (ft/s) * cos(gamma)
            "distance_rate": (255 * 1.68781) * np.cos(np.deg2rad(-2.56)),
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -1294,
            Dynamic.Mission.FLIGHT_PATH_ANGLE: np.deg2rad(-2.56),
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-1)  # TODO tighten

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('statics_specs/descent1.json')
    def test_descent1_ode_spec(self):
        """Test descent1 phase spec"""
        self.sys.options["input_speed_type"] = SpeedType.MACH
        self.prob.setup()
        subsystem = self.prob.model
        assert_match_spec(subsystem, "statics_specs/descent1.json")

    @skipIfMissingXDSM('statics_specs/descent2.json')
    def test_descent2_ode_spec(self):
        """Test descent2 phase spec"""
        self.sys.options["input_speed_type"] = SpeedType.EAS
        self.prob.setup()
        subsystem = self.prob.model
        assert_match_spec(subsystem, "statics_specs/descent2.json")


if __name__ == "__main__":
    unittest.main()
