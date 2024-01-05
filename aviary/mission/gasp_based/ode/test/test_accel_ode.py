import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.accel_ode import AccelODE
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.IO_test_util import (assert_match_spec,
                                                  check_prob_outputs,
                                                  skipIfMissingXDSM)
from aviary.variable_info.variables import Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


class AccelerationODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.sys = self.prob.model = AccelODE(num_nodes=2,
                                              aviary_options=get_option_defaults(),
                                              core_subsystems=default_mission_subsystems)

    def test_accel(self):
        """Test both points in GASP Large Single Aisle 1 acceleration segment"""
        self.prob.setup(check=False, force_alloc_complex=True)

        throttle_climb = 0.956
        self.prob.set_val(Dynamic.Mission.ALTITUDE, [500, 500], units="ft")
        self.prob.set_val(
            Dynamic.Mission.THROTTLE, [
                throttle_climb, throttle_climb], units='unitless')
        self.prob.set_val("TAS", [185, 252], units="kn")
        self.prob.set_val(Dynamic.Mission.MASS, [174974, 174878], units="lbm")

        self.prob.run_model()
        testvals = {
            Dynamic.Mission.LIFT: [174974, 174878],
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [
                -13189.24129984, -13490.78047417]  # lbm/h
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-2)

        partial_data = self.prob.check_partials(
            method="cs", out_stream=None, excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, rtol=1e-10)

    @skipIfMissingXDSM('statics_specs/accelerate.json')
    def test_accel_spec(self):
        """Test accel ODE spec"""
        subsystem = self.prob.model
        assert_match_spec(subsystem, "statics_specs/accelerate.json")


if __name__ == "__main__":
    unittest.main()
