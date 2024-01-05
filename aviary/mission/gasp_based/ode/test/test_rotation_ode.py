import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.rotation_ode import RotationODE
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


class RotationODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model = RotationODE(num_nodes=2,
                                      aviary_options=get_option_defaults(),
                                      core_subsystems=default_mission_subsystems)

    def test_rotation_partials(self):
        """Check partial derivatives"""
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Aircraft.Wing.INCIDENCE, 1.5, units="deg")
        self.prob.set_val(Dynamic.Mission.MASS, [100000, 100000], units="lbm")
        self.prob.set_val("alpha", [1.5, 1.5], units="deg")
        self.prob.set_val("TAS", [100, 100], units="kn")
        self.prob.set_val("t_curr", [1, 2], units="s")

        self.prob.run_model()

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('statics_specs/rotation.json')
    def test_rotation_spec(self):
        """Test rotation ODE spec"""
        subsystem = self.prob.model
        assert_match_spec(subsystem, "statics_specs/rotation.json")


if __name__ == "__main__":
    unittest.main()
