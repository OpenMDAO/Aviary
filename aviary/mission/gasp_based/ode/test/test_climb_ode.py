import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.constants import KNOT_TO_FT_PER_SEC
from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems


class ClimbODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options))

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
            "alpha": 5.16398,
            "CL": 0.59766664,
            "CD": 0.03070836,
            Dynamic.Mission.ALTITUDE_RATE: 3414.63 / 60,
            # TAS (kts -> ft/s) * cos(gamma)
            Dynamic.Mission.DISTANCE_RATE: (253.6827 * KNOT_TO_FT_PER_SEC) * np.cos(np.deg2rad(7.638135)),
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -13448.29,  # lbm/h
            "theta": np.deg2rad(12.8021),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: np.deg2rad(7.638135),
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        self.prob.setup(check=False, force_alloc_complex=True)
        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*params*", "*aero*"]
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
            "alpha": [4.05559, 4.08245],
            "CL": [0.512629, 0.617725],
            "CD": [0.02692764, 0.03311237],
            Dynamic.Mission.ALTITUDE_RATE: [3053.754 / 60, 429.665 / 60],
            # TAS (kts -> ft/s) * cos(gamma)
            Dynamic.Mission.DISTANCE_RATE: [
                (319.167 * KNOT_TO_FT_PER_SEC) * np.cos(np.deg2rad(5.42140)),
                (458.846 * KNOT_TO_FT_PER_SEC) * np.cos(np.deg2rad(0.52981)),
            ],
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [-11420.05,  -6050.26],
            "theta": np.deg2rad([9.476996378247684, 4.61226]),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: np.deg2rad([5.42140, 0.52981]),
            Dynamic.Mission.THRUST_TOTAL: [25560.51, 10784.25],
        }
        check_prob_outputs(self.prob, testvals, 1e-6)


if __name__ == "__main__":
    unittest.main()
