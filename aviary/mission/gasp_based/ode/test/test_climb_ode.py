import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class ClimbODETestCase(unittest.TestCase):
    """
    Test 2-degree of freedom climb ODE
    """

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
        # Test against GASP start of climb at 250 kts EAS, check partials
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
        self.prob.set_val("interference_independent_of_shielded_area", 1.89927266)
        self.prob.set_val("drag_loss_due_to_shielded_wing_area", 68.02065834)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            "alpha": 5.16398,
            "CL": 0.59766664,
            "CD": 0.03070836,
            Dynamic.Mission.ALTITUDE_RATE: 3414.63 / 60,  # ft/s
            # TAS (kts -> ft/s) * cos(gamma), 253.6827 * 1.68781 * cos(0.13331060446181708)
            Dynamic.Mission.DISTANCE_RATE: 424.36918705874785,  # ft/s
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -13448.29,  # lbm/h
            "theta": 0.22343879616956605,  # rad (12.8021 deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: 0.13331060446181708,  # rad (7.638135 deg)
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        self.prob.setup(check=False, force_alloc_complex=True)
        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_end_of_climb(self):
        # Test against GASP at 270 kts EAS and at cruise Mach.
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
        self.prob.set_val("interference_independent_of_shielded_area", 1.89927266)
        self.prob.set_val("drag_loss_due_to_shielded_wing_area", 68.02065834)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            "alpha": [4.05559, 4.08245],
            "CL": [0.512629, 0.617725],
            "CD": [0.02692764, 0.03311237],
            Dynamic.Mission.ALTITUDE_RATE: [3053.754 / 60, 429.665 / 60],  # ft/s
            # TAS (kts -> ft/s) * cos(gamma), [319, 459] kts
            Dynamic.Mission.DISTANCE_RATE: [536.2835, 774.4118],  # ft/s
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [-11420.05,  -6050.26],
            "theta": [0.16540479, 0.08049912],  # rad ([9.47699, 4.61226] deg),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [0.09462135, 0.00924686],  # rad, gamma
            Dynamic.Mission.THRUST_TOTAL: [25560.51, 10784.25],
        }
        check_prob_outputs(self.prob, testvals, 1e-6)


if __name__ == "__main__":
    unittest.main()
