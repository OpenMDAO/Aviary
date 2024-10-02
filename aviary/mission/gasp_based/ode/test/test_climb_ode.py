import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


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

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            "alpha": 5.16376881,
            "CL": 0.59764714,
            "CD": 0.03056306,
            Dynamic.Mission.ALTITUDE_RATE: 57.01361283,  # ft/s
            Dynamic.Mission.DISTANCE_RATE: 424.35532272,  # ft/s
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -13448.29,  # lbm/h
            "theta": 0.22367849,  # rad
            Dynamic.Mission.FLIGHT_PATH_ANGLE: 0.13355372,  # rad
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

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            "alpha": [4.05545557, 4.08244122],
            "CL": [0.51261517, 0.61772367],
            "CD": [0.02678719, 0.03296697],
            Dynamic.Mission.ALTITUDE_RATE: [51.04282581, 7.34336078],  # ft/s
            Dynamic.Mission.DISTANCE_RATE: [536.26997036, 774.40958246],  # ft/s
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [-11420.05,  -6050.26],
            "theta": [0.16567639, 0.08073428],  # rad
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [0.09489533, 0.00948224],  # rad, gamma
            Dynamic.Mission.THRUST_TOTAL: [25560.51, 10784.25],
        }
        check_prob_outputs(self.prob, testvals, 1e-6)


if __name__ == "__main__":
    unittest.main()
