import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from packaging import version

from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic


class DescentODETestCase(unittest.TestCase):
    """
    Test 2-degree of freedom descent ODE
    """

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options))

        self.sys = self.prob.model = DescentODE(num_nodes=1,
                                                mach_cruise=0.8,
                                                aviary_options=get_option_defaults(),
                                                core_subsystems=default_mission_subsystems)

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"), "Skipping due to OpenMDAO version being too low (<3.26)")
    def test_high_alt(self):
        # Test descent above 10k ft with Mach under and over the EAS limit
        self.sys.options["num_nodes"] = 2
        self.sys.options["input_speed_type"] = SpeedType.MACH
        self.sys.options["EAS_limit"] = 350

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(
            Dynamic.Mission.THROTTLE, np.array([
                0, 0]), units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, np.array([36500, 14500]), units="ft")
        self.prob.set_val(Dynamic.Mission.MASS, np.array([147661, 147572]), units="lbm")

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            "alpha": np.array([3.23388, 1.203234]),
            "CL": np.array([0.51849367, 0.25908653]),
            "CD": np.array([0.02794324, 0.01862946]),
            # ft/s
            Dynamic.Mission.ALTITUDE_RATE: np.array([-2356.7705, -2877.9606]) / 60,
            # TAS (ft/s) * cos(gamma), [458.67774, 437.62297] kts
            Dynamic.Mission.DISTANCE_RATE: [773.1637, 737.0653],  # ft/s
            # lbm/h
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: np.array([-451.0239, -997.1514]),
            "EAS": [417.87419406, 590.73344937],  # ft/s ([247.58367, 349.99997] kts)
            Dynamic.Mission.MACH: [0.8, 0.697266],
            # gamma, rad ([-2.908332, -3.723388] deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [-0.05075997, -0.06498538],
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            method="cs", out_stream=None, excludes=["*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_low_alt(self):
        # Test descent below 10k ft
        self.sys.options["input_speed_type"] = SpeedType.EAS
        self.sys.options["EAS_limit"] = 350

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Mission.THROTTLE, 0, units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, 1500, units="ft")
        self.prob.set_val(Dynamic.Mission.MASS, 147410, units="lbm")
        self.prob.set_val("EAS", 250, units="kn")

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            "alpha": 4.19956,
            "CL": 0.507578,
            "CD": 0.0268404,
            Dynamic.Mission.ALTITUDE_RATE: -1138.583 / 60,
            # TAS (ft/s) * cos(gamma) = 255.5613 * 1.68781 * cos(-0.0440083)
            Dynamic.Mission.DISTANCE_RATE: 430.9213,
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -1295.11,
            Dynamic.Mission.FLIGHT_PATH_ANGLE: -0.0440083,  # rad (-2.52149 deg)
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
