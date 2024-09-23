import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic


class CruiseODETestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_options))

        self.prob.model = BreguetCruiseODESolution(
            num_nodes=2,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems)

        self.prob.model.set_input_defaults(
            Dynamic.Mission.MACH, np.array([0, 0]), units="unitless"
        )

    def test_cruise(self):
        # test partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Mission.MACH, [0.7, 0.7], units="unitless")

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        tol = tol = 1e-6
        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE], np.array(
                [1.0, 1.0]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE], np.array(
                [0.0, 881.8116]), tol)
        assert_near_equal(
            self.prob["time"], np.array(
                [0, 7906.83]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS], np.array(
                [3.429719,  4.433518]), tol)
        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE_MAX], np.array(
                [-17.63194, -16.62814]), tol)

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
