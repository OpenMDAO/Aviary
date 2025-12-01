import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class ClimbODETestCase(unittest.TestCase):
    """Test 2-degree of freedom climb ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val('verbosity', Verbosity.QUIET)
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)

        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.sys = self.prob.model = ClimbODE(
            num_nodes=1,
            EAS_target=250,
            mach_cruise=0.8,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
        )

        setup_model_options(self.prob, aviary_options)

    def test_start_of_climb(self):
        # Test against GASP start of climb at 250 kts EAS, check partials
        self.sys.options['EAS_target'] = 250

        self.prob.setup(check=False, force_alloc_complex=True)

        throttle_climb = 0.956
        self.prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, throttle_climb, units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, 1000, units='ft')
        self.prob.set_val(Dynamic.Vehicle.MASS, 174845, units='lbm')
        self.prob.set_val('EAS', 250, units='kn')
        # slightly greater than zero to help check partials
        self.prob.set_val(Aircraft.Wing.INCIDENCE, 0.0000001, units='deg')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            Dynamic.Vehicle.ANGLE_OF_ATTACK: 5.16173398,
            'CL': 0.59745598,
            'CD': 0.02914888,
            Dynamic.Mission.ALTITUDE_RATE: 58.0148345,  # ft/s
            # TAS (kts -> ft/s) * cos(gamma), 253.6827 * 1.68781 *
            # cos(0.13331060446181708)
            Dynamic.Mission.DISTANCE_RATE: 424.21929709,  # ft/s
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -13447.80297433,  # lbm/h
            'theta': 0.22600284,  # rad (12.8021 deg)
            # rad (7.638135 deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: 0.13591359,
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_end_of_climb(self):
        # Test against GASP at 270 kts EAS and at cruise Mach.
        self.sys.options['num_nodes'] = 2
        self.sys.options['EAS_target'] = 270

        self.prob.setup(check=False, force_alloc_complex=True)

        throttle_climb = 0.956
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            np.array([throttle_climb, throttle_climb]),
            units='unitless',
        )
        self.prob.set_val(Dynamic.Mission.ALTITUDE, np.array([11000, 37000]), units='ft')
        self.prob.set_val(Dynamic.Vehicle.MASS, np.array([174149, 171592]), units='lbm')
        self.prob.set_val('EAS', np.array([270, 270]), units='kn')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            Dynamic.Vehicle.ANGLE_OF_ATTACK: [4.05423953, 4.06601219],
            'CL': [0.51248541, 0.61580247],
            'CD': [0.02543166, 0.03141751],
            Dynamic.Mission.ALTITUDE_RATE: [52.44471763, 9.11523198],  # ft/s
            # TAS (kts -> ft/s) * cos(gamma), [319, 459] kts
            # ft/s
            Dynamic.Mission.DISTANCE_RATE: [536.08501758, 774.38047573],
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [
                -11417.86519196,
                -6042.88107957,
            ],
            'theta': [0.16827862, 0.08273576],  # rad ([9.47740, 4.59730] deg),
            # rad, gamma
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [0.09751879, 0.01177046],
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL: [25556.83497662, 10773.48189764],
        }
        check_prob_outputs(self.prob, testvals, 1e-6)


if __name__ == '__main__':
    unittest.main()
    # test = ClimbODETestCase()
    # test.setUp()
    # test.test_end_of_climb()
