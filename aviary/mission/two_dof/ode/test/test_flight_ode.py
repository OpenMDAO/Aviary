import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from packaging import version

from aviary.mission.two_dof.ode.flight_ode import FlightODE
from aviary.mission.two_dof.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.enums import SpeedType, Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class FlightODEClimbTestCase(unittest.TestCase):
    """Test 2-degree of freedom climb ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val('verbosity', Verbosity.QUIET)
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)

        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.sys = self.prob.model = FlightODE(
            num_nodes=1,
            EAS_target=250,
            mach_target=0.8,
            aviary_options=aviary_options,
            subsystems=default_mission_subsystems,
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
            Dynamic.Vehicle.ANGLE_OF_ATTACK: 5.0569239,
            'CL': 0.58762109,
            'CD': 0.02867741,
            Dynamic.Mission.ALTITUDE_RATE: 58.3497354,  # ft/s
            # TAS (kts -> ft/s) * cos(gamma), 253.6827 * 1.68781 *
            # cos(0.13331060446181708)
            Dynamic.Mission.DISTANCE_RATE: 424.19921863,  # ft/s
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -13447.98081484,  # lbm/h
            'theta': 0.22495483,  # rad (12.8021 deg)
            # rad (7.638135 deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: 0.13669486,
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
            Dynamic.Vehicle.ANGLE_OF_ATTACK: [3.99983993, 4.04378041],
            'CL': [0.50719248, 0.61320271],
            'CD': [0.02520388, 0.0312557],
            Dynamic.Mission.ALTITUDE_RATE: [52.68288688, 9.32639661],  # ft/s
            # TAS (kts -> ft/s) * cos(gamma), [319, 459] kts
            # ft/s
            Dynamic.Mission.DISTANCE_RATE: [536.0936254, 774.32986512],
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [
                -11418.00064615,
                -6042.88107957,
            ],
            'theta': [0.16776765, 0.08262117],  # rad ([9.47740, 4.59730] deg),
            # rad, gamma
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [0.09795727, 0.01204389],
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL: [25555.79617743, 10773.48189764],
        }
        check_prob_outputs(self.prob, testvals, 1e-6)


class FlightODEDescenTestCase(unittest.TestCase):
    """Test 2-degree of freedom descent ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.sys = self.prob.model = FlightODE(
            num_nodes=1,
            mach_target=0.8,
            aviary_options=get_option_defaults(),
            subsystems=default_mission_subsystems,
        )

        setup_model_options(self.prob, aviary_options)

    @unittest.skipIf(
        version.parse(openmdao.__version__) < version.parse('3.26'),
        'Skipping due to OpenMDAO version being too low (<3.26)',
    )
    def test_high_alt(self):
        # Test descent above 10k ft with Mach under and over the EAS limit
        self.sys.options['num_nodes'] = 2
        self.sys.options['input_speed_type'] = SpeedType.MACH
        self.sys.options['EAS_target'] = 350

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.array([0, 0]), units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, np.array([36500, 14500]), units='ft')
        self.prob.set_val(Dynamic.Vehicle.MASS, np.array([147661, 147572]), units='lbm')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            Dynamic.Vehicle.ANGLE_OF_ATTACK: np.array([3.21974886, 1.20407839]),
            'CL': np.array([0.51684124, 0.25916936]),
            'CD': np.array([0.02633437, 0.01729238]),
            # ft/s
            Dynamic.Mission.ALTITUDE_RATE: np.array([-37.03297068, -44.19020778]),
            # TAS (ft/s) * cos(gamma), [458.67774, 437.62297] kts
            Dynamic.Mission.DISTANCE_RATE: [773.50001989, 737.22403068],  # ft/s
            # lbm/h
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: np.array(
                [-452.29666667, -997.48350936]
            ),
            'EAS': [418.57187298, 590.73344999],  # ft/s ([247.95894, 349.99997] kts)
            Dynamic.Atmosphere.MACH: [0.8, 0.69721946],
            # gamma, rad ([-2.908332, -3.723388] deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [-0.04784061, -0.05986972],
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            method='cs', out_stream=None, excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_low_alt(self):
        # Test descent below 10k ft
        self.sys.options['input_speed_type'] = SpeedType.EAS
        self.sys.options['EAS_target'] = 350

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, 0, units='unitless')
        self.prob.set_val(Dynamic.Mission.ALTITUDE, 1500, units='ft')
        self.prob.set_val(Dynamic.Vehicle.MASS, 147410, units='lbm')
        self.prob.set_val('EAS', 250, units='kn')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()

        testvals = {
            Dynamic.Vehicle.ANGLE_OF_ATTACK: 4.2001692,
            'CL': 0.50764259,
            'CD': 0.02532094,
            Dynamic.Mission.ALTITUDE_RATE: -17.6942839,
            # TAS (ft/s) * cos(gamma) = 255.5613 * 1.68781 * cos(-0.0440083)
            Dynamic.Mission.DISTANCE_RATE: 431.0014619,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -1295.1511839,
            # rad (-2.52149 deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: -0.04103086,
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
    # test = DescentODETestCase()
    # test.setUp()
    # test.test_high_alt()
