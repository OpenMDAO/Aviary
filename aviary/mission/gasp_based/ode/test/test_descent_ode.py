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
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class DescentODETestCase(unittest.TestCase):
    """Test 2-degree of freedom descent ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.sys = self.prob.model = DescentODE(
            num_nodes=1,
            mach_cruise=0.8,
            aviary_options=get_option_defaults(),
            core_subsystems=default_mission_subsystems,
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
        self.sys.options['EAS_limit'] = 350

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
            Dynamic.Vehicle.ANGLE_OF_ATTACK: np.array([3.22047, 1.20346]),
            'CL': np.array([0.5169255, 0.25908651]),
            'CD': np.array([0.02786507, 0.01862951]),
            # ft/s
            Dynamic.Mission.ALTITUDE_RATE: np.array([-39.28140894, -47.95697037]),
            # TAS (ft/s) * cos(gamma), [458.67774, 437.62297] kts
            Dynamic.Mission.DISTANCE_RATE: [773.1451, 736.9446],  # ft/s
            # lbm/h
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: np.array(
                [-452.29466667, -997.41373745]
            ),
            'EAS': [418.50757579, 590.73344999],  # ft/s ([247.95894, 349.99997] kts)
            Dynamic.Atmosphere.MACH: [0.8, 0.697125],
            # gamma, rad ([-2.908332, -3.723388] deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: [-0.05076362, -0.06498377],
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            method='cs', out_stream=None, excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_low_alt(self):
        # Test descent below 10k ft
        self.sys.options['input_speed_type'] = SpeedType.EAS
        self.sys.options['EAS_limit'] = 350

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
            Dynamic.Vehicle.ANGLE_OF_ATTACK: 4.19956,
            'CL': 0.507578,
            'CD': 0.0268404,
            Dynamic.Mission.ALTITUDE_RATE: -18.97632876,
            # TAS (ft/s) * cos(gamma) = 255.5613 * 1.68781 * cos(-0.0440083)
            Dynamic.Mission.DISTANCE_RATE: 430.92063193,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: -1295.11818529,
            # rad (-2.52149 deg)
            Dynamic.Mission.FLIGHT_PATH_ANGLE: -0.0440083,
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
