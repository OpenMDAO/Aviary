import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class GroundrollODETestCase(unittest.TestCase):
    """Test groundroll ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.prob.model = GroundrollODE(
            num_nodes=2,
            aviary_options=get_option_defaults(),
            core_subsystems=default_mission_subsystems,
        )

        setup_model_options(self.prob, aviary_options)

    def test_groundroll_partials(self):
        # Check partial derivatives
        self.prob.setup(check=False, force_alloc_complex=True)

        set_params_for_unit_tests(self.prob)

        self.prob.set_val(Dynamic.Mission.VELOCITY, [100, 100], units='kn')
        self.prob.set_val('t_curr', [1, 2], units='s')
        self.prob.set_val('aircraft:wing:incidence', 0, units='deg')
        self.prob.set_val('interference_independent_of_shielded_area', 1.89927266)
        self.prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
        self.prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, 1.25)
        self.prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, 1.25)

        self.prob.run_model()

        testvals = {
            Dynamic.Mission.VELOCITY_RATE: [1413275.41170121, 1413275.41170121],
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: [0.0, 0.0],
            Dynamic.Mission.ALTITUDE_RATE: [0.0, 0.0],
            Dynamic.Mission.DISTANCE_RATE: [168.781, 168.781],
            'normal_force': [0.0, 0.0],
            'fuselage_pitch': [0.0, 0.0],
            'dmass_dv': [-5.03258084e-06, -5.03258084e-06],
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
