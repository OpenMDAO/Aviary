import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.solved_two_dof.ode.groundroll_ode import GroundrollODE
from aviary.mission.two_dof.ode.test.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


@use_tempdirs
class GroundrollODETestCase(unittest.TestCase):
    """Test groundroll ODE."""

    def setUp(self):
        self.prob = om.Problem()

        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for GroundrollODE and its subsystems
        # (GroundrollEOM, LowSpeedAero / AeroGeom / WingTailRatios / Xlifts / SIWB,
        # PropulsionMission / EngineScaling / PropulsionSum).
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, val=Verbosity.BRIEF)
        aviary_options.set_val(Aircraft.Design.TYPE, val='transport', units='unitless')
        aviary_options.set_val(Aircraft.Wing.HAS_STRUT, val=False, units='unitless')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless')
        aviary_options.set_val(Mission.GRAVITY, val=32.2, units='ft/s**2')

        # Engine deck build inputs (hardcoded, not read from _MetaData).
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, val=True)
        aviary_options.set_val(
            Aircraft.Engine.DATA_FILE, val=get_path('models/engines/turbofan_23k_1.csv')
        )
        aviary_options.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST, val=28690, units='lbf')

        # Build the engine deck and vectorize engine variables in aviary_options.
        engine_deck = build_engine_deck(aviary_options)
        preprocess_propulsion(aviary_options, [engine_deck])

        default_mission_subsystems = get_default_mission_subsystems('GASP', [engine_deck])

        self.prob.model = GroundrollODE(
            num_nodes=2,
            aviary_options=aviary_options,
            subsystems=default_mission_subsystems,
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
        self.prob.set_val(Dynamic.Vehicle.MASS, [1.0, 1.0], units='lbm')

        self.prob.run_model()

        testvals = {
            Dynamic.Mission.VELOCITY_RATE: [1415713.83389512, 1415713.83389512],
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: [0.0, 0.0],
            Dynamic.Mission.ALTITUDE_RATE: [0.0, 0.0],
            Dynamic.Mission.DISTANCE_RATE: [168.781, 168.781],
            'normal_force': [0.0, 0.0],
            'fuselage_pitch': [0.0, 0.0],
            'dmass_dv': [-5.02392469e-06, -5.02392469e-06],
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
