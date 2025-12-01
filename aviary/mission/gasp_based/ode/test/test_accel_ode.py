import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.accel_ode import AccelODE
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class AccelerationODETestCase(unittest.TestCase):
    """Test 2-degree of freedom acceleration ODE."""

    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        self.sys = self.prob.model = AccelODE(
            num_nodes=2, aviary_options=aviary_options, core_subsystems=default_mission_subsystems
        )

    def test_accel(self):
        # Test both points in GASP Large Single Aisle 1 acceleration segment
        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

        throttle_climb = 0.956
        self.prob.set_val(Dynamic.Mission.ALTITUDE, [500, 500], units='ft')
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            [throttle_climb, throttle_climb],
            units='unitless',
        )
        self.prob.set_val(Dynamic.Mission.VELOCITY, [185, 252], units='kn')
        self.prob.set_val(Dynamic.Vehicle.MASS, [174974, 174878], units='lbm')

        set_params_for_unit_tests(self.prob)

        self.prob.run_model()
        testvals = {
            Dynamic.Vehicle.LIFT: [174974, 174878],
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL: [
                -13264.82336817,
                -13567.23449581,
            ],  # lbm/h
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-6)

        partial_data = self.prob.check_partials(
            method='cs', out_stream=None, excludes=['*params*', '*aero*']
        )
        assert_check_partials(partial_data, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
