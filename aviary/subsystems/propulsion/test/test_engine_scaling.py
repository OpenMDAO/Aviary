import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.engine_scaling import EngineScaling
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class EngineScalingTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem(model=om.Group())

    def test_case(self):
        nn = 4
        count = 1

        filename = 'models/engines/turbofan_28k.deck'
        filename = get_path(filename)

        options = AviaryValues()
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 0.9)
        # make supersonic scaling factor extremely high so it is obvious if it gets used
        options.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 100)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 1.15)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.05)
        options.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 10.0, units='lbm/h')
        options.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 0.9)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')

        # engine1 uses all scaling factors
        engine1 = EngineDeck(options=options)

        preprocess_propulsion(options, [engine1])

        options.set_val(Mission.Summary.FUEL_FLOW_SCALER, 10.)

        self.prob.model.add_subsystem('engine', EngineScaling(
            num_nodes=nn,
            aviary_options=options),
            promotes=['*'])
        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val('thrust_net_unscaled', np.ones(
            [nn, count]) * 1000, units='lbf')
        self.prob.set_val('fuel_flow_rate_unscaled', np.ones(
            [nn, count]) * 100, units='lbm/h')
        self.prob.set_val('nox_rate_unscaled', np.ones([nn, count]) * 10, units='lbm/h')
        self.prob.set_val(Dynamic.Mission.MACH, np.linspace(
            0, 0.75, nn), units='unitless')
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR,
                          options.get_val(Aircraft.Engine.SCALE_FACTOR))

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Mission.THRUST)
        fuel_flow = self.prob.get_val(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE)
        nox_rate = self.prob.get_val(Dynamic.Mission.NOX_RATE)
        # exit_area = self.prob.get_val(Dynamic.Mission.EXIT_AREA)

        thrust_expected = np.array([900., 900., 900., 900])

        fuel_flow_expected = np.array([-1836.55, -1836.55, -1836.55, -1836.55])

        nox_rate_expected = np.array([9., 9., 9., 9])

        assert_near_equal(thrust, thrust_expected, tolerance=1e-10)
        assert_near_equal(fuel_flow, fuel_flow_expected, tolerance=1e-10)
        assert_near_equal(nox_rate, nox_rate_expected, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
