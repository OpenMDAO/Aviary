import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.engine_scaling import EngineScaling
from aviary.subsystems.propulsion.utils import EngineModelVariables
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


class EngineScalingTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        nn = 4
        count = 1

        filename = get_path('models/engines/turbofan_28k.csv')

        options = AviaryValues()
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 9)
        # make supersonic scaling factor extremely high so it is obvious if it gets used
        options.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1000)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 1.15)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 0.05)
        options.set_val(Aircraft.Engine.CONSTANT_FUEL_MASS_CONSUMPTION, 10.0, units='lbm/h')
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 0.9)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, True)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')

        engine = EngineDeck(options=options)
        preprocess_propulsion(options, [engine])

        engine_variables = {
            EngineModelVariables.THRUST: 'lbf',
            EngineModelVariables.FUEL_FLOW: 'lbm/h',
            EngineModelVariables.NOX_RATE: 'lbm/h',
        }

        self.prob.model.add_subsystem(
            'engine',
            EngineScaling(num_nodes=nn, engine_variables=engine_variables),
            promotes=['*'],
        )

        setup_model_options(self.prob, options)

        self.prob.setup(force_alloc_complex=True)

        self.prob.set_val('thrust_net_unscaled', np.ones([nn, count]) * 1000, units='lbf')
        self.prob.set_val('fuel_flow_rate_unscaled', np.ones([nn, count]) * 100, units='lbm/h')
        self.prob.set_val('nox_rate_unscaled', np.ones([nn, count]) * 10, units='lbm/h')
        self.prob.set_val(Dynamic.Atmosphere.MACH, np.linspace(0, 0.75, nn), units='unitless')
        self.prob.set_val(
            Aircraft.Engine.SCALE_FACTOR, options.get_val(Aircraft.Engine.SCALE_FACTOR)
        )

        self.prob.run_model()

        expected_values = {
            Dynamic.Vehicle.Propulsion.THRUST: (
                np.array([900.0, 900.0, 900.0, 900.0]),
                'lbf',
            ),
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE: (
                np.array([-1755.55, -1755.55, -1755.55, -1755.55]),
                'lbm/h',
            ),
            Dynamic.Vehicle.Propulsion.NOX_RATE: (
                np.array([9.0, 9.0, 9.0, 9.0]),
                'lbm/h',
            ),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
