import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.interface.default_phase_info.gasp import default_mission_subsystems
from aviary.mission.gasp_based.idle_descent_estimation import descent_range_and_fuel
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.preprocessors import preprocess_propulsion
import importlib


@unittest.skipUnless(importlib.util.find_spec("pyoptsparse") is not None, "pyoptsparse is not installed")
class IdleDescentTestCase(unittest.TestCase):
    def test_case1(self):
        tol = 1e-5
        input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GwGm.csv'
        aviary_inputs, _ = create_vehicle(input_deck)
        aviary_inputs.set_val('debug_mode', False)
        aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units="lbf")
        aviary_inputs.set_val(Dynamic.Mission.THROTTLE, val=0, units="unitless")
        ode_args = dict(aviary_options=aviary_inputs,
                        core_subsystems=default_mission_subsystems)
        engine = EngineDeck(options=aviary_inputs)
        preprocess_propulsion(aviary_inputs, [engine])

        results = descent_range_and_fuel(ode_args=ode_args)['refined_guess']

        # Values obtained by running idle_descent_estimation
        assert_near_equal(results['distance_flown'], 91.8911599691433, tol)
        assert_near_equal(results['fuel_burned'], 236.73893823639082, tol)


if __name__ == "__main__":
    unittest.main()
