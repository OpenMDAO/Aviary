import unittest

import openmdao.api as om
from aviary.interface.default_phase_info.two_dof_fiti import create_2dof_based_descent_phases
from aviary.mission.gasp_based.idle_descent_estimation import add_descent_estimation_as_submodel

from openmdao.utils.assert_utils import assert_near_equal

from aviary.interface.default_phase_info.two_dof import default_mission_subsystems
from aviary.mission.gasp_based.idle_descent_estimation import descent_range_and_fuel
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.enums import Verbosity
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.preprocessors import preprocess_propulsion
import importlib


@unittest.skipUnless(importlib.util.find_spec("pyoptsparse") is not None, "pyoptsparse is not installed")
class IdleDescentTestCase(unittest.TestCase):
    def setUp(self):
        input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GwGm.csv'
        aviary_inputs, _ = create_vehicle(input_deck)
        aviary_inputs.set_val('verbosity', Verbosity.QUIET)
        aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units="lbf")
        aviary_inputs.set_val(Dynamic.Mission.THROTTLE, val=0, units="unitless")
        ode_args = dict(aviary_options=aviary_inputs,
                        core_subsystems=default_mission_subsystems)
        engine = EngineDeck(options=aviary_inputs)
        preprocess_propulsion(aviary_inputs, [engine])

        self.ode_args = ode_args
        self.aviary_inputs = aviary_inputs
        self.tol = 1e-5

    def test_case1(self):

        results = descent_range_and_fuel(ode_args=self.ode_args)['refined_guess']

        # Values obtained by running idle_descent_estimation
        assert_near_equal(results['distance_flown'], 91.8911599691433, self.tol)
        assert_near_equal(results['fuel_burned'], 236.73893823639082, self.tol)

    # def test_subproblem(self):
    #     prob = om.Problem()
    #     prob.model = om.Group()

    #     descent_phases = create_2dof_based_descent_phases(
    #         self.ode_args,
    #         cruise_mach=.8)

    #     add_descent_estimation_as_submodel(prob, descent_phases)

    #     prob.setup()
    #     om.n2(prob, 'idle_descent_n2.html', show_browser=False)
    #     prob.run_model()
    #     prob.get_val('descent_range', 'NM')
    #     prob.get_val('descent_fuel', 'lbm')


if __name__ == "__main__":
    unittest.main()
