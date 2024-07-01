import unittest
import warnings
import importlib

import openmdao.api as om
from aviary.interface.default_phase_info.two_dof_fiti_deprecated import create_2dof_based_descent_phases
from aviary.interface.default_phase_info.two_dof_fiti import descent_phases, add_default_sgm_args

from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.mission.gasp_based.idle_descent_estimation import descent_range_and_fuel, add_descent_estimation_as_submodel
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.variable_info.variables import Aircraft, Dynamic, Settings
from aviary.variable_info.enums import Verbosity
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.preprocessors import preprocess_propulsion


@unittest.skipUnless(importlib.util.find_spec("pyoptsparse") is not None, "pyoptsparse is not installed")
class IdleDescentTestCase(unittest.TestCase):
    def setUp(self):
        input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GwGm.csv'
        aviary_inputs, _ = create_vehicle(input_deck)
        aviary_inputs.set_val(Settings.VERBOSITY, Verbosity.QUIET)
        aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units="lbf")
        aviary_inputs.set_val(Dynamic.Mission.THROTTLE, val=0, units="unitless")

        engine = build_engine_deck(aviary_options=aviary_inputs)
        preprocess_propulsion(aviary_inputs, engine)

        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(aviary_inputs))

        ode_args = dict(aviary_options=aviary_inputs,
                        core_subsystems=default_mission_subsystems)

        self.ode_args = ode_args
        self.aviary_inputs = aviary_inputs
        self.tol = 1e-5

        add_default_sgm_args(descent_phases, self.ode_args)
        self.phases = descent_phases

    def test_case1(self):

        results = descent_range_and_fuel(phases=self.phases)['refined_guess']

        # Values obtained by running idle_descent_estimation
        assert_near_equal(results['distance_flown'], 91.8911599691433, self.tol)
        assert_near_equal(results['fuel_burned'], 236.73893823639082, self.tol)

    def test_subproblem(self):
        prob = om.Problem()
        prob.model = om.Group()

        ivc = om.IndepVarComp()
        ivc.add_output(Aircraft.Design.OPERATING_MASS, 97500, units='lbm')
        ivc.add_output(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 36000, units='lbm')
        prob.model.add_subsystem('IVC', ivc, promotes=['*'])

        add_descent_estimation_as_submodel(
            prob,
            phases=self.phases,
            ode_args=self.ode_args,
            cruise_alt=35000,
            reserve_fuel=4500,
        )

        prob.setup()

        warnings.filterwarnings('ignore', category=UserWarning)
        prob.run_model()
        warnings.filterwarnings('default', category=UserWarning)

        # Values obtained by running idle_descent_estimation
        assert_near_equal(prob.get_val('descent_range', 'NM'), 98.38026813, self.tol)
        assert_near_equal(prob.get_val('descent_fuel', 'lbm'), 250.84809336, self.tol)


if __name__ == "__main__":
    unittest.main()
