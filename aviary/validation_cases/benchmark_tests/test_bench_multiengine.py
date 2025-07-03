import unittest
from copy import deepcopy

import numpy as np
import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.multi_engine_single_aisle.multi_engine_single_aisle_data import (
    engine_1_inputs,
    engine_2_inputs,
    inputs,
)
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.variable_info.enums import ThrottleAllocation
from aviary.variable_info.variables import Aircraft

# Build problem
local_phase_info = deepcopy(phase_info)

local_phase_info['climb']['user_options']['mach_optimize'] = False
local_phase_info['climb']['user_options']['mach_polynomial_order'] = 1
local_phase_info['climb']['user_options']['altitude_optimize'] = False
local_phase_info['climb']['user_options']['altitude_polynomial_order'] = 1
local_phase_info['climb']['user_options']['no_descent'] = True

local_phase_info['cruise']['user_options']['mach_optimize'] = False
local_phase_info['cruise']['user_options']['mach_polynomial_order'] = 1
local_phase_info['cruise']['user_options']['altitude_optimize'] = False
local_phase_info['cruise']['user_options']['altitude_polynomial_order'] = 1
local_phase_info['cruise']['user_options']['altitude_bounds'] = ((32000.0, 34000.0), 'ft')
local_phase_info['cruise']['user_options']['throttle_enforcement'] = 'path_constraint'

local_phase_info['descent']['user_options']['mach_optimize'] = False
local_phase_info['descent']['user_options']['mach_polynomial_order'] = 1
local_phase_info['descent']['user_options']['altitude_optimize'] = False
local_phase_info['descent']['user_options']['altitude_polynomial_order'] = 1
local_phase_info['descent']['user_options']['no_climb'] = True

inputs.set_val(Aircraft.Nacelle.LAMINAR_FLOW_LOWER, np.zeros(2))
inputs.set_val(Aircraft.Nacelle.LAMINAR_FLOW_UPPER, np.zeros(2))


@use_tempdirs
class MultiengineTestcase(unittest.TestCase):
    """Test the different throttle allocation methods for models with multiple, unique EngineModels."""

    def setUp(self):
        om.clear_reports()
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='SNOPT')
    def test_multiengine_fixed(self):
        test_phase_info = deepcopy(local_phase_info)
        method = ThrottleAllocation.FIXED

        test_phase_info['climb']['user_options']['throttle_allocation'] = method
        test_phase_info['cruise']['user_options']['throttle_allocation'] = method
        test_phase_info['descent']['user_options']['throttle_allocation'] = method

        engine1 = build_engine_deck(engine_1_inputs)
        engine1.name = 'engine_1'
        engine2 = build_engine_deck(engine_2_inputs)
        engine2.name = 'engine_2'

        prob = AviaryProblem(verbosity=0)

        prob.load_inputs(inputs, test_phase_info, engine_builders=[engine1, engine2])

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.add_driver('SNOPT', max_iter=50, use_coloring=True)

        prob.add_design_variables()
        prob.add_objective()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_aviary_problem('dymos_solution.db', suppress_solver_print=True)

        alloc_climb = prob.get_val('traj.climb.parameter_vals:throttle_allocations')
        alloc_cruise = prob.get_val('traj.cruise.parameter_vals:throttle_allocations')
        alloc_descent = prob.get_val('traj.descent.parameter_vals:throttle_allocations')

        assert_near_equal(alloc_climb[0], 0.5, tolerance=1e-3)
        assert_near_equal(alloc_cruise[0], 0.5, tolerance=1e-3)
        assert_near_equal(alloc_descent[0], 0.5, tolerance=1e-3)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_multiengine_static(self):
        test_phase_info = deepcopy(local_phase_info)
        method = ThrottleAllocation.STATIC

        test_phase_info['climb']['user_options']['throttle_allocation'] = method
        test_phase_info['cruise']['user_options']['throttle_allocation'] = method
        test_phase_info['descent']['user_options']['throttle_allocation'] = method

        engine1 = build_engine_deck(engine_1_inputs)
        engine2 = build_engine_deck(engine_2_inputs)

        prob = AviaryProblem(verbosity=0)

        prob.load_inputs(inputs, test_phase_info, engine_builders=[engine1, engine2])

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.add_driver('SNOPT', max_iter=50, use_coloring=True)

        prob.add_design_variables()
        prob.add_objective()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_aviary_problem('dymos_solution.db', suppress_solver_print=True)

        alloc_climb = prob.get_val('traj.climb.parameter_vals:throttle_allocations')
        alloc_cruise = prob.get_val('traj.cruise.parameter_vals:throttle_allocations')
        alloc_descent = prob.get_val('traj.descent.parameter_vals:throttle_allocations')

        assert_near_equal(alloc_climb[0], 0.5, tolerance=1e-2)
        assert_near_equal(alloc_cruise[0], 0.593, tolerance=1e-2)
        assert_near_equal(alloc_descent[0], 0.408, tolerance=1e-2)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_multiengine_dynamic(self):
        test_phase_info = deepcopy(local_phase_info)
        method = ThrottleAllocation.DYNAMIC

        test_phase_info['climb']['user_options']['throttle_allocation'] = method
        test_phase_info['cruise']['user_options']['throttle_allocation'] = method
        test_phase_info['descent']['user_options']['throttle_allocation'] = method

        prob = AviaryProblem(verbosity=0)

        engine1 = build_engine_deck(engine_1_inputs)
        engine2 = build_engine_deck(engine_2_inputs)

        prob.load_inputs(inputs, test_phase_info, engine_builders=[engine1, engine2])

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.add_driver('SNOPT', max_iter=50, use_coloring=True)

        prob.add_design_variables()
        prob.add_objective()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_aviary_problem('dymos_solution.db', suppress_solver_print=True)

        alloc_climb = prob.get_val('traj.climb.controls:throttle_allocations')
        alloc_cruise = prob.get_val('traj.cruise.controls:throttle_allocations')
        alloc_descent = prob.get_val('traj.descent.controls:throttle_allocations')

        # Cruise is pretty constant, check exact value.
        assert_near_equal(alloc_cruise[0], 0.595, tolerance=1e-2)

        # Check general trend: favors engine 1.
        self.assertGreater(alloc_climb[2], 0.55)
        self.assertGreater(alloc_descent[3], 0.65)


if __name__ == '__main__':
    unittest.main()
