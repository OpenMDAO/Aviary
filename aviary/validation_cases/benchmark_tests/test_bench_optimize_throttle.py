import unittest
from copy import deepcopy

import numpy as np

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.api import Mission
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': True},
    'climb': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_initial': (0.2, 'unitless'),
            'mach_final': (0.72, 'unitless'),
            'mach_bounds': ((0.18, 0.74), 'unitless'),
            'altitude_optimize': False,
            'altitude_initial': (0.0, 'ft'),
            'altitude_final': (32000.0, 'ft'),
            'altitude_bounds': ((0.0, 32000.0), 'ft'),
            'altitude_polynomial_order': 3,
            'mass_ref': (2.0e5, 'lbm'),
            'throttle_enforcement': 'control',
            'throttle_optimize': True,
            'time_initial': (0.0, 's'),
            'time_initial': (0.0, 'min'),
            'time_duration_bounds': ((32.0, 128.0), 'min'),
        },
        'initial_guesses': {
            'throttle': ([1.0, 1.0], 'unitless'),
        },
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.72, 'unitless'),
            'mach_bounds': ((0.72, 0.72), 'unitless'),
            'mach_polynomial_order': 3,
            'altitude_optimize': False,
            'altitude_initial': (32000.0, 'ft'),
            'altitude_final': (34000.0, 'ft'),
            'altitude_bounds': ((32000.0, 34000.0), 'ft'),
            'altitude_polynomial_order': 1,
            'mass_ref': (2.0e5, 'lbm'),
            'distance_ref': (1906, 'nmi'),
            'throttle_enforcement': 'control',
            'throttle_optimize': True,
            'throttle_polynomial_order': 1,
            'time_initial_bounds': ((32.0, 128.0), 'min'),
            'time_duration_bounds': ((60.5, 240.0), 'min'),
        },
        'initial_guesses': {
            'throttle': ([0.8, 0.8], 'unitless'),
        },
    },
    'descent': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_initial': (0.72, 'unitless'),
            'mach_final': (0.36, 'unitless'),
            'mach_bounds': ((0.34, 0.74), 'unitless'),
            'altitude_optimize': False,
            'altitude_initial': (34000.0, 'ft'),
            'altitude_final': (500.0, 'ft'),
            'altitude_bounds': ((500.0, 34000.0), 'ft'),
            'altitude_polynomial_order': 3,
            'distance_ref': (1906, 'nmi'),
            'mass_ref': (2.0e5, 'lbm'),
            'throttle_enforcement': 'control',
            'throttle_optimize': True,
            'time_initial_bounds': ((90.0, 361.5), 'min'),
            'time_duration_bounds': ((29.0, 87.0), 'min'),
        },
        'initial_guesses': {
            'throttle': ([0.0, 0.0], 'unitless'),
        },
    },
    'post_mission': {
        'include_landing': True,
        'constrain_range': True,
        'target_range': (1906.0, 'nmi'),
    },
}


@use_tempdirs
class OptimizeThrottleTestCase(unittest.TestCase):
    """Basic test of feature."""

    def setUp(self):
        om.clear_reports()
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='SNOPT')
    def test_throttle_optimize_SNOPT(self):
        prob = AviaryProblem()

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv',
            phase_info,
            verbosity=0,
        )

        prob.aviary_inputs.set_val(Settings.VERBOSITY, 0)

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.add_driver('SNOPT', max_iter=50)
        prob.driver.opt_settings['Major optimality tolerance'] = 1e-4
        prob.driver.opt_settings['Major feasibility tolerance'] = 1e-4

        prob.add_design_variables()
        prob.add_objective(objective_type='fuel_burned')

        prob.setup()

        prob.run_aviary_problem(simulate=False)

        self.assertTrue(prob.result.success)

        gross_mass = prob.get_val(Mission.Summary.GROSS_MASS, units='lbm')
        assert_near_equal(gross_mass, 160506.0, tolerance=1e-3)

        cruise_throttle = prob.get_val('traj.cruise.timeseries.throttle')
        assert_near_equal(cruise_throttle[-1], 0.6925, tolerance=1e-2)


if __name__ == '__main__':
    unittest.main()
