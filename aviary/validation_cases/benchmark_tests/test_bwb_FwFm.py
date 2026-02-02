import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Mission

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': True},
    'climb': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise', 'solve_alpha': 'true'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': True,
            'mach_initial': (0.3, 'unitless'),
            'mach_bounds': ((0.3, 0.85), 'unitless'),
            'altitude_optimize': True,
            'altitude_initial': (500.0, 'ft'),
            'altitude_bounds': ((500.0, 35000.0), 'ft'),
            'no_descent': True,
            'mass_ref': (875000, 'lbm'),
            'throttle_enforcement': 'control',
            'time_initial_bounds': ((0.0, 0.0), 'min'),
            'time_duration_bounds': ((24.0, 90.0), 'min'),
        },
        'initial_guesses': {
            'altitude': ([500.5, 35000.0], 'ft'),
            'mach': ([0.3, 0.85], 'unitless'),
        },
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise', 'solve_alpha': 'true'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_initial': (0.85, 'unitless'),
            'mach_bounds': ((0.85, 0.85), 'unitless'),
            'altitude_optimize': False,
            'altitude_initial': (35000.0, 'ft'),
            'altitude_bounds': ((35000.0, 43000.0), 'ft'),
            'mass_ref': (875000, 'lbm'),
            'distance_ref': (7750, 'nmi'),
            'throttle_enforcement': 'control',
            'time_initial_bounds': ((24.0, 180.0), 'min'),
            'time_duration_bounds': ((10.0, 19.0), 'h'),
        },
        'initial_guesses': {
            'altitude': ([35000, 42000.0], 'ft'),
            'mach': ([0.85, 0.85], 'unitless'),
        },
    },
    'descent': {
        'subsystem_options': {'core_aerodynamics': {'method': 'cruise', 'solve_alpha': 'true'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': True,
            'mach_initial': (0.85, 'unitless'),
            'mach_final': (0.2, 'unitless'),
            'mach_bounds': ((0.2, 0.85), 'unitless'),
            'altitude_optimize': True,
            'altitude_initial': (42000.0, 'ft'),
            'altitude_final': (500.0, 'ft'),
            'altitude_bounds': ((500.0, 42000.0), 'ft'),
            'mass_ref': (875000, 'lbm'),
            'distance_ref': (7750, 'nmi'),
            'no_climb': True,
            'throttle_enforcement': 'control',
            'time_initial_bounds': ((10, 19.0), 'h'),
            'time_duration_bounds': ((0.15, 1.0), 'h'),
        },
    },
    'post_mission': {
        'include_landing': False,
        'constrain_range': True,
        'target_range': (7750.0, 'nmi'),
    },
}


# @use_tempdirs
class BWBProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a BWB aircraft using FLOPS mass and aero method
    and HEIGHT_ENERGY mission method. Expected outputs based on
    'models/aircraft/blended_wing_body/bwb_simple_FLOPS.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_bwb_FwFm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/blended_wing_body/bwb_simple_FLOPS.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=1,
            max_iter=60,
        )
        # prob.model.list_vars(units=True, print_arrays=True)
        # prob.list_indep_vars()
        # prob.list_problem_vars()
        # prob.model.list_outputs()

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            139803.667415,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.OPERATING_MASS, units='lbm'),
            79873.05255347,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            26180.61486153,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3500.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2216.0066613,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            116003.31044998,
            tolerance=rtol,
        )


if __name__ == '__main__':
    # unittest.main()
    test = BWBProblemPhaseTestCase()
    test.setUp()
    test.test_bench_bwb_FwFm_SNOPT()
