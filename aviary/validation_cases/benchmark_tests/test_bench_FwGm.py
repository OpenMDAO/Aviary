from copy import deepcopy
import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from openmdao.core.problem import _clear_problem_names

from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Mission

mission_distance = 3675

phase_info = {
    'groundroll': {
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'connect_initial_mass': False,
            'fix_initial': True,
            'fix_initial_mass': False,
            'duration_ref': (50., 's'),
            'duration_bounds': ((1., 100.), 's'),
            'velocity_lower': (0, 'kn'),
            'velocity_upper': (1000, 'kn'),
            'velocity_ref': (150, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'ft'),
            'distance_upper': (10.e3, 'ft'),
            'distance_ref': (3000, 'ft'),
            'distance_defect_ref': (3000, 'ft'),
        },
        'initial_guesses': {
            'times': ([0.0, 40.0], 's'),
            'velocity': ([0.066, 143.1], 'kn'),
            'distance': ([0.0, 1000.], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'rotation': {
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'duration_bounds': ((1, 100), 's'),
            'duration_ref': (50.0, 's'),
            'velocity_lower': (0, 'kn'),
            'velocity_upper': (1000, 'kn'),
            'velocity_ref': (100, 'kn'),
            'velocity_ref0': (0, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'ft'),
            'distance_upper': (10_000, 'ft'),
            'distance_ref': (5000, 'ft'),
            'distance_defect_ref': (5000, 'ft'),
            'angle_lower': (0., 'rad'),
            'angle_upper': (5., 'rad'),
            'angle_ref': (5., 'rad'),
            'angle_defect_ref': (5., 'rad'),
            'normal_ref': (10000, 'lbf'),
        },
        'initial_guesses': {
            'times': ([40.0, 5.0], 's'),
            'alpha': ([0.0, 2.5], 'deg'),
            'velocity': ([143, 150.], 'kn'),
            'distance': ([3680.37217765, 4000], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'ascent': {
        'user_options': {
            'num_segments': 4,
            'order': 3,
            'fix_initial': False,
            'velocity_lower': (0, 'kn'),
            'velocity_upper': (700, 'kn'),
            'velocity_ref': (200, 'kn'),
            'velocity_ref0': (0, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'ft'),
            'distance_upper': (15_000, 'ft'),
            'distance_ref': (1e4, 'ft'),
            'distance_defect_ref': (1e4, 'ft'),
            'alt_lower': (0, 'ft'),
            'alt_upper': (700, 'ft'),
            'alt_ref': (1000, 'ft'),
            'alt_defect_ref': (1000, 'ft'),
            'final_altitude': (500, 'ft'),
            'alt_constraint_ref': (500, 'ft'),
            'angle_lower': (-10, 'rad'),
            'angle_upper': (20, 'rad'),
            'angle_ref': (57.2958, 'deg'),
            'angle_defect_ref': (57.2958, 'deg'),
            'pitch_constraint_lower': (0., 'deg'),
            'pitch_constraint_upper': (15., 'deg'),
            'pitch_constraint_ref': (1., 'deg'),
        },
        'initial_guesses': {
            'times': ([45., 25.], 's'),
            'flight_path_angle': ([0.0, 8.], 'deg'),
            'alpha': ([2.5, 1.5], 'deg'),
            'velocity': ([150., 185.], 'kn'),
            'distance': ([4.e3, 10.e3], 'ft'),
            'altitude': ([0.0, 500.], 'ft'),
            'tau_gear': (0.2, 'unitless'),
            'tau_flaps': (0.9, 'unitless'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'accel': {
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'alt': (500, 'ft'),
            'EAS_constraint_eq': (250, 'kn'),
            'duration_bounds': ((1, 200), 's'),
            'duration_ref': (1000, 's'),
            'velocity_lower': (150, 'kn'),
            'velocity_upper': (270, 'kn'),
            'velocity_ref': (250, 'kn'),
            'velocity_ref0': (150, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'NM'),
            'distance_upper': (150, 'NM'),
            'distance_ref': (5, 'NM'),
            'distance_defect_ref': (5, 'NM'),
        },
        'initial_guesses': {
            'times': ([70., 13.], 's'),
            'velocity': ([185., 250.], 'kn'),
            'distance': ([10.e3, 20.e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'climb1': {
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'fix_initial': False,
            'EAS_target': (250, 'kn'),
            'mach_cruise': 0.8,
            'target_mach': False,
            'final_altitude': (10.e3, 'ft'),
            'duration_bounds': ((30, 300), 's'),
            'duration_ref': (1000, 's'),
            'alt_lower': (400, 'ft'),
            'alt_upper': (11_000, 'ft'),
            'alt_ref': (10.e3, 'ft'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'NM'),
            'distance_upper': (500, 'NM'),
            'distance_ref': (10, 'NM'),
            'distance_ref0': (0, 'NM'),
        },
        'initial_guesses': {
            'times': ([1., 2.], 'min'),
            'distance': ([20.e3, 100.e3], 'ft'),
            'altitude': ([500., 10.e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'climb2': {
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'fix_initial': False,
            'EAS_target': (270, 'kn'),
            'mach_cruise': 0.8,
            'target_mach': True,
            'final_altitude': (37.5e3, 'ft'),
            'required_available_climb_rate': (0.1, 'ft/min'),
            'duration_bounds': ((200, 17_000), 's'),
            'duration_ref': (5000, 's'),
            'alt_lower': (9000, 'ft'),
            'alt_upper': (40000, 'ft'),
            'alt_ref': (30000, 'ft'),
            'alt_ref0': (0, 'ft'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (10, 'NM'),
            'distance_upper': (1000, 'NM'),
            'distance_ref': (500, 'NM'),
            'distance_ref0': (0, 'NM'),
            'distance_defect_ref': (500, 'NM'),
        },
        'initial_guesses': {
            'times': ([216., 1300.], 's'),
            'distance': ([100.e3, 200.e3], 'ft'),
            'altitude': ([10.e3, 37.5e3], 'ft'),
            'throttle': ([0.956, 0.956], 'unitless'),
        }
    },
    'cruise': {
        'user_options': {
            'alt_cruise': (37.5e3, 'ft'),
            'mach_cruise': 0.8,
        },
        'initial_guesses': {
            # [Initial mass, delta mass] for special cruise phase.
            'mass': ([171481., -35000], 'lbm'),
            'initial_distance': (200.e3, 'ft'),
            'initial_time': (1516., 's'),
            'altitude': (37.5e3, 'ft'),
            'mach': (0.8, 'unitless'),
        }
    },
    'desc1': {
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'fix_initial': False,
            'input_initial': False,
            'EAS_limit': (350, 'kn'),
            'mach_cruise': 0.8,
            'input_speed_type': SpeedType.MACH,
            'final_altitude': (10.e3, 'ft'),
            'duration_bounds': ((300., 900.), 's'),
            'duration_ref': (1000, 's'),
            'alt_lower': (1000, 'ft'),
            'alt_upper': (40_000, 'ft'),
            'alt_ref': (30_000, 'ft'),
            'alt_ref0': (0, 'ft'),
            'alt_constraint_ref': (10000, 'ft'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (140_000, 'lbm'),
            'mass_ref0': (0, 'lbm'),
            'mass_defect_ref': (140_000, 'lbm'),
            'distance_lower': (3000., 'NM'),
            'distance_upper': (5000., 'NM'),
            'distance_ref': (mission_distance, 'NM'),
            'distance_ref0': (0, 'NM'),
            'distance_defect_ref': (100, 'NM'),
        },
        'initial_guesses': {
            'mass': (136000., 'lbm'),
            'altitude': ([37.5e3, 10.e3], 'ft'),
            'throttle': ([0.0, 0.0], 'unitless'),
            'distance': ([.92*mission_distance, .96*mission_distance], 'NM'),
            'times': ([28000., 500.], 's'),
        }
    },
    'desc2': {
        'user_options': {
            'num_segments': 1,
            'order': 7,
            'fix_initial': False,
            'input_initial': False,
            'EAS_limit': (250, 'kn'),
            'mach_cruise': 0.80,
            'input_speed_type': SpeedType.EAS,
            'final_altitude': (1000, 'ft'),
            'duration_bounds': ((100., 5000), 's'),
            'duration_ref': (500, 's'),
            'alt_lower': (500, 'ft'),
            'alt_upper': (11_000, 'ft'),
            'alt_ref': (10.e3, 'ft'),
            'alt_ref0': (1000, 'ft'),
            'alt_constraint_ref': (1000, 'ft'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (150_000, 'lbm'),
            'mass_defect_ref': (150_000, 'lbm'),
            'distance_lower': (0, 'NM'),
            'distance_upper': (5000, 'NM'),
            'distance_ref': (3500, 'NM'),
            'distance_defect_ref': (100, 'NM'),
        },
        'initial_guesses': {
            'mass': (136000., 'lbm'),
            'altitude': ([10.e3, 1.e3], 'ft'),
            'throttle': ([0., 0.], 'unitless'),
            'distance': ([.96*mission_distance, mission_distance], 'NM'),
            'times': ([28500., 500.], 's'),
        }
    },
}


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer="IPOPT")
    def bench_test_swap_3_FwGm_IPOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary('models/test_aircraft/aircraft_for_bench_FwGm.csv',
                          local_phase_info, optimizer='IPOPT')

        rtol = 1e-2

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS),
                          186418., tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS),
                          104530., tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS),
                          44032., tolerance=rtol)

        assert_near_equal(prob.get_val('landing.' + Mission.Landing.GROUND_DISTANCE),
                          2528., tolerance=rtol)

        assert_near_equal(prob.get_val("traj.desc2.timeseries.distance")[-1],
                          3675.0, tolerance=rtol)

    @require_pyoptsparse(optimizer="SNOPT")
    def bench_test_swap_3_FwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary('models/test_aircraft/aircraft_for_bench_FwGm.csv',
                          local_phase_info, optimizer='SNOPT')

        rtol = 1e-2

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS),
                          186418., tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS),
                          104530., tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS),
                          44032., tolerance=rtol)

        assert_near_equal(prob.get_val('landing.' + Mission.Landing.GROUND_DISTANCE),
                          2528., tolerance=rtol)

        assert_near_equal(prob.get_val("traj.desc2.timeseries.distance")[-1],
                          3675.0, tolerance=rtol)


if __name__ == "__main__":
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.bench_test_swap_3_FwGm_SNOPT()
