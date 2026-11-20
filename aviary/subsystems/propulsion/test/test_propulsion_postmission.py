import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info
from aviary.variable_info.variables import Mission


@use_tempdirs
class PropulsionPreMissionTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_nox_sum(self):
        # Add reserve phases to existing phase_info
        phase_info.update(
            {
                'reserve_climb': {
                    'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                    'user_options': {
                        'reserve': True,
                        'num_segments': 5,
                        'order': 3,
                        'mach_optimize': False,
                        'mach_polynomial_order': 1,
                        'mach_initial': (0.36, 'unitless'),
                        'mach_final': (0.72, 'unitless'),
                        'altitude_optimize': False,
                        'altitude_polynomial_order': 1,
                        'altitude_initial': (500.0, 'ft'),
                        'altitude_final': (32000.0, 'ft'),
                        'throttle_enforcement': 'path_constraint',
                        'time_initial': (0.0, 'min'),
                        'time_duration_bounds': ((30.0, 192.0), 'min'),
                    },
                    'initial_guesses': {
                        'time': ([0, 128], 'min'),
                    },
                },
                'reserve_cruise_fixed_range': {
                    'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                    'user_options': {
                        'reserve': True,
                        # Distance traveled in this phase
                        'target_distance': (300, 'km'),
                        'num_segments': 5,
                        'order': 3,
                        'mach_optimize': False,
                        'mach_polynomial_order': 1,
                        'mach_initial': (0.72, 'unitless'),
                        'mach_final': (0.72, 'unitless'),
                        'altitude_optimize': False,
                        'altitude_polynomial_order': 1,
                        'altitude_initial': (32000.0, 'ft'),
                        'altitude_final': (32000.0, 'ft'),
                        'throttle_enforcement': 'boundary_constraint',
                        'time_initial_bounds': ((149.5, 448.5), 'min'),
                        'time_duration_bounds': ((0, 300), 'min'),
                    },
                    'initial_guesses': {
                        'time': ([30, 120], 'min'),
                    },
                },
                'reserve_descent': {
                    'subsystem_options': {'aerodynamics': {'method': 'computed'}},
                    'user_options': {
                        'reserve': True,
                        'num_segments': 5,
                        'order': 3,
                        'mach_optimize': False,
                        'mach_polynomial_order': 1,
                        'mach_initial': (0.72, 'unitless'),
                        'mach_final': (0.36, 'unitless'),
                        'altitude_optimize': False,
                        'altitude_polynomial_order': 1,
                        'altitude_initial': (32000.0, 'ft'),
                        'altitude_final': (500.0, 'ft'),
                        'throttle_enforcement': 'path_constraint',
                        'time_initial_bounds': ((120.5, 550.0), 'min'),
                        'time_duration_bounds': ((29.0, 87.0), 'min'),
                    },
                },
            }
        )

        prob = AviaryProblem(reports=False)

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_FwFm.csv',
            phase_info,
        )

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('SLSQP')

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.run_aviary_problem()

        expected_values = {
            Mission.RESERVE_NOX_MASS: (32.22202375, 'lbm'),
            Mission.NOX_MASS: (127.18293777, 'lbm'),
            Mission.TOTAL_NOX_MASS: (159.40496153, 'lbm'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=1e-6)


if __name__ == '__main__':
    unittest.main()
