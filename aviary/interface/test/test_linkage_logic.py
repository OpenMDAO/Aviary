import copy
import unittest

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem


@use_tempdirs
class AircraftMissionTestSuite(unittest.TestCase):
    def setUp(self):
        cruise_dict = {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'num_segments': 5,
                'order': 3,
                'mach_optimize': False,
                'mach_initial': (0.72, 'unitless'),
                'mach_final': (0.72, 'unitless'),
                'mach_bounds': ((0.7, 0.74), 'unitless'),
                'altitude_optimize': False,
                'altitude_initial': (32000.0, 'ft'),
                'altitude_final': (34000.0, 'ft'),
                'altitude_bounds': ((23000.0, 38000.0), 'ft'),
                'throttle_enforcement': 'boundary_constraint',
                'time_initial_bounds': ((64.0, 192.0), 'min'),
                'time_duration_bounds': ((56.5, 169.5), 'min'),
            },
        }
        cruise_dicts = [copy.deepcopy(cruise_dict) for _ in range(5)]
        for i, cruise_dict in enumerate(cruise_dicts):
            cruise_dict['user_options']['time_initial_bounds'] = (
                (64.0 + i * 10, 192.0 + i * 10),
                'min',
            )
            cruise_dict['user_options']['time_duration_bounds'] = (
                (56.5 + i * 10, 169.5 + i * 10),
                'min',
            )
        cruise_dicts[0]['user_options']['mach_optimize'] = True
        cruise_dicts[1]['user_options']['mach_optimize'] = True
        cruise_dicts[4]['user_options']['mach_optimize'] = True
        cruise_dicts[2]['user_options']['altitude_optimize'] = True
        cruise_dicts[3]['user_options']['altitude_optimize'] = True

        # Create the phase_info
        self.phase_info = {
            'pre_mission': {'include_takeoff': False, 'optimize_mass': True},
            'climb': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 5,
                    'order': 3,
                    'mach_optimize': False,
                    'mach_polynomial_order': 1,
                    'mach_initial': (0.2, 'unitless'),
                    'mach_final': (0.72, 'unitless'),
                    'mach_bounds': ((0.18, 0.74), 'unitless'),
                    'altitude_optimize': False,
                    'altitude_polynomial_order': 1,
                    'altitude_initial': (0.0, 'ft'),
                    'altitude_final': (32000.0, 'ft'),
                    'altitude_bounds': ((0.0, 34000.0), 'ft'),
                    'throttle_enforcement': 'path_constraint',
                    'time_initial_bounds': ((0.0, 0.0), 'min'),
                    'time_duration_bounds': ((64.0, 192.0), 'min'),
                },
            },
            'cruise0': cruise_dicts[0],
            'cruise1': cruise_dicts[1],
            'cruise2': cruise_dicts[2],
            'cruise3': cruise_dicts[3],
            'cruise4': cruise_dicts[4],
            'descent': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 5,
                    'order': 3,
                    'mach_optimize': False,
                    'mach_polynomial_order': 1,
                    'mach_initial': (0.72, 'unitless'),
                    'mach_final': (0.36, 'unitless'),
                    'mach_bounds': ((0.34, 0.74), 'unitless'),
                    'altitude_optimize': False,
                    'altitude_polynomial_order': 1,
                    'altitude_initial': (34000.0, 'ft'),
                    'altitude_final': (500.0, 'ft'),
                    'altitude_bounds': ((0.0, 38000.0), 'ft'),
                    'throttle_enforcement': 'path_constraint',
                    'time_initial_bounds': ((120.5, 361.5), 'min'),
                    'time_duration_bounds': ((29.0, 87.0), 'min'),
                },
            },
            'post_mission': {
                'include_landing': False,
                'constrain_range': True,
                'target_range': (1906, 'nmi'),
            },
        }

        self.aircraft_definition_file = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'
        self.make_plots = False
        self.max_iter = 0

        _clear_problem_names()  # need to reset these to simulate separate runs

    def test_linkages(self):
        prob = AviaryProblem()

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs(self.aircraft_definition_file, self.phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('SLSQP', verbosity=0)

        prob.add_design_variables()

        # Load optimization problem formulation
        # Detail which variables the optimizer can control
        prob.add_objective()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_aviary_problem(run_driver=False, make_plots=False)

        constraints = prob.model.get_constraints()

        def get_linkage_string(phase_1, var, phase_2):
            return f'traj.linkages.{phase_1}:{var}_final|{phase_2}:{var}_initial'

        linkage_data = [
            ('cruise1', 'altitude', 'cruise2'),
            ('cruise2', 'altitude', 'cruise3'),
            ('cruise3', 'altitude', 'cruise4'),
            ('climb', 'mach', 'cruise0'),
            ('cruise0', 'mach', 'cruise1'),
            ('cruise1', 'mach', 'cruise2'),
            ('cruise3', 'mach', 'cruise4'),
            ('cruise4', 'mach', 'descent'),
        ]

        not_linkage_data = [
            ('climb', 'altitude', 'cruise0'),
            ('cruise0', 'altitude', 'cruise1'),
            ('cruise2', 'mach', 'cruise3'),
            ('cruise4', 'altitude', 'descent'),
        ]

        for phase_1, var, phase_2 in linkage_data:
            linkage_string = get_linkage_string(phase_1, var, phase_2)
            self.assertIn(linkage_string, constraints, f'Linkage {linkage_string} is missing')

        for phase_1, var, phase_2 in not_linkage_data:
            linkage_string = get_linkage_string(phase_1, var, phase_2)
            self.assertNotIn(linkage_string, constraints, f'Linkage {linkage_string} is present')


if __name__ == '__main__':
    unittest.main()
    # test = AircraftMissionTestSuite()
    # test.setUp()
    # test.test_linkages()
