
import numpy as np
import unittest

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle
from aviary.variable_info.variables import Aircraft, Mission, Settings
from aviary.variable_info.enums import SpeedType


@use_tempdirs
class LargeTurbopropFreighterBenchmark(unittest.TestCase):

    def build_and_run_problem(self):
        # Define Mission
        # phase_info = {
        #     "pre_mission": {"include_takeoff": False, "optimize_mass": True},
        #     "climb": {
        #         "subsystem_options": {"core_aerodynamics": {"method": "solved_alpha"}},
        #         "user_options": {
        #             "optimize_mach": False,
        #             "optimize_altitude": False,
        #             "num_segments": 5,
        #             "order": 3,
        #             "solve_for_distance": False,
        #             "initial_mach": (0.2, "unitless"),
        #             "final_mach": (0.475, "unitless"),
        #             "mach_bounds": ((0.08, 0.478), "unitless"),
        #             "initial_altitude": (0.0, "ft"),
        #             "final_altitude": (21_000.0, "ft"),
        #             "altitude_bounds": ((0.0, 22_000.0), "ft"),
        #             "throttle_enforcement": "path_constraint",
        #             "fix_initial": True,
        #             "constrain_final": False,
        #             "fix_duration": False,
        #             "initial_bounds": ((0.0, 0.0), "min"),
        #             "duration_bounds": ((24.0, 192.0), "min"),
        #             "add_initial_mass_constraint": False,
        #         },
        #     },
        #     "cruise": {
        #         "subsystem_options": {"core_aerodynamics": {"method": "solved_alpha"}},
        #         "user_options": {
        #             "optimize_mach": False,
        #             "optimize_altitude": False,
        #             "num_segments": 5,
        #             "order": 3,
        #             "solve_for_distance": False,
        #             "initial_mach": (0.475, "unitless"),
        #             "final_mach": (0.475, "unitless"),
        #             "mach_bounds": ((0.47, 0.48), "unitless"),
        #             "initial_altitude": (21_000.0, "ft"),
        #             "final_altitude": (21_000.0, "ft"),
        #             "altitude_bounds": ((20_000.0, 22_000.0), "ft"),
        #             "throttle_enforcement": "boundary_constraint",
        #             "fix_initial": False,
        #             "constrain_final": False,
        #             "fix_duration": False,
        #             "initial_bounds": ((64.0, 192.0), "min"),
        #             "duration_bounds": ((56.5, 169.5), "min"),
        #         },
        #     },
        #     "descent": {
        #         "subsystem_options": {"core_aerodynamics": {"method": "solved_alpha"}},
        #         "user_options": {
        #             "optimize_mach": False,
        #             "optimize_altitude": False,
        #             "num_segments": 5,
        #             "order": 3,
        #             "solve_for_distance": False,
        #             "initial_mach": (0.475, "unitless"),
        #             "final_mach": (0.1, "unitless"),
        #             "mach_bounds": ((0.08, 0.48), "unitless"),
        #             "initial_altitude": (21_000.0, "ft"),
        #             "final_altitude": (500.0, "ft"),
        #             "altitude_bounds": ((0.0, 22_000.0), "ft"),
        #             "throttle_enforcement": "path_constraint",
        #             "fix_initial": False,
        #             "constrain_final": True,
        #             "fix_duration": False,
        #             "initial_bounds": ((100, 361.5), "min"),
        #             "duration_bounds": ((29.0, 87.0), "min"),
        #         },
        #     },
        #     "post_mission": {
        #         "include_landing": False,
        #         "constrain_range": True,
        #         "target_range": (2_020., "nmi"),
        #     },
        # }

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
                    'time': ([0.0, 40.0], 's'),
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
                    'time': ([40.0, 5.0], 's'),
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
                    'time': ([45., 25.], 's'),
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
                    'time': ([70., 13.], 's'),
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
                    'time': ([1., 2.], 'min'),
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
                    'time': ([216., 1300.], 's'),
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
                    'distance_ref': (2_020, 'NM'),
                    'distance_ref0': (0, 'NM'),
                    'distance_defect_ref': (100, 'NM'),
                },
                'initial_guesses': {
                    'mass': (136000., 'lbm'),
                    'altitude': ([37.5e3, 10.e3], 'ft'),
                    'throttle': ([0.0, 0.0], 'unitless'),
                    'distance': ([.92*2_020, .96*2_020], 'NM'),
                    'time': ([28000., 500.], 's'),
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
                    'distance': ([.96*2_020, 2_020], 'NM'),
                    'time': ([28500., 500.], 's'),
                }
            },
        }

        # Build problem
        prob = AviaryProblem()

        # load inputs from .csv to build engine
        options, _ = create_vehicle(
            "models/large_turboprop_freighter/large_turboprop_freighter_2.csv")

        turboprop = TurbopropModel('turboprop', options=options)

        # load_inputs needs to be updated to accept an already existing aviary options
        prob.load_inputs(
            "models/large_turboprop_freighter/large_turboprop_freighter_2.csv", phase_info,
            engine_builders=[turboprop])

        # FLOPS aero specific stuff? Best guesses for values here
        prob.aviary_inputs.set_val(Mission.Constraints.MAX_MACH, 0.5)
        prob.aviary_inputs.set_val(Aircraft.Fuselage.AVG_DIAMETER, 4.125, 'm')

        # Prop stuff
        # prob.aviary_inputs.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 4, 'm')
        # prob.aviary_inputs.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES, 4, 'unitless')
        # prob.aviary_inputs.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 167, 'unitless')
        # prob.aviary_inputs.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, 'unitless')

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver("SLSQP", max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        import openmdao.api as om
        om.n2(prob)
        prob.set_initial_guesses()
        prob.run_aviary_problem("dymos_solution.db", make_plots=False)
        om.n2(prob)


if __name__ == '__main__':
    test = LargeTurbopropFreighterBenchmark()
    test.build_and_run_problem()
