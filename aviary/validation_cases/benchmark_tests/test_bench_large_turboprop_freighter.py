
import numpy as np
import unittest

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.turboprop_model import TurbopropModel
from aviary.utils.process_input_decks import create_vehicle


@use_tempdirs
class LargeTurbopropFreighterBenchmark(unittest.TestCase):

    def build_and_run_problem(self):
        # Define Mission
        phase_info = {
            "pre_mission": {"include_takeoff": False, "optimize_mass": True},
            "climb": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.2, "unitless"),
                    "final_mach": (0.475, "unitless"),
                    "mach_bounds": ((0.08, 0.478), "unitless"),
                    "initial_altitude": (0.0, "ft"),
                    "final_altitude": (21_000.0, "ft"),
                    "altitude_bounds": ((0.0, 22_000.0), "ft"),
                    "throttle_enforcement": "path_constraint",
                    "fix_initial": True,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((24.0, 192.0), "min"),
                    "add_initial_mass_constraint": False,
                },
            },
            "cruise": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.475, "unitless"),
                    "final_mach": (0.475, "unitless"),
                    "mach_bounds": ((0.47, 0.48), "unitless"),
                    "initial_altitude": (21_000.0, "ft"),
                    "final_altitude": (21_000.0, "ft"),
                    "altitude_bounds": ((20_000.0, 22_000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": False,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((64.0, 192.0), "min"),
                    "duration_bounds": ((56.5, 169.5), "min"),
                },
            },
            "descent": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.475, "unitless"),
                    "final_mach": (0.1, "unitless"),
                    "mach_bounds": ((0.08, 0.48), "unitless"),
                    "initial_altitude": (21_000.0, "ft"),
                    "final_altitude": (500.0, "ft"),
                    "altitude_bounds": ((0.0, 22_000.0), "ft"),
                    "throttle_enforcement": "path_constraint",
                    "fix_initial": False,
                    "constrain_final": True,
                    "fix_duration": False,
                    "initial_bounds": ((100, 361.5), "min"),
                    "duration_bounds": ((29.0, 87.0), "min"),
                },
            },
            "post_mission": {
                "include_landing": False,
                "constrain_range": True,
                "target_range": (2_020., "nmi"),
            },
        }

        # Build problem
        prob = AviaryProblem()

        # being able to load aviary inputs from a .csv needs to be its own util function
        options, _ = create_vehicle(
            "models/large_turboprop_freighter/large_turboprop_freighter.csv")
        turboprop = TurbopropModel('turboprop', options=options)

        # load inputs needs to
        prob.load_inputs(
            "models/large_turboprop_freighter/large_turboprop_freighter.csv", phase_info,
            engine_builders=[turboprop])

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver("SLSQP", max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()
        prob.run_aviary_problem("dymos_solution.db", make_plots=False)


if __name__ == '__main__':
    test = LargeTurbopropFreighterBenchmark()
    test.build_and_run_problem()
