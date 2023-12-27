import unittest

from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level1 import run_aviary
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.test.test_dummy_subsystem import ArrayGuessSubsystemBuilder
from aviary.variable_info.enums import EquationsOfMotion


@use_tempdirs
class AircraftMissionTestSuite(unittest.TestCase):

    def setUp(self):
        # Load the phase_info and other common setup tasks
        self.phase_info = {
            "pre_mission": {"include_takeoff": False, "optimize_mass": True},
            "climb_1": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_range": False,
                    "initial_mach": (0.2, "unitless"),
                    "final_mach": (0.72, "unitless"),
                    "mach_bounds": ((0.18, 0.74), "unitless"),
                    "initial_altitude": (0.0, "ft"),
                    "final_altitude": (32000.0, "ft"),
                    "altitude_bounds": ((0.0, 34000.0), "ft"),
                    "throttle_enforcement": "path_constraint",
                    "fix_initial": True,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 0.0), "min"),
                    "duration_bounds": ((64.0, 192.0), "min"),
                },
                "initial_guesses": {"times": ([0, 128], "min")},
            },
            "climb_2": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_range": False,
                    "initial_mach": (0.72, "unitless"),
                    "final_mach": (0.72, "unitless"),
                    "mach_bounds": ((0.7, 0.74), "unitless"),
                    "initial_altitude": (32000.0, "ft"),
                    "final_altitude": (34000.0, "ft"),
                    "altitude_bounds": ((23000.0, 38000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": False,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((64.0, 192.0), "min"),
                    "duration_bounds": ((56.5, 169.5), "min"),
                },
                "initial_guesses": {"times": ([128, 113], "min")},
            },
            "descent_1": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_range": False,
                    "initial_mach": (0.72, "unitless"),
                    "final_mach": (0.36, "unitless"),
                    "mach_bounds": ((0.34, 0.74), "unitless"),
                    "initial_altitude": (34000.0, "ft"),
                    "final_altitude": (500.0, "ft"),
                    "altitude_bounds": ((0.0, 38000.0), "ft"),
                    "throttle_enforcement": "path_constraint",
                    "fix_initial": False,
                    "constrain_final": True,
                    "fix_duration": False,
                    "initial_bounds": ((120.5, 361.5), "min"),
                    "duration_bounds": ((29.0, 87.0), "min"),
                },
                "initial_guesses": {"times": ([241, 58], "min")},
            },
            "post_mission": {
                "include_landing": False,
                "constrain_range": True,
                "target_range": (1906, "nmi"),
            },
        }

        self.aircraft_definition_file = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'
        self.make_plots = False
        self.max_iter = 100

    def add_external_subsystem(self, phase_info, subsystem_builder):
        """
        Add an external subsystem to all phases in the mission.
        """
        for phase in phase_info:
            if 'user_options' in phase_info[phase]:
                if 'external_subsystems' not in phase_info[phase]:
                    phase_info[phase]['external_subsystems'] = []
                phase_info[phase]['external_subsystems'].append(subsystem_builder)

    def run_mission(self, phase_info, optimizer):

        prob = AviaryProblem()

        prob.load_inputs(self.aircraft_definition_file, phase_info)
        prob.mission_method = EquationsOfMotion.SIMPLE
        prob.check_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver(optimizer)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()
        prob.failed = prob.run_aviary_problem()
        return prob

    def test_mission_basic(self):
        prob = self.run_mission(self.phase_info, "SLSQP")
        self.assertIsNotNone(prob)
        self.assertFalse(prob.failed)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_basic(self):
        prob = self.run_mission(self.phase_info, "IPOPT")
        self.assertIsNotNone(prob)
        self.assertFalse(prob.failed)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_optimize_mach_only(self):
        # Test with optimize_mach flag set to True
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb_1", "climb_2", "descent_1"]:
            modified_phase_info[phase]["user_options"]["optimize_mach"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertFalse(prob.failed)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_optimize_altitude_and_mach(self):
        # Test with optimize_altitude flag set to True
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb_1", "climb_2", "descent_1"]:
            modified_phase_info[phase]["user_options"]["optimize_altitude"] = True
            modified_phase_info[phase]["user_options"]["optimize_mach"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertFalse(prob.failed)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_optimize_altitude_only(self):
        # Test with optimize_altitude flag set to True
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb_1", "climb_2", "descent_1"]:
            modified_phase_info[phase]["user_options"]["optimize_altitude"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertFalse(prob.failed)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_solve_for_range(self):
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb_1", "climb_2", "descent_1"]:
            modified_phase_info[phase]["user_options"]["solve_for_range"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertFalse(prob.failed)

    def test_mission_solve_for_range(self):
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb_1", "climb_2", "descent_1"]:
            modified_phase_info[phase]["user_options"]["solve_for_range"] = True
        prob = self.run_mission(modified_phase_info, "SLSQP")
        self.assertFalse(prob.failed)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_with_external_subsystem(self):
        # Test mission with an external subsystem added to each phase
        modified_phase_info = self.phase_info.copy()
        dummy_subsystem_builder = ArrayGuessSubsystemBuilder()
        self.add_external_subsystem(modified_phase_info, dummy_subsystem_builder)

        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertFalse(prob.failed)


if __name__ == '__main__':
    unittest.main()
