import os
import unittest
import subprocess

from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.reports_system import clear_reports

from aviary.interface.methods_for_level1 import run_aviary
from aviary.subsystems.test.test_dummy_subsystem import ArrayGuessSubsystemBuilder
from aviary.mission.energy_phase import EnergyPhase
from aviary.variable_info.variables import Dynamic
from aviary.variable_info.enums import Verbosity


@use_tempdirs
class AircraftMissionTestSuite(unittest.TestCase):

    def setUp(self):

        # Load the phase_info and other common setup tasks
        self.phase_info = {
            "pre_mission": {"include_takeoff": False, "optimize_mass": True},
            "climb": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
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
                "initial_guesses": {"time": ([0, 128], "min")},
            },
            "cruise": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
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
                "initial_guesses": {"time": ([128, 113], "min")},
            },
            "descent": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": False,
                    "optimize_altitude": False,
                    "polynomial_control_order": 1,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
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
                "initial_guesses": {"time": ([241, 58], "min")},
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

        # need to reset these to simulate separate runs
        _clear_problem_names()
        clear_reports()

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
        return run_aviary(
            self.aircraft_definition_file, phase_info,
            make_plots=self.make_plots, max_iter=self.max_iter, optimizer=optimizer,
            optimization_history_filename="driver_test.db", verbosity=Verbosity.QUIET)

    def test_mission_basic_and_dashboard(self):
        # We need to remove the TESTFLO_RUNNING environment variable for this test to run.
        # The reports code checks to see if TESTFLO_RUNNING is set and will not do anything if set
        # But we need to remember whether it was set so we can restore it
        testflo_running = os.environ.pop('TESTFLO_RUNNING', None)

        prob = self.run_mission(self.phase_info, "SLSQP")

        # restore what was there before running the test
        if testflo_running is not None:
            os.environ['TESTFLO_RUNNING'] = testflo_running

        self.assertIsNotNone(prob)
        self.assertTrue(prob.problem_ran_successfully)

        cmd = f'aviary dashboard --problem_recorder dymos_solution.db --driver_recorder driver_test.db {prob.driver._problem()._name}'
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        try:
            subprocess.check_output(cmd.split())
        except subprocess.CalledProcessError as err:
            self.fail("Command '{}' failed.  Return code: {}".format(cmd, err.returncode))

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_basic_pyopt(self):
        prob = self.run_mission(self.phase_info, "IPOPT")
        self.assertIsNotNone(prob)
        self.assertTrue(prob.problem_ran_successfully)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_optimize_mach_only(self):
        # Test with optimize_mach flag set to True
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb", "cruise", "descent"]:
            modified_phase_info[phase]["user_options"]["optimize_mach"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertTrue(prob.problem_ran_successfully)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_optimize_altitude_and_mach(self):
        # Test with optimize_altitude flag set to True
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb", "cruise", "descent"]:
            modified_phase_info[phase]["user_options"]["optimize_altitude"] = True
            modified_phase_info[phase]["user_options"]["optimize_mach"] = True
        modified_phase_info['climb']['user_options']['constraints'] = {
            Dynamic.Mission.THROTTLE: {
                'lower': 0.2,
                'upper': 0.9,
                'type': 'path',
            },
        }
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertTrue(prob.problem_ran_successfully)

        constraints = prob.driver._cons
        for name, meta in constraints.items():
            if 'traj.phases.climb->path_constraint->throttle' in name:
                self.assertEqual(meta['upper'], 0.9)
                self.assertEqual(meta['lower'], 0.2)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_optimize_altitude_only(self):
        # Test with optimize_altitude flag set to True
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb", "cruise", "descent"]:
            modified_phase_info[phase]["user_options"]["optimize_altitude"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertTrue(prob.problem_ran_successfully)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_solve_for_distance_IPOPT(self):
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb", "cruise", "descent"]:
            modified_phase_info[phase]["user_options"]["solve_for_distance"] = True
        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertTrue(prob.problem_ran_successfully)

    def test_mission_solve_for_distance_SLSQP(self):
        modified_phase_info = self.phase_info.copy()
        for phase in ["climb", "cruise", "descent"]:
            modified_phase_info[phase]["user_options"]["solve_for_distance"] = True
        prob = self.run_mission(modified_phase_info, "SLSQP")
        self.assertTrue(prob.problem_ran_successfully)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_mission_with_external_subsystem(self):
        # Test mission with an external subsystem added to each phase
        modified_phase_info = self.phase_info.copy()
        dummy_subsystem_builder = ArrayGuessSubsystemBuilder()
        self.add_external_subsystem(modified_phase_info, dummy_subsystem_builder)

        prob = self.run_mission(modified_phase_info, "IPOPT")
        self.assertTrue(prob.problem_ran_successfully)

    def test_custom_phase_builder(self):
        local_phase_info = self.phase_info.copy()
        local_phase_info['climb']['phase_builder'] = EnergyPhase

        run_aviary(self.aircraft_definition_file, local_phase_info,
                   verbosity=Verbosity.QUIET, max_iter=1, optimizer='SLSQP')

    def test_custom_phase_builder_error(self):
        local_phase_info = self.phase_info.copy()
        local_phase_info['climb']['phase_builder'] = "fake phase object"

        with self.assertRaises(TypeError):
            run_aviary(self.aircraft_definition_file, local_phase_info,
                       verbosity=Verbosity.QUIET, max_iter=1, optimizer='SLSQP')


if __name__ == '__main__':
    unittest.main()
