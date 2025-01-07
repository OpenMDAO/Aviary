"""
Test for some features when using an external subsystem in the mission.
"""
from copy import deepcopy
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.interface.methods_for_level2 import AviaryProblem

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.csv_data_file import read_data_file
from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.variable_info.variables import Aircraft


# The drag-polar-generating component reads this in, instead of computing the polars.
polar_file = "subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_free_reduced_alpha.txt"

phase_info = deepcopy(phase_info)

phase_info['pre_mission']['include_takeoff'] = False
phase_info['post_mission']['include_landing'] = False
phase_info.pop('climb')
phase_info.pop('descent')


class TestSolvedAero(unittest.TestCase):

    def test_mission_solver(self):

        local_phase_info = deepcopy(phase_info)
        local_phase_info['cruise']['external_subsystems'] = [
            SolverBuilder(name='solve_me')]

        prob = AviaryProblem()

        prob.load_inputs(
            "subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv", local_phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        self.assertTrue(hasattr(
            prob.model.traj.phases.cruise.rhs_all.solver_sub.external_subsystems,
            "solve_me"
        ))

    def test_no_mission_solver(self):

        local_phase_info = deepcopy(phase_info)
        local_phase_info['cruise']['external_subsystems'] = [
            NoSolverBuilder(name='do_not_solve_me')]

        prob = AviaryProblem()

        prob.load_inputs(
            "subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv",
            local_phase_info
        )

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        self.assertTrue(hasattr(
            prob.model.traj.phases.cruise.rhs_all.external_subsystems,
            "do_not_solve_me"
        ))


class ExternNoSolve(om.ExplicitComponent):
    """
    This component should not have a solver above it.
    """

    def setup(self):
        self.add_input(Aircraft.Wing.AREA, 1.0, units='ft**2')
        self.add_output("stuff", 1.0, units='ft**2')

    def compute(self, inputs, outputs):
        pass


class NoSolverBuilder(SubsystemBuilderBase):
    """
    Mission only. No solver.
    """

    def needs_mission_solver(self, aviary_options):
        return False

    def build_mission(self, num_nodes, aviary_inputs):
        return ExternNoSolve()


class SolverBuilder(SubsystemBuilderBase):
    """
    Mission only. No solver.
    """

    def needs_mission_solver(self, aviary_options):
        return True

    def build_mission(self, num_nodes, aviary_inputs):
        return ExternNoSolve()


if __name__ == "__main__":
    unittest.main()
