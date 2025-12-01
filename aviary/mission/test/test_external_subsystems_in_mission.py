"""Test for some features when using an external subsystem in the mission."""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.models.missions.height_energy_default import phase_info as energy_phase_info
from aviary.models.missions.two_dof_default import phase_info as two_dof_phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Aircraft

energy_phase_info = deepcopy(energy_phase_info)

energy_phase_info['pre_mission']['include_takeoff'] = False
energy_phase_info['post_mission']['include_landing'] = False
energy_phase_info.pop('climb')
energy_phase_info.pop('descent')

# TODO once 2DOF doesn't force exact mission phases + order remove everything but cruise
#      for faster test evaluation
# two_dof_phase_info = deepcopy(two_dof_phase_info)
# two_dof_phase_info.pop('groundroll')
# two_dof_phase_info.pop('rotation')
# two_dof_phase_info.pop('ascent')
# two_dof_phase_info.pop('accel')
# two_dof_phase_info.pop('climb1')
# two_dof_phase_info.pop('climb2')
# two_dof_phase_info.pop('desc1')
# two_dof_phase_info.pop('desc2')


@use_tempdirs
class TestExternalSubsystems(unittest.TestCase):
    def test_mission_solver_energy(self):
        local_phase_info = deepcopy(energy_phase_info)
        local_phase_info['cruise']['external_subsystems'] = [SolverBuilder(name='solve_me')]

        prob = AviaryProblem()

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs_all.solver_sub.external_subsystems,
                'solve_me',
            )
        )

    def test_no_mission_solver_energy(self):
        local_phase_info = deepcopy(energy_phase_info)
        local_phase_info['cruise']['external_subsystems'] = [
            NoSolverBuilder(name='do_not_solve_me')
        ]

        prob = AviaryProblem()

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs_all.external_subsystems,
                'do_not_solve_me',
            )
        )

    def test_mission_solver_2DOF(self):
        local_phase_info = deepcopy(two_dof_phase_info)
        local_phase_info['cruise']['external_subsystems'] = [SolverBuilder(name='solve_me')]

        prob = AviaryProblem()

        prob.load_inputs(
            'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        # NOTE currently 2DOF ODEs do not use the solver subsystem
        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs.external_subsystems,
                'solve_me',
            )
        )

    def test_no_mission_solver_2DOF(self):
        local_phase_info = deepcopy(two_dof_phase_info)
        local_phase_info['cruise']['external_subsystems'] = [
            NoSolverBuilder(name='do_not_solve_me')
        ]

        prob = AviaryProblem()

        prob.load_inputs(
            'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs.external_subsystems,
                'do_not_solve_me',
            )
        )


class ExternNoSolve(om.ExplicitComponent):
    """This component should not have a solver above it."""

    def setup(self):
        self.add_input(Aircraft.Wing.AREA, 1.0, units='ft**2')
        self.add_output('stuff', 1.0, units='ft**2')

    def compute(self, inputs, outputs):
        pass


class NoSolverBuilder(SubsystemBuilderBase):
    """Mission only. No solver."""

    def needs_mission_solver(self, aviary_options):
        return False

    def build_mission(self, num_nodes, aviary_inputs):
        return ExternNoSolve()


class SolverBuilder(SubsystemBuilderBase):
    """Mission only. Solver."""

    def needs_mission_solver(self, aviary_options):
        return True

    def build_mission(self, num_nodes, aviary_inputs):
        return ExternNoSolve()


if __name__ == '__main__':
    unittest.main()
    # test = TestExternalSubsystems()
    # test.test_no_mission_solver_2DOF()
