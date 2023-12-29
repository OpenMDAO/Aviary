import unittest
import dymos
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from packaging import version

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.interface.default_phase_info.two_dof import phase_info as two_dof_phase_info
from aviary.interface.default_phase_info.height_energy import phase_info as height_energy_phase_info
from aviary.interface.default_phase_info.solved import phase_info as solved_phase_info
from aviary.variable_info.enums import EquationsOfMotion


class BaseProblemPhaseTestCase(unittest.TestCase):

    def build_and_run_problem(self, phase_info, input_filename, objective_type=None, mission_method=None):
        # Build problem
        prob = AviaryProblem()

        prob.load_inputs(input_filename, phase_info)

        prob.check_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver("SLSQP", max_iter=0)
        prob.add_design_variables()
        prob.add_objective(objective_type if objective_type else None)
        prob.setup()
        prob.set_initial_guesses()
        prob.run_aviary_problem("dymos_solution.db", make_plots=False)


@use_tempdirs
class TwoDOFZeroItersTestCase(BaseProblemPhaseTestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def test_2dof_zero_iters(self):
        self.build_and_run_problem(
            two_dof_phase_info, 'models/test_aircraft/aircraft_for_bench_GwGm.csv')


@use_tempdirs
class HEZeroItersTestCase(BaseProblemPhaseTestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def test_height_energy_zero_iters(self):
        self.build_and_run_problem(
            height_energy_phase_info, 'models/test_aircraft/aircraft_for_bench_FwFm.csv')


@unittest.skipIf(version.parse(dymos.__version__) <= version.parse("1.8.0"),
                 "Older version of Dymos treats non-time integration variables differently.")
@use_tempdirs
class SolvedProblemTestCase(BaseProblemPhaseTestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def test_zero_iters_solved(self):
        # Modify Aviary inputs before running the common operations
        self.build_and_run_problem(
            solved_phase_info, 'models/test_aircraft/aircraft_for_bench_GwGm_solved.csv', objective_type="hybrid_objective")


if __name__ == "__main__":
    # unittest.main()
    test = SolvedProblemTestCase()
    test.test_zero_iters_solved()
