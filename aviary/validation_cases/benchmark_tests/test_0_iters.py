import unittest
import dymos
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from packaging import version

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.interface.default_phase_info.gasp import phase_info as gasp_phase_info
from aviary.interface.default_phase_info.simple import phase_info as flops_phase_info
from aviary.interface.default_phase_info.solved import phase_info as solved_phase_info


class BaseProblemPhaseTestCase(unittest.TestCase):

    def build_and_run_problem(self, phase_info, mission_method, mass_method, input_filename, objective_type=None):
        # Build problem
        prob = AviaryProblem(
            phase_info, mission_method=mission_method, mass_method=mass_method)

        prob.load_inputs(input_filename)

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
class GASPZeroItersTestCase(BaseProblemPhaseTestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def test_gasp_zero_iters(self):
        self.build_and_run_problem(gasp_phase_info, "GASP",
                                   "GASP", 'models/test_aircraft/aircraft_for_bench_GwGm.csv')


@use_tempdirs
class FLOPSZeroItersTestCase(BaseProblemPhaseTestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def test_flops_zero_iters(self):
        self.build_and_run_problem(flops_phase_info, "simple",
                                   "FLOPS", 'models/test_aircraft/aircraft_for_bench_FwFm.csv')


@unittest.skipIf(version.parse(dymos.__version__) <= version.parse("1.8.0"),
                 "Older version of Dymos treats non-time integration variables differently.")
@use_tempdirs
class SolvedProblemTestCase(BaseProblemPhaseTestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def test_zero_iters_solved(self):
        # Modify Aviary inputs before running the common operations
        self.build_and_run_problem(solved_phase_info, "solved",
                                   "GASP", 'models/test_aircraft/aircraft_for_bench_GwGm.csv', objective_type="hybrid_objective")


if __name__ == "__main__":
    unittest.main()
    # test = FLOPSZeroItersTestCase()
    # test.test_flops_zero_iters()
