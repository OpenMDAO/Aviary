import unittest
import aviary.api as av

from aviary.models.missions.solved2dof_default import phase_info
from aviary.models.missions.solved2dof_landing_default import phase_info as phase_info_landing
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


@use_tempdirs
class TestBenchSolved2DOF(unittest.TestCase):
    """Run the model in serial that is setup in ProblemPhaseTestCase class."""

    def test_bench_Solved2DOF(self):
        prob = av.run_aviary(
            aircraft_data='validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
            phase_info=phase_info,
            optimizer='IPOPT',
            objective_type='time',
            max_iter=100,
        )

        self.assertTrue(prob.result.success)
        tol = 1e-4
        assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 108.84030411, tol)
        assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 459.3830223, tol)

    def test_bench_Solved2DOF_landing(self):
        prob = av.run_aviary(
            aircraft_data='validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
            phase_info=phase_info_landing,
            optimizer='IPOPT',
            objective_type='time',
            max_iter=100,
        )

        self.assertTrue(prob.result.success)
        tol = 1e-4
        assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 68.30257602, tol)
        assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 98.90626609, tol)

    # def test_bench_Solved2DOF_fuel_burned(self):
    #     prob = av.run_aviary(
    #         aircraft_data='validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
    #         phase_info=phase_info,
    #         optimizer='IPOPT',
    #         objective_type='fuel_burned',
    #         max_iter=100,
    #     )

    #     self.assertTrue(prob.result.success)
    #     tol = 1e-3
    #     assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 112.27911183, tol)
    #     assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 431.72131658, tol)

    #     # this test aims to optimize  for maximum mass (minimum fuel burn) for this mission
    #     prob = av.AviaryProblem()
    #     prob.load_inputs(
    #         'validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
    #         phase_info,
    #     )
    #     prob.load_external_subsystems()
    #     prob.check_and_preprocess_inputs()
    #     prob.add_pre_mission_systems()
    #     prob.add_phases()
    #     prob.add_post_mission_systems()
    #     prob.link_phases()
    #     prob.add_driver('IPOPT', max_iter=100)
    #     prob.add_design_variables()
    #     prob.add_objective(objective_type='mass', ref=-1e4)
    #     prob.setup()
    #     prob.run_aviary_problem()

    #     self.assertTrue(prob.result.success)


if __name__ == '__main__':
    unittest.main()
