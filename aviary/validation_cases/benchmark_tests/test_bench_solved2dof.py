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
        tol = 1e-2
        assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 108.84030411, tol)
        assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 459.3830223, tol)

    def test_bench_Solved2DOF_landing(self):
        # This problem solves better with a reduced ref for objective time, therefore need to call add_objectve()
        prob = av.AviaryProblem()
        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
            phase_info_landing,
        )
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('IPOPT', max_iter=100)
        prob.add_design_variables()
        prob.add_objective('time', ref=1e2)
        prob.setup()
        prob.run_aviary_problem()

        self.assertTrue(prob.result.success)
        tol = 1e-2
        assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 68.30353617, tol)
        assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 98.91566618, tol)


if __name__ == '__main__':
    unittest.main()
