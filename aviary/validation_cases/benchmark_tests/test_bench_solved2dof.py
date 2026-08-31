import unittest
import aviary.api as av

from aviary.models.missions.solved2dof_default import phase_info
from aviary.models.missions.solved2dof_landing_default import phase_info as phase_info_landing
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from aviary.validation_cases.benchmark_utils import print_benchmark_results


@use_tempdirs
class TestBenchSolved2DOF(unittest.TestCase):
    """Run the model in serial that is setup in ProblemPhaseTestCase class."""

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_Solved2DOF(self):
        subsystem_options = {
            'aerodynamics': {
                'method': 'low_speed',
                'ground_altitude': 0.0,  # units='ft'
                'angles_of_attack': [
                    -5.0,
                    -4.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ],  # units='deg'
                'lift_coefficients': [
                    0.01,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5178,
                    0.6,
                    0.75,
                    0.85,
                    0.95,
                    1.05,
                    1.15,
                    1.25,
                    1.35,
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.85,
                    1.9,
                    1.95,
                ],
                'drag_coefficients': [
                    0.04,
                    0.02,
                    0.01,
                    0.02,
                    0.04,
                    0.0674,
                    0.065,
                    0.065,
                    0.07,
                    0.072,
                    0.076,
                    0.084,
                    0.09,
                    0.10,
                    0.11,
                    0.12,
                    0.13,
                    0.15,
                    0.16,
                    0.18,
                    0.20,
                ],
                'lift_coefficient_factor': 1.0,
                'drag_coefficient_factor': 1.0,
            }
        }
        for phase in phase_info:
            if not (phase == 'pre_mission' or phase == 'post_mission'):
                phase_info[phase]['subsystem_options'] = subsystem_options

        prob = av.run_aviary(
            aircraft_data='validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
            phase_info=phase_info,
            optimizer='SNOPT',
            objective_type='time',
            max_iter=100,
        )

        print_benchmark_results(prob)
        # self.assertTrue(prob.result.success)

        tol = 1e-2
        assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 108.84030411, tol)
        assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 459.3830223, tol)

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_Solved2DOF_landing(self):
        # This problem solves better with a reduced ref for objective time, therefore need to call add_objectve()
        subsystem_options = {
            'aerodynamics': {
                'method': 'low_speed',
                'ground_altitude': 0.0,  # units='ft'
                'angles_of_attack': [
                    -5.0,
                    -4.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ],  # units='deg'
                'lift_coefficients': [
                    0.01,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5178,
                    0.6,
                    0.75,
                    0.85,
                    0.95,
                    1.05,
                    1.15,
                    1.25,
                    1.35,
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.85,
                    1.9,
                    1.95,
                ],
                'drag_coefficients': [
                    0.04,
                    0.02,
                    0.01,
                    0.02,
                    0.04,
                    0.0674,
                    0.065,
                    0.065,
                    0.07,
                    0.072,
                    0.076,
                    0.084,
                    0.09,
                    0.10,
                    0.11,
                    0.12,
                    0.13,
                    0.15,
                    0.16,
                    0.18,
                    0.20,
                ],
                'lift_coefficient_factor': 2.0,
                'drag_coefficient_factor': 2.0,
            }
        }
        subsystem_options_landing = subsystem_options.copy()
        subsystem_options_landing['aerodynamics']['drag_coefficient_factor'] = 3.0

        for phase in phase_info_landing:
            if not (phase == 'pre_mission' or phase == 'post_mission'):
                if phase == 'IJ':
                    phase_info_landing[phase]['subsystem_options'] = subsystem_options
                else:
                    phase_info_landing[phase]['subsystem_options'] = subsystem_options_landing

        prob = av.AviaryProblem()
        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_solved2dof.csv',
            phase_info_landing,
        )
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('SNOPT', max_iter=100)
        prob.add_design_variables()
        prob.add_objective('time', ref=1e2)
        prob.setup()
        prob.run_aviary_problem()

        print_benchmark_results(prob)
        # self.assertTrue(prob.result.success)

        tol = 1e-2
        assert_near_equal(prob.get_val(av.Mission.FINAL_TIME, units='s'), 68.30353617, tol)
        assert_near_equal(prob.get_val(av.Mission.FUEL_MASS, units='lbm'), 98.91566618, tol)


if __name__ == '__main__':
    # unittest.main()
    z = TestBenchSolved2DOF()
    z.bench_test_Solved2DOF_landing()
