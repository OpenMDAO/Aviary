import unittest
import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.examples.multi_mission.run_multimission_example import multi_mission_example


@use_tempdirs
class MultiMissionTestcase(unittest.TestCase):
    """Test the different throttle allocation methods for models with multiple, unique EngineModels."""

    def setUp(self):
        om.clear_reports()
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multimission(self):
        prob = multi_mission_example()

        objective = prob.get_val('composite_objective', units=None)
        objective_expected_value = 24851.7

        mission1_fuel = prob.get_val('mission1.mission:summary:fuel_burned', units='lbm')
        mission1_fuel_expected_value = 26211.5

        mission2_fuel = prob.get_val('mission2.mission:summary:fuel_burned', units='lbm')
        mission2_fuel_expected_value = 22132.2

        mission1_cargo = prob.get_val(
            'mission1.aircraft:crew_and_payload:total_payload_mass', units='lbm'
        )
        mission1_cargo_expected_value = 36477.0

        mission2_cargo = prob.get_val(
            'mission2.aircraft:crew_and_payload:total_payload_mass', units='lbm'
        )
        mission2_cargo_expected_value = 4277.0

        # alloc_climb = prob.get_val('traj.climb.parameter_vals:throttle_allocations', units='')

        self.assertTrue(prob.result.success)

        expected_values = {
            'objective': (objective_expected_value, objective),
            'mission1_fuel': (mission1_fuel_expected_value, mission1_fuel),
            'mission2_fuel': (mission2_fuel_expected_value, mission2_fuel),
            'mission1_cargo': (mission1_cargo_expected_value, mission1_cargo),
            'mission2_cargo': (mission2_cargo_expected_value, mission2_cargo),
        }

        for var_name, (expected, actual) in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(expected, actual, tolerance=1e-3)


if __name__ == '__main__':
    unittest.main()
