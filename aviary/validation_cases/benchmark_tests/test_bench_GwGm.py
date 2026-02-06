import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.models.missions.two_dof_default import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a large single aisle commercial transport aircraft using
    GASP mass and aero method and TWO_DEGREES_OF_FREEDOM mission method. Expected outputs
    based on 'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='IPOPT')
    def test_bench_GwGm_IPOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='IPOPT',
            verbosity=0,
        )

        self.assertTrue(prob.result.success)

        rtol = 1e-3

        # There are no truth values for these.
        expected_values = {
            (Mission.Design.GROSS_MASS, 'lbm'): 171646.48312684,
            (Mission.Summary.OPERATING_MASS, 'lbm'): 95100.07583783,
            (Mission.Summary.TOTAL_FUEL_MASS, 'lbm'): 40546.40728901,
            (Mission.Landing.GROUND_DISTANCE, 'ft'): 2655.5028412,
            (Mission.Summary.RANGE, 'NM'): 3675.0,
            (Mission.Landing.TOUCHDOWN_MASS, 'lbm'): 136098.07583783,
        }

        for (var_name, units), expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob.get_val(var_name, units=units), expected_val, tolerance=rtol)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_GwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
        )

        self.assertTrue(prob.result.success)

        rtol = 1e-3

        # There are no truth values for these.
        expected_values = {
            (Mission.Design.GROSS_MASS, 'lbm'): 171646.48312684,
            (Mission.Summary.OPERATING_MASS, 'lbm'): 95100.07583783,
            (Mission.Summary.TOTAL_FUEL_MASS, 'lbm'): 40546.40728901,
            (Mission.Landing.GROUND_DISTANCE, 'ft'): 2655.5028412,
            (Mission.Summary.RANGE, 'NM'): 3675.0,
            (Mission.Landing.TOUCHDOWN_MASS, 'lbm'): 136098.07583783,
        }

        for (var_name, units), expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob.get_val(var_name, units=units), expected_val, tolerance=rtol)


if __name__ == '__main__':
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.setUp()
    test.test_bench_GwGm_SNOPT()
