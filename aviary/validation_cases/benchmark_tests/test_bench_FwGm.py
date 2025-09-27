import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.models.missions.two_dof_default import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a large single aisle commercial transport aircraft using
    FLOPS mass method, GASP aero method, and TWO_DEGREES_OF_FREEDOM mission method.
    Expected outputs based on 'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_swap_3_FwGm_IPOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwGm.csv',
            local_phase_info,
            max_iter=100,
            verbosity=0,
            optimizer='IPOPT',
        )

        # TODO: This problem does not always converge.
        # self.assertTrue(prob.result.success)

        rtol = 1e-2

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS), 176990.2, tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS), 101556.0, tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS), 37956.0, tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Landing.GROUND_DISTANCE), 2595.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val('traj.desc2.timeseries.distance')[-1], 3675.0, tolerance=rtol
        )

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_swap_3_FwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwGm.csv',
            local_phase_info,
            verbosity=0,
            optimizer='SNOPT',
            max_iter=60,
        )

        # TODO: This problem does not always converge.
        # self.assertTrue(prob.result.success)

        rtol = 1e-2

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS), 176965.48, tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS), 101556.0, tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS), 37918.8, tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Landing.GROUND_DISTANCE), 2595.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val('traj.desc2.timeseries.distance')[-1], 3675.0, tolerance=rtol
        )


if __name__ == '__main__':
    test = ProblemPhaseTestCase()
    test.setUp()
    test.bench_test_swap_3_FwGm_SNOPT()
    # test.bench_test_swap_3_FwGm_IPOPT()
