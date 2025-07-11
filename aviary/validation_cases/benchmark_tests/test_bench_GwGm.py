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
    def test_bench_GwGm(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='IPOPT',
            verbosity=0,
        )

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            173806.9409,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'),
            95620.2190,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            42186.7219,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2636.6612,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3675.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            136618.2190,
            tolerance=rtol,
        )

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_GwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
        )

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            173806.9386,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'),
            95620.2185,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            42186.7201,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2636.6802,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3675.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            136618.2185,
            tolerance=rtol,
        )


if __name__ == '__main__':
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.setUp()
    test.test_bench_GwGm()
