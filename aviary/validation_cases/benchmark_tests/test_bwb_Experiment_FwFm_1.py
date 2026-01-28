import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.models.aircraft.large_turboprop_freighter.phase_info import (
    energy_phase_info as phase_info,
)

from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Mission


# @use_tempdirs
class BWBProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a BWB aircraft using FLOPS mass and aero method
    and HEIGHT_ENERGY mission method. Expected outputs based on
    'models/aircraft/blended_wing_body/bwb_simple_FLOPS.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_bwb_FwFm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/blended_wing_body/bwb_simple_FLOPS.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
            max_iter=60,
        )
        # prob.model.list_vars(units=True, print_arrays=True)
        # prob.list_indep_vars()
        # prob.list_problem_vars()
        # prob.model.list_outputs()

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            139803.667415,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.OPERATING_MASS, units='lbm'),
            79873.05255347,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            26180.61486153,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3500.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2216.0066613,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            116003.31044998,
            tolerance=rtol,
        )


class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a test aircraft using FLOPS mass and aero method
    and HEIGHT_ENERGY mission method. Expected outputs based on
    'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_FwFm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
            max_iter=60,
        )
        # prob.list_indep_vars()
        # prob.list_problem_vars()
        # prob.model.list_outputs()

        # self.assertTrue(prob.result.success)

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            169804.16225263,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.OPERATING_MASS, units='lbm'),
            97096.89284117,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            34682.26941131,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 2020.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2216.0066613,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            116003.31044998,
            tolerance=rtol,
        )


if __name__ == '__main__':
    # unittest.main()
    test = BWBProblemPhaseTestCase()
    test.setUp()
    test.test_bench_bwb_FwFm_SNOPT()
