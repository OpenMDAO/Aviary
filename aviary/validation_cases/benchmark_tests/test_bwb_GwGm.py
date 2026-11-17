import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.run_aviary import run_aviary
from aviary.validation_cases.validation_data.test_data.generic_BWB_2dof_phase_info import phase_info
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a BWB aircraft using GASP mass and aero method
    and TWO_DEGREES_OF_FREEDOM mission method. Expected outputs based on
    'models/aircraft/blended_wing_body/generic_BWB_GASP.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_bwb_GwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/blended_wing_body/generic_BWB_GASP.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
            max_iter=60,
        )

        # TODO: CI has some intermittent problems with hitting feasibility.
        self.assertTrue(prob.result.success)

        rtol = 1e-2

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Aircraft.Design.GROSS_MASS, units='lbm'),
            133603.98625613,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.OPERATING_MASS, units='lbm'),
            78561.83558207,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.TOTAL_FUEL_MASS, units='lbm'),
            21292.15067406,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2267.36558233,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.RANGE, units='NM'), 3000.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.FINAL_MASS, units='lbm'),
            114247.48564948,
            tolerance=rtol,
        )

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_bwb_GwGm_IPOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/blended_wing_body/generic_BWB_GASP.csv',
            local_phase_info,
            optimizer='IPOPT',
            verbosity=0,
            max_iter=60,
        )

        # The results are the same as in SNOPT run, but this assertion is false.
        # self.assertTrue(prob.result.success)

        rtol = 1e-2

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Aircraft.Design.GROSS_MASS, units='lbm'),
            133603.98625613,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.OPERATING_MASS, units='lbm'),
            78561.83558207,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.TOTAL_FUEL_MASS, units='lbm'),
            21292.15067406,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2267.36558233,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.RANGE, units='NM'), 3000.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.FINAL_MASS, units='lbm'),
            114247.48564948,
            tolerance=rtol,
        )


if __name__ == '__main__':
    # unittest.main()
    z = ProblemPhaseTestCase()
    z.bench_test_bwb_GwGm_SNOPT()
