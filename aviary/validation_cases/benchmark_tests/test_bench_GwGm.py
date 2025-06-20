import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.default_phase_info.two_dof import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.enums import AnalysisScheme
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a large single aisle commercial transport aircraft using
    GASP mass and aero method and TWO_DEGREES_OF_FREEDOM mission method. Expected outputs
    based on 'models/test_aircraft/aircraft_for_bench_FwFm.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='IPOPT')
    def test_bench_GwGm(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/test_aircraft/aircraft_for_bench_GwGm.csv',
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
            'models/test_aircraft/aircraft_for_bench_GwGm.csv',
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

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_GwGm_SNOPT_lbm_s(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/test_aircraft/aircraft_for_bench_GwGm_lbm_s.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
        )

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            173807.5346,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'),
            95620.3626,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            42187.1719,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2636.6632,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3675.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            136618.3626,
            tolerance=rtol,
        )

    @unittest.skip(
        'Shooting method is not correctly receiving user-set options, and is currently '
        'using default values for most options'
    )
    @require_pyoptsparse(optimizer='IPOPT')
    def test_bench_GwGm_shooting(self):
        self.skipTest('SGM currently failing this test.')

        from aviary.interface.default_phase_info.two_dof_fiti import (
            phase_info,
            phase_info_parameterization,
        )

        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='IPOPT',
            run_driver=False,
            analysis_scheme=AnalysisScheme.SHOOTING,
            verbosity=0,
            phase_info_parameterization=phase_info_parameterization,
        )

        rtol = 0.01

        assert_near_equal(
            prob.get_val(Mission.Design.RESERVE_FUEL, units='lbm'), 4998, tolerance=rtol
        )

        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            174039.0,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'),
            95509,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            43574.0,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2623.4,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3765.48, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            136823.47,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val('traj.cruise_' + Dynamic.Mission.DISTANCE + '_final', units='nmi'),
            3668.3,
            tolerance=rtol,
        )


if __name__ == '__main__':
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.setUp()
    test.test_bench_GwGm()
