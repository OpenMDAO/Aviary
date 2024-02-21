from copy import deepcopy
import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from openmdao.core.problem import _clear_problem_names

from aviary.interface.default_phase_info.two_dof import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.enums import AnalysisScheme


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):

    def setup(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer="IPOPT")
    def test_bench_GwGm(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary('models/test_aircraft/aircraft_for_bench_GwGm.csv',
                          local_phase_info, optimizer='IPOPT')

        rtol = 0.01

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS),
                          174039., tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS),
                          95509, tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS),
                          42529., tolerance=rtol)

        assert_near_equal(prob.get_val('landing.' + Mission.Landing.GROUND_DISTANCE),
                          2634.8, tolerance=rtol)

        assert_near_equal(prob.get_val("traj.desc2.timeseries.distance")[-1],
                          3675.0, tolerance=rtol)

    @require_pyoptsparse(optimizer="SNOPT")
    def test_bench_GwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary('models/test_aircraft/aircraft_for_bench_GwGm.csv',
                          local_phase_info, optimizer='SNOPT')

        rtol = 0.01

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS),
                          174039., tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS),
                          95509, tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS),
                          42529., tolerance=rtol)

        assert_near_equal(prob.get_val('landing.' + Mission.Landing.GROUND_DISTANCE),
                          2634.8, tolerance=rtol)

        assert_near_equal(prob.get_val("traj.desc2.timeseries.distance")[-1],
                          3675.0, tolerance=rtol)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_bench_GwGm_shooting(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary('models/test_aircraft/aircraft_for_bench_GwGm.csv',
                          local_phase_info, optimizer='IPOPT', run_driver=False,
                          analysis_scheme=AnalysisScheme.SHOOTING)

        rtol = 0.01

        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS),
                          174039., tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS),
                          95509, tolerance=rtol)

        assert_near_equal(prob.get_val('traj.distance_final', units='NM'),
                          3774.3, tolerance=rtol)

        assert_near_equal(prob.get_val('traj.mass_final', units='lbm'),
                          136823.47, tolerance=rtol)


if __name__ == '__main__':
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.test_bench_GwGm_SNOPT()
    test.test_bench_GwGm_shooting()
