from copy import deepcopy
import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.default_phase_info.two_dof import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):

    @require_pyoptsparse(optimizer="IPOPT")
    def bench_test_swap_2_GwGm(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary('models/test_aircraft/aircraft_for_bench_GwGm.csv',
                          local_phase_info, optimizer='IPOPT')

        rtol = 0.01

        # There are no truth values for these.
        assert_near_equal(prob.get_val(Mission.Design.GROSS_MASS),
                          181654., tolerance=rtol)

        assert_near_equal(prob.get_val(Aircraft.Design.OPERATING_MASS),
                          101555., tolerance=rtol)

        assert_near_equal(prob.get_val(Mission.Summary.TOTAL_FUEL_MASS),
                          44098., tolerance=rtol)

        assert_near_equal(prob.get_val('landing.' + Mission.Landing.GROUND_DISTANCE),
                          2637.86, tolerance=rtol)

        assert_near_equal(prob.get_val("traj.desc2.timeseries.states:distance")[-1],
                          3675.0, tolerance=rtol)


if __name__ == '__main__':
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.bench_test_swap_2_GwGm()
