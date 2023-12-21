import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_desc1 import run_desc1


@unittest.skip('this benchmark is defunct and needs to be updated')
class Desc1PhaseTestCase(unittest.TestCase):
    def bench_test_desc1(self):

        prob = run_desc1(make_plots=False)

        alt = prob.get_val("desc1.timeseries.states:altitude", units="ft")
        weight = prob.get_val("desc1.timeseries.states:weight", units="lbm")
        distance = prob.get_val("desc1.timeseries.states:distance", units="ft")
        time = prob.get_val("desc1.timeseries.time", units="s")

        assert_near_equal(alt[0], 37500, 1e-5)
        assert_near_equal(alt[-1], 10.e3, 1e-5)

        assert_near_equal(weight[0], 147664, 1e-5)
        assert_near_equal(weight[-1], 147539.71665978, 1e-5)

        assert_near_equal(distance[0], 15129527.55905512, 1e-5)
        assert_near_equal(distance[-1], 15595089.68838349, 1e-5)

        assert_near_equal(time[0], 0, 1e-4)
        assert_near_equal(time[-1], 598.6264441, 1e-5)


if __name__ == "__main__":
    unittest.main()
