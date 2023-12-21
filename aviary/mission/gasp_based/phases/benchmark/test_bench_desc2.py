import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_desc2 import run_desc2


@unittest.skip('this benchmark is defunct and needs to be updated')
class Desc2PhaseTestCase(unittest.TestCase):
    def bench_test_desc2(self):

        prob = run_desc2(make_plots=False)

        alt = prob.get_val("desc2.timeseries.states:altitude", units="ft")
        weight = prob.get_val("desc2.timeseries.states:weight", units="lbm")
        distance = prob.get_val("desc2.timeseries.states:distance", units="ft")
        time = prob.get_val("desc2.timeseries.time", units="s")

        assert_near_equal(alt[0], 10.e3, 1e-5)
        assert_near_equal(alt[-1], 1000, 1e-4)

        assert_near_equal(weight[0], 147541.5, 1e-5)
        assert_near_equal(weight[-1], 147401.60702539, 1e-5)

        assert_near_equal(distance[0], 15576122.0472441, 1e-5)
        assert_near_equal(distance[-1], 15776177.53801624, 1e-5)

        assert_near_equal(time[0], 0, 1e-4)
        assert_near_equal(time[-1], 437.62278284, 1e-5)


if __name__ == "__main__":
    unittest.main()
