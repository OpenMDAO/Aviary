import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_accel import run_accel


class AccelPhaseTestCase(unittest.TestCase):
    def bench_test_accel(self):

        prob = run_accel()

        TAS = prob.get_val("accel.timeseries.TAS", units="kn")
        weight = prob.get_val("accel.timeseries.mass", units="lbm")
        distance = prob.get_val("accel.timeseries.distance", units="ft")
        time = prob.get_val("accel.timeseries.time", units="s")

        assert_near_equal(TAS[0], 185, 1e-4)
        assert_near_equal(TAS[-1], 251.82436866, 1e-4)

        assert_near_equal(weight[0], 174974, 1e-4)
        assert_near_equal(weight[-1], 174886.73023817, 1e-4)

        assert_near_equal(distance[0], 0, 1e-5)
        assert_near_equal(distance[-1], 7519.70802292, 1e-5)

        assert_near_equal(time[0], 0, 1e-3)
        assert_near_equal(time[-1], 20.36335559, 1e-4)


if __name__ == "__main__":
    unittest.main()
