import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_groundroll import \
    run_groundroll


@unittest.skip('this benchmark is defunct and needs to be updated')
class GroundrollPhaseTestCase(unittest.TestCase):
    def bench_test_groundroll(self):

        prob = run_groundroll(make_plots=False)

        TAS = prob.get_val("traj.groundroll.timeseries.states:TAS", units="kn")
        mass = prob.get_val("traj.groundroll.timeseries.states:mass", units="lbm")
        distance = prob.get_val(
            "traj.groundroll.timeseries.states:distance", units="ft"
        )
        time = prob.get_val("traj.groundroll.timeseries.time", units="s")

        assert_near_equal(TAS[0], 0, 1e-4)
        assert_near_equal(TAS[-1], 145.37726917, 1e-4)

        assert_near_equal(mass[0], 175100, 1e-6)
        assert_near_equal(mass[-1], 174978.19460438, 1e-6)

        assert_near_equal(distance[0], 0, 1e-4)
        assert_near_equal(distance[-1], 3547.50615602, 1e-4)

        assert_near_equal(time[0], 0, 1e-3)
        assert_near_equal(time[-1], 28.03023543, 1e-3)


if __name__ == "__main__":
    unittest.main()
