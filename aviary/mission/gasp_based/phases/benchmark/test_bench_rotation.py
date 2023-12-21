import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_rotation import run_rotation


@unittest.skip('this benchmark is defunct and needs to be updated')
class RotationPhaseTestCase(unittest.TestCase):
    def bench_test_rotation(self):

        prob = run_rotation(make_plots=False)

        alpha = prob.get_val("traj.rotation.timeseries.states:alpha", units="deg")
        TAS = prob.get_val("traj.rotation.timeseries.states:TAS", units="kn")
        weight = prob.get_val("traj.rotation.timeseries.states:weight", units="lbm")
        distance = prob.get_val("traj.rotation.timeseries.states:distance", units="ft")
        normal_force = prob.get_val(
            "traj.rotation.timeseries.normal_force", units="lbf"
        )
        time = prob.get_val("traj.rotation.timeseries.time", units="s")

        assert_near_equal(alpha[0], 0, 1e-4)
        assert_near_equal(alpha[-1], 25, 1e-4)

        assert_near_equal(TAS[0], 143, 1e-4)
        assert_near_equal(TAS[-1], 173.4636624, 1e-4)

        assert_near_equal(weight[0], 174975.12776915, 1e-6)
        assert_near_equal(weight[-1], 174942.19649927, 1e-6)

        assert_near_equal(distance[0], 3680.37217765, 1e-4)
        assert_near_equal(distance[-1], 5693.05954272, 1e-4)

        assert_near_equal(normal_force[0], 118093.50754183, 1e-5)
        assert_near_equal(normal_force[-1], 0, 1e-5)

        assert_near_equal(time[0], 0, 1e-3)
        assert_near_equal(time[-1], 7.50750751, 1e-3)


if __name__ == "__main__":
    unittest.main()
