import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_takeoff import run_takeoff


@unittest.skip('this benchmark is defunct and needs to be updated')
class TakeoffPhaseTestCase(unittest.TestCase):
    def bench_test_takeoff(self):

        prob = run_takeoff(make_plots=False)

        TAS0 = prob.get_val("traj.groundroll.timeseries.states:TAS", units="kn")
        distance0 = prob.get_val(
            "traj.groundroll.timeseries.states:distance", units="ft"
        )
        weight0 = prob.get_val("traj.groundroll.timeseries.states:weight", units="lbm")
        time0 = prob.get_val("traj.groundroll.timeseries.time", units="s")

        gamma1 = prob.get_val(
            "traj.ascent.timeseries.states:flight_path_angle", units="deg")
        alt1 = prob.get_val("traj.ascent.timeseries.states:altitude", units="ft")
        TAS1 = prob.get_val("traj.ascent.timeseries.states:TAS", units="kn")
        distance1 = prob.get_val("traj.ascent.timeseries.states:distance", units="ft")
        load_factor1 = prob.get_val(
            "traj.ascent.timeseries.load_factor", units="unitless")
        fuselage_pitch1 = prob.get_val(
            "traj.ascent.timeseries.fuselage_pitch", units="deg"
        )
        alpha1 = prob.get_val("traj.ascent.timeseries.controls:alpha", units="deg")
        weight1 = prob.get_val("traj.ascent.timeseries.states:weight", units="lbm")
        time1 = prob.get_val("traj.ascent.timeseries.time", units="s")

        assert_near_equal(gamma1[-1], -0.62446459, 1e-3)

        assert_near_equal(alt1[-1], 500, 1e-4)

        assert_near_equal(TAS0[0], 0, 1e-4)
        assert_near_equal(TAS1[-1], 191.15818794, 1e-4)

        assert_near_equal(distance0[0], 0, 1e-5)
        assert_near_equal(distance1[-1], 10000.0, 1e-5)

        assert_near_equal(load_factor1[-1], 0.16259711, 1e-2)

        assert_near_equal(fuselage_pitch1[-1], 0, 1e-3)

        assert_near_equal(alpha1[-1], 0.62446459, 1e-3)

        assert_near_equal(weight0[0], 175100, 1e-5)
        assert_near_equal(weight1[-1], 174879.40581381, 1e-5)

        assert_near_equal(time0[0], 0, 1e-3)
        assert_near_equal(time1[-1], 50.5763888, 1e-3)


if __name__ == "__main__":
    unittest.main()
