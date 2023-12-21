import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.gasp_based.phases.run_phases.run_ascent import run_ascent


@unittest.skip("this wasn't being run and needs to be updated")
class AccelPhaseTestCase(unittest.TestCase):
    def bench_test_ascent(self):

        prob = run_ascent(make_plots=False)

        gamma = prob.get_val(
            "traj.ascent.timeseries.states:flight_path_angle", units="deg")
        alt = prob.get_val("traj.ascent.timeseries.states:altitude", units="ft")
        TAS = prob.get_val("traj.ascent.timeseries.states:TAS", units="kn")
        distance = prob.get_val("traj.ascent.timeseries.states:distance", units="ft")
        load_factor = prob.get_val(
            "traj.ascent.timeseries.load_factor", units="unitless")
        fuselage_pitch = prob.get_val(
            "traj.ascent.timeseries.fuselage_pitch", units="deg"
        )
        alpha = prob.get_val("traj.ascent.timeseries.controls:alpha", units="deg")
        weight = prob.get_val("traj.ascent.timeseries.states:weight", units="lbm")
        time = prob.get_val("traj.ascent.timeseries.time", units="s")

        assert_near_equal(gamma[0], 11.4591559, 1e-3)
        assert_near_equal(gamma[-1], 6.12703133, 1e-3)

        assert_near_equal(alt[0], 0, 1e-4)
        assert_near_equal(alt[-1], 500, 1e-4)

        assert_near_equal(TAS[0], 153.3196491, 1e-4)
        assert_near_equal(TAS[-1], 176.63794989, 1e-4)

        assert_near_equal(distance[0], 4330.83393029, 1e-5)
        assert_near_equal(distance[-1], 8847.37626223, 1e-5)

        assert_near_equal(load_factor[0], 0.67732548, 1e-2)
        assert_near_equal(load_factor[-1], 0.84455828, 1e-2)

        assert_near_equal(fuselage_pitch[0], 15, 1e-3)
        assert_near_equal(fuselage_pitch[-1], 14.92380341, 1e-3)

        assert_near_equal(alpha[0], 3.5408441, 1e-3)
        assert_near_equal(alpha[-1], 8.79677208, 1e-3)

        assert_near_equal(weight[0], 174963.74211336, 1e-5)
        assert_near_equal(weight[-1], 174892.40464724, 1e-5)

        assert_near_equal(time[0], 31.2, 1e-3)
        assert_near_equal(time[-1], 47.5236357, 1e-3)


if __name__ == "__main__":
    unittest.main()
