"""
Sizing the N3CC using the level 3 API.

Includes:
  Takeoff, Climb, Cruise, Descent, Landing
  Computed Aero
  N3CC data
"""
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.testing_utils import require_pyoptsparse

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.N3CC.phase_info import phase_info
from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal
from aviary.validation_cases.benchmark_utils import \
    compare_against_expected_values
from aviary.variable_info.variables import Mission


# benchmark for simple sizing problem on the N3CC
def run_trajectory(sim=True):
    prob = AviaryProblem()

    # load_inputs needs to be updated to accept an already existing aviary options
    prob.load_inputs(
        "models/N3CC/N3CC_FLOPS.csv",
        phase_info,
        # engine_builders=[engine_model],
    )

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    takeoff_fuel_burned = 577  # lbm TODO: where should this get connected from?
    takeoff_thrust_per_eng = 24555.5  # lbf TODO: where should this get connected from?
    takeoff_L_over_D = 17.35  # TODO: should this come from aero?

    prob.aviary_inputs.set_val(Mission.Takeoff.FUEL_SIMPLE,
                               takeoff_fuel_burned, units='lbm')
    prob.aviary_inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG,
                               takeoff_L_over_D, units="unitless")
    prob.aviary_inputs.set_val(Mission.Design.THRUST_TAKEOFF_PER_ENG,
                               takeoff_thrust_per_eng, units='lbf')

    prob.check_and_preprocess_inputs()
    prob.add_pre_mission_systems()
    prob.add_phases()
    prob.add_post_mission_systems()
    prob.link_phases()
    prob.add_driver("SNOPT", max_iter=50, verbosity=1)

    ##########################
    # Design Variables       #
    ##########################
    prob.add_design_variables()

    # Nudge it a bit off the correct answer to verify that the optimize takes us there.
    prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 135000.0, units='lbm')

    ##########################
    # Add Objective Function #
    ##########################
    prob.add_objective()

    ############################################
    # Initial Settings for States and Controls #
    ############################################
    prob.setup()
    prob.set_initial_guesses()
    prob.run_aviary_problem("dymos_solution.db")

    om.n2(prob)

    return prob


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test sizing using N3CC data.
    """

    @require_pyoptsparse(optimizer="SNOPT")
    def bench_test_sizing_N3CC(self):

        prob = run_trajectory(sim=False)

        times_climb = prob.get_val('traj.climb.timeseries.time', units='s')
        thrusts_climb = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')
        times_cruise = prob.get_val('traj.cruise.timeseries.time', units='s')
        thrusts_cruise = prob.get_val(
            'traj.cruise.timeseries.thrust_net_total', units='N')
        times_descent = prob.get_val('traj.descent.timeseries.time', units='s')
        thrusts_descent = prob.get_val(
            'traj.descent.timeseries.thrust_net_total', units='N')

        print(thrusts_climb)
        print(thrusts_cruise)
        print(thrusts_descent)

        expected_times_s_climb = [[120.], [163.76268451], [224.14625705],
                                  [243.2574513], [243.2574513], [336.40804357],
                                  [464.9368486], [505.61577594], [505.61577594],
                                  [626.46954383], [793.22307692], [846.],
                                  [846.], [966.85376789], [1133.60730098],
                                  [1186.38422406], [1186.38422406], [1279.53481633],
                                  [1408.06362136], [1448.7425487], [1448.7425487],
                                  [1492.50523321], [1552.88880575], [1572.]]

        expected_altitudes_m_climb = [[0.], [321.52914489], [765.17373981],
                                      [905.58573725], [905.58573725], [1589.97314657],
                                      [2534.28808598], [2833.16053563], [2833.16053563],
                                      [3721.08615262], [4946.24227588], [5334.],
                                      [5334.], [6221.92561699], [7447.08174026],
                                      [7834.83946437], [7834.83946437], [8519.22687369],
                                      [9463.54181311], [9762.41426275], [9762.41426275],
                                      [10083.94340764], [10527.58800256], [10668.]]
        expected_masses_kg_climb = [[58331.64520977], [58289.14326023], [58210.1552262],
                                    [58183.99841695], [58183.99841695], [58064.21371439],
                                    [57928.02594919], [57893.85802859], [57893.85802859],
                                    [57803.5753272], [57701.17154376], [57671.70469428],
                                    [57671.70469428], [57609.99954455], [57532.67046087],
                                    [57509.28816794], [57509.28816794], [57468.36184014],
                                    [57411.39726843], [57392.97463799], [57392.97463799],
                                    [57372.80054331], [57344.30023996], [57335.11578186]]
        expected_distances_m_climb = [[1453.24648698], [4541.59528274], [9218.32974593],
                                      [10798.05432237], [10798.05432237], [19175.48832752],
                                      [32552.21767508], [37217.45114105], [37217.45114105],
                                      [52277.75199853], [75940.08052044], [84108.7210583],
                                      [84108.7210583], [104013.98891024], [134150.12359132],
                                      [144315.97475612], [144315.97475612], [
                                          162975.88568142],
                                      [190190.53671538], [199150.13943988], [
                                          199150.13943988],
                                      [208971.10963721], [222827.7518861], [227286.29149481]]
        expected_velocities_ms_climb = [[77.19291754], [132.35228283], [162.28279625],
                                        [166.11250634], [166.11250634], [176.30796789],
                                        [178.55049791], [178.67862048], [178.67862048],
                                        [181.75616771], [188.37687907], [189.78671051],
                                        [189.78671051], [190.67789763], [191.79702196],
                                        [193.51306342], [193.51306342], [199.28763405],
                                        [212.62699415], [218.00379223], [218.00379223],
                                        [224.21350812], [232.18822869], [234.25795132]]
        expected_thrusts_N_climb = [[82387.90445676], [105138.56964482], [113204.81413537],
                                    [110631.69881503], [110631.69881503], [97595.73505557],
                                    [74672.35033284], [68787.54639919], [68787.54639919],
                                    [56537.02658008], [46486.74146611], [43988.99250831],
                                    [43988.99250831], [39814.93772493], [36454.43040965],
                                    [35877.85248136], [35877.85248136], [35222.20684522],
                                    [34829.78305946], [34857.84665827], [34857.84665827],
                                    [34988.40432301], [35243.28462529], [35286.14075168]]

        expected_times_s_cruise = [[1572.01903685], [10224.85688577],
                                   [22164.00704809], [25942.70725365]]
        expected_altitudes_m_cruise = [[10668.], [10668.],
                                       [10668.], [10668.]]
        expected_masses_kg_cruise = [[57335.11578186], [53895.3524649],
                                     [49306.34176818], [47887.72131688]]
        expected_distances_m_cruise = [[1572.0], [10224.87753766],
                                       [22164.08246234], [25942.8]]

        expected_velocities_ms_cruise = [[234.25795132], [234.25795132],
                                         [234.25795132], [234.25795132]]
        expected_thrusts_N_cruise = [[28998.46944214], [28027.44677784],
                                     [26853.54343662], [26522.10071819]]

        expected_times_s_descent = [[25942.8], [25979.38684893], [26029.86923298],
                                    [26045.84673492], [26045.84673492], [26120.56747962],
                                    [26223.66685657], [26256.29745687], [26256.29745687],
                                    [26345.13302939], [26467.70798786], [26506.50254313],
                                    [26506.50254313], [26581.22328782], [26684.32266477],
                                    [26716.95326508], [26716.95326508], [26753.54011401],
                                    [26804.02249805], [26820.0]]
        expected_altitudes_m_descent = [[10668.], [10223.49681269], [9610.1731386],
                                        [9416.05829274], [9416.05829274], [8508.25644201],
                                        [7255.67517298], [6859.237484], [6859.237484],
                                        [5779.95090202], [4290.75570439], [3819.430516],
                                        [3819.430516], [2911.62866527], [1659.04739624],
                                        [1262.60970726], [1262.60970726], [818.10651995],
                                        [204.78284585], [10.668]]
        expected_masses_kg_descent = [[47887.72131688], [47887.72131688], [47887.72131688],
                                      [47887.72131688], [47887.72131688], [47887.72131688],
                                      [47887.72131688], [47887.72131688], [47887.72131688],
                                      [47887.87994829], [47886.14050369], [47884.40804261],
                                      [47884.40804261], [47872.68009732], [47849.34258173],
                                      [47842.09391697], [47842.09391697], [47833.150133],
                                      [47820.60083267], [47816.69389115]]
        expected_distances_m_descent = [[5937855.75657951], [5946333.90423671],
                                        [5957754.05343732], [5961300.12527496],
                                        [5961300.12527496], [5977437.03841276],
                                        [5998456.22230278], [6004797.661059],
                                        [6004797.661059], [6021279.17813816],
                                        [6042075.9952574], [6048172.56585825],
                                        [6048172.56585825], [6059237.47650301],
                                        [6073001.57667498], [6076985.65080086],
                                        [6076985.65080086], [6081235.7432792],
                                        [6086718.20915512], [6088360.09504693]]
        expected_velocities_ms_descent = [[234.25795132], [197.64415171], [182.5029101],
                                          [181.15994177], [181.15994177], [172.42254637],
                                          [156.92424445], [152.68023428], [152.68023428],
                                          [145.15327267], [141.83129659], [141.72294853],
                                          [141.72294853], [141.82174008], [140.7384589],
                                          [139.65952688], [139.65952688], [136.46721905],
                                          [115.68627038], [102.07377559]]
        expected_thrusts_N_descent = [[0.], [8.11038373e-13], [4.89024663e-13],
                                      [-3.80747087e-14], [0.], [1.10831527e-12],
                                      [7.77983996e-13], [-6.64041439e-16], [0.],
                                      [0.], [-5.7108371e-14], [-8.28280737e-14],
                                      [0.], [1.14382967e-13], [5.36023712e-14],
                                      [-3.04972888e-14], [0.], [-1.0682523e-13],
                                      [-6.30544276e-14], [5.37050667e-15]]
        expected_times_s_climb = np.array(expected_times_s_climb)
        expected_altitudes_m_climb = np.array(expected_altitudes_m_climb)
        expected_masses_kg_climb = np.array(expected_masses_kg_climb)
        expected_distances_m_climb = np.array(expected_distances_m_climb)
        expected_velocities_ms_climb = np.array(expected_velocities_ms_climb)
        expected_thrusts_N_climb = np.array(expected_thrusts_N_climb)

        expected_times_s_cruise = np.array(expected_times_s_cruise)
        expected_altitudes_m_cruise = np.array(expected_altitudes_m_cruise)
        expected_masses_kg_cruise = np.array(expected_masses_kg_cruise)
        expected_distances_m_cruise = np.array(expected_distances_m_cruise)
        expected_velocities_ms_cruise = np.array(expected_velocities_ms_cruise)
        expected_thrusts_N_cruise = np.array(expected_thrusts_N_cruise)

        expected_times_s_descent = np.array(expected_times_s_descent)
        expected_altitudes_m_descent = np.array(expected_altitudes_m_descent)
        expected_masses_kg_descent = np.array(expected_masses_kg_descent)
        expected_distances_m_descent = np.array(expected_distances_m_descent)
        expected_velocities_ms_descent = np.array(expected_velocities_ms_descent)
        expected_thrusts_N_descent = np.array(expected_thrusts_N_descent)

        expected_dict = {}
        expected_dict['times'] = np.concatenate((expected_times_s_climb,
                                                 expected_times_s_cruise,
                                                 expected_times_s_descent))
        expected_dict['altitudes'] = np.concatenate((expected_altitudes_m_climb,
                                                     expected_altitudes_m_cruise,
                                                     expected_altitudes_m_descent))
        expected_dict['masses'] = np.concatenate((expected_masses_kg_climb,
                                                  expected_masses_kg_cruise,
                                                  expected_masses_kg_descent))
        expected_dict['ranges'] = np.concatenate((expected_distances_m_climb,
                                                  expected_distances_m_cruise,
                                                  expected_distances_m_descent))
        expected_dict['velocities'] = np.concatenate((expected_velocities_ms_climb,
                                                      expected_velocities_ms_cruise,
                                                      expected_velocities_ms_descent))
        self.expected_dict = expected_dict

        # Check Objective and other key variables to a reasonably tight tolerance.

        # TODO: update truth values once everyone is using latest Dymos
        rtol = 5e-2

        # Check mission values.

        # NOTE rtol = 0.05 = 5% different from truth (first timeseries)
        #      atol = 2 = no more than +/-2 meter/second/kg difference between values
        #      atol_altitude - 30 ft. There is occasional time-shifting with the N3CC
        #                      model during climb and descent so we need a looser
        #                      absolute tolerance for the points near the ground.
        rtol = .05
        atol = 2.0

        # FLIGHT PATH
        # CLIMB
        warn_timeseries_near_equal(
            times_climb, thrusts_climb, expected_times_s_climb,
            expected_thrusts_N_climb, abs_tolerance=atol, rel_tolerance=rtol)

        # CRUISE
        warn_timeseries_near_equal(
            times_cruise, thrusts_cruise, expected_times_s_cruise,
            expected_thrusts_N_cruise, abs_tolerance=atol, rel_tolerance=rtol)

        # DESCENT
        warn_timeseries_near_equal(
            times_descent, thrusts_descent, expected_times_s_descent,
            expected_thrusts_N_descent, abs_tolerance=atol, rel_tolerance=rtol)

        compare_against_expected_values(prob, self.expected_dict)


if __name__ == '__main__':
    z = ProblemPhaseTestCase()
    z.bench_test_sizing_N3CC()
