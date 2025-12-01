"""
Sizing the N3CC using the level 3 API.

Includes:
  Takeoff, Climb, Cruise, Descent, Landing
  Computed Aero
  N3CC data
"""

from copy import deepcopy
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.testing_utils import require_pyoptsparse

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.advanced_single_aisle.phase_info import phase_info
from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal
from aviary.validation_cases.benchmark_utils import compare_against_expected_values
from aviary.variable_info.variables import Mission


# benchmark for simple sizing problem on the N3CC
def run_trajectory(sim=True):
    prob = AviaryProblem()
    local_phase_info = deepcopy(phase_info)

    prob.load_inputs(
        'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
        local_phase_info,
    )

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    takeoff_fuel_burned = 577  # lbm TODO: where should this get connected from?
    takeoff_thrust_per_eng = 24555.5  # lbf TODO: where should this get connected from?
    takeoff_L_over_D = 17.35  # TODO: should this come from aero?

    prob.aviary_inputs.set_val(Mission.Takeoff.FUEL_SIMPLE, takeoff_fuel_burned, units='lbm')
    prob.aviary_inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG, takeoff_L_over_D, units='unitless')
    prob.aviary_inputs.set_val(
        Mission.Design.THRUST_TAKEOFF_PER_ENG, takeoff_thrust_per_eng, units='lbf'
    )

    prob.check_and_preprocess_inputs()

    prob.build_model()
    prob.add_driver('SNOPT', max_iter=50, verbosity=1)

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
    prob.run_aviary_problem()

    return prob


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """Test sizing using N3CC data."""

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_sizing_N3CC(self):
        prob = run_trajectory(sim=False)

        # self.assertTrue(prob.result.success)

        times_climb = prob.get_val('traj.climb.timeseries.time', units='s')
        thrusts_climb = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')
        times_cruise = prob.get_val('traj.cruise.timeseries.time', units='s')
        thrusts_cruise = prob.get_val('traj.cruise.timeseries.thrust_net_total', units='N')
        times_descent = prob.get_val('traj.descent.timeseries.time', units='s')
        thrusts_descent = prob.get_val('traj.descent.timeseries.thrust_net_total', units='N')

        expected_times_s_climb = [
            [0.0],
            [84.0863228],
            [200.10828601],
            [236.82884068],
            [236.82884068],
            [415.80988031],
            [662.76718444],
            [740.92832684],
            [740.92832684],
            [973.13869172],
            [1293.54160046],
            [1394.94802559],
            [1394.94802559],
            [1627.15839048],
            [1947.56129921],
            [2048.96772435],
            [2048.96772435],
            [2227.94876397],
            [2474.90606811],
            [2553.06721051],
            [2553.06721051],
            [2637.15353331],
            [2753.17549652],
            [2789.89605118],
        ]

        expected_altitudes_m_climb = [
            [10.668],
            [1113.15892907],
            [2505.31910212],
            [2915.84612589],
            [2915.84612589],
            [4720.22329407],
            [6714.19059843],
            [7236.03048999],
            [7236.03048999],
            [8509.49393709],
            [9678.02373053],
            [9928.61760729],
            [9928.61760729],
            [10328.05698328],
            [10575.61727564],
            [10603.57950367],
            [10603.57950367],
            [10618.54858105],
            [10612.64011873],
            [10614.86229636],
            [10614.86229636],
            [10624.02992826],
            [10653.51252987],
            [10668.0],
        ]

        expected_masses_kg_climb = [
            [57336.24193681],
            [57232.33711103],
            [57115.85206902],
            [57081.31733056],
            [57081.31733056],
            [56932.11443592],
            [56757.7116076],
            [56706.5613175],
            [56706.5613175],
            [56570.14762048],
            [56411.45004403],
            [56367.35681493],
            [56367.35681493],
            [56274.55398409],
            [56159.15000551],
            [56124.14860773],
            [56124.14860773],
            [56062.52701216],
            [55974.27102613],
            [55944.69095805],
            [55944.69095805],
            [55911.32442321],
            [55861.55377337],
            [55844.61091641],
        ]

        expected_distances_m_climb = [
            [1421.8855236],
            [8910.25297478],
            [22162.18513561],
            [26969.15379381],
            [26969.15379381],
            [53889.88674169],
            [98274.91723261],
            [113493.56841],
            [113493.56841],
            [160867.47482179],
            [228949.21419049],
            [250580.7400101],
            [250580.7400101],
            [299935.51831796],
            [367183.26127212],
            [388239.83394021],
            [388239.83394021],
            [425476.85128199],
            [477670.19333243],
            [494581.98686398],
            [494581.98686398],
            [513127.92793595],
            [539457.59288457],
            [548004.08377766],
        ]

        expected_velocities_ms_climb = [
            [78.49573492],
            [101.1016394],
            [127.69880867],
            [135.09899815],
            [135.09899815],
            [164.98084757],
            [192.10475793],
            [197.92873153],
            [197.92873153],
            [209.1463892],
            [213.79712949],
            [213.57324106],
            [213.57324106],
            [211.48593005],
            [208.20820027],
            [207.83218491],
            [207.83218491],
            [208.75754552],
            [214.86690966],
            [218.31439428],
            [218.31439428],
            [222.98550489],
            [231.23769674],
            [234.32014809],
        ]

        expected_thrusts_N_climb = [
            [152151.45391014],
            [113069.49989588],
            [88010.04683775],
            [82790.00338831],
            [82790.00338831],
            [66061.94148375],
            [52924.90549389],
            [49776.7994342],
            [49776.7994342],
            [41959.36848989],
            [33969.13957345],
            [32030.53104675],
            [32030.53104675],
            [28700.73489851],
            [26384.34223423],
            [26167.73184602],
            [26167.73184602],
            [26350.18803002],
            [28010.57199427],
            [28976.37867947],
            [28976.37867947],
            [30354.62140874],
            [33256.05489502],
            [34561.27738221],
        ]

        expected_times_s_cruise = [
            [2789.89605118],
            [10842.31125131],
            [21953.00070629],
            [25469.49593054],
        ]

        expected_altitudes_m_cruise = [[10668.0], [10668.0], [10668.0], [10668.0]]

        expected_masses_kg_cruise = [
            [55844.61091641],
            [52677.35420548],
            [48437.42986223],
            [47123.79311112],
        ]

        expected_distances_m_cruise = [
            [548004.08377766],
            [2434847.2059566],
            [5038305.60443551],
            [5862291.28614165],
        ]

        expected_velocities_ms_cruise = [
            [234.32014809],
            [234.32014809],
            [234.32014809],
            [234.32014809],
        ]

        expected_thrusts_N_cruise = [
            [28597.93714764],
            [27735.30112377],
            [26683.82813953],
            [26382.05441973],
        ]

        expected_times_s_descent = [
            [25469.49593054],
            [25575.61865901],
            [25722.04636435],
            [25768.39023245],
            [25768.39023245],
            [25985.12300752],
            [26284.17000134],
            [26378.81735242],
            [26378.81735242],
            [26636.49117829],
            [26992.02846603],
            [27104.55480199],
            [27104.55480199],
            [27321.28757707],
            [27620.33457088],
            [27714.98192196],
            [27714.98192196],
            [27821.10465044],
            [27967.53235578],
            [28013.87622387],
        ]

        expected_altitudes_m_descent = [
            [10668.0],
            [10629.3077369],
            [10455.72330144],
            [10373.70672575],
            [10373.70672575],
            [9834.46607022],
            [8732.91197866],
            [8314.59862184],
            [8314.59862184],
            [7049.40049088],
            [5113.49927701],
            [4482.90035867],
            [4482.90035867],
            [3279.9486113],
            [1716.36236716],
            [1261.43596673],
            [1261.43596673],
            [782.00748378],
            [183.12312185],
            [10.668],
        ]

        expected_masses_kg_descent = [
            [47123.79311112],
            [47094.35643767],
            [47060.51112858],
            [47050.73917493],
            [47050.73917493],
            [47010.63763596],
            [46969.04641852],
            [46959.35666069],
            [46959.35666069],
            [46941.2980746],
            [46930.45302474],
            [46929.08312096],
            [46929.08312096],
            [46917.94567025],
            [46888.12549136],
            [46878.62822765],
            [46878.62822765],
            [46866.76117329],
            [46849.23898947],
            [46843.36760307],
        ]

        expected_distances_m_descent = [
            [5862291.28614165],
            [5886497.96834813],
            [5917909.4237556],
            [5927421.58774285],
            [5927421.58774285],
            [5969540.61482824],
            [6022389.68649162],
            [6038131.37179067],
            [6038131.37179067],
            [6079037.43017674],
            [6131650.9798226],
            [6147551.15911222],
            [6147551.15911222],
            [6177096.10219989],
            [6215315.97220355],
            [6226712.31501162],
            [6226712.31501162],
            [6238990.09745233],
            [6254964.28421588],
            [6259760.0],
        ]

        expected_velocities_ms_descent = [
            [234.32014809],
            [222.16608332],
            [207.49206629],
            [203.30917468],
            [203.30917468],
            [186.33195622],
            [168.69090335],
            [164.21097701],
            [164.21097701],
            [153.98766763],
            [142.81314196],
            [139.53610066],
            [139.53610066],
            [132.99120577],
            [122.26790663],
            [118.18171604],
            [118.18171604],
            [113.0709453],
            [104.9404634],
            [102.07377561],
        ]

        expected_thrusts_N_descent = [
            [20679.14374965],
            [17150.55708079],
            [14208.68058208],
            [13389.53407776],
            [13389.53407776],
            [10253.30474317],
            [6665.21263404],
            [5654.24120595],
            [5654.24120595],
            [3286.30925166],
            [1039.89484864],
            [604.78704925],
            [604.78704925],
            [152.80643718],
            [47.30677203],
            [126.66634835],
            [126.66634835],
            [357.59318988],
            [1023.64759606],
            [1324.08820659],
        ]

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
        expected_dict['times'] = np.concatenate(
            (expected_times_s_climb, expected_times_s_cruise, expected_times_s_descent)
        )
        expected_dict['altitudes'] = np.concatenate(
            (expected_altitudes_m_climb, expected_altitudes_m_cruise, expected_altitudes_m_descent)
        )
        expected_dict['masses'] = np.concatenate(
            (expected_masses_kg_climb, expected_masses_kg_cruise, expected_masses_kg_descent)
        )
        expected_dict['ranges'] = np.concatenate(
            (expected_distances_m_climb, expected_distances_m_cruise, expected_distances_m_descent)
        )
        expected_dict['velocities'] = np.concatenate(
            (
                expected_velocities_ms_climb,
                expected_velocities_ms_cruise,
                expected_velocities_ms_descent,
            )
        )
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
        rtol = 0.05
        atol = 2.0

        # FLIGHT PATH
        # CLIMB
        warn_timeseries_near_equal(
            times_climb,
            thrusts_climb,
            expected_times_s_climb,
            expected_thrusts_N_climb,
            abs_tolerance=atol,
            rel_tolerance=rtol,
        )

        # CRUISE
        warn_timeseries_near_equal(
            times_cruise,
            thrusts_cruise,
            expected_times_s_cruise,
            expected_thrusts_N_cruise,
            abs_tolerance=atol,
            rel_tolerance=rtol,
        )

        # DESCENT
        warn_timeseries_near_equal(
            times_descent,
            thrusts_descent,
            expected_times_s_descent,
            expected_thrusts_N_descent,
            abs_tolerance=atol,
            rel_tolerance=rtol,
        )

        compare_against_expected_values(prob, self.expected_dict)


if __name__ == '__main__':
    z = ProblemPhaseTestCase()
    z.bench_test_sizing_N3CC()
