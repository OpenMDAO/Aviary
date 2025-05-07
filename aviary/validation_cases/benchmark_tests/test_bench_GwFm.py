"""
Notes
-----
Includes:
Takeoff, Climb, Cruise, Descent, Landing
Computed Aero
Large Single Aisle 1 data
"""

import unittest

import numpy as np
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.interface.methods_for_level1 import run_aviary
from aviary.validation_cases.benchmark_utils import compare_against_expected_values


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a large single aisle commercial transport aircraft using
    GASP mass method and HEIGHT_ENERGY mission method. Expected outputs
    based on 'models/test_aircraft/aircraft_for_bench_GwFm.csv' model.
    """

    def setUp(self):
        expected_dict = {}
        # block auto-formatting of tables
        # fmt: off
        expected_dict['times'] = np.array(
            [
                [0.0], [55.733689551933274], [132.63480572667558], [156.97374606550738],
                [156.97374606550738], [275.60509258726296], [439.29213778700404],
                [491.0985279348842], [491.0985279348842], [645.0110782250607],
                [857.378983629686], [924.5926995890327], [924.5926995890327],
                [1078.505249879209], [1290.8731552838344], [1358.0868712431811],
                [1358.0868712431811], [1476.7182177649368], [1640.4052629646778],
                [1692.211653112558], [1692.211653112558], [1747.9453426644914],
                [1824.8464588392337], [1849.1853991780654], [1849.1853991780654],
                [10330.947200407132], [22034.04733403507], [25738.038512583124],
                [25738.038512583124], [25812.86796688032], [25916.117340902812],
                [25948.795414763863], [25948.795414763863], [26101.61842940165],
                [26312.482997983352], [26379.22091297826], [26379.22091297826],
                [26560.912356969104], [26811.609465928253], [26890.954243762288],
                [26890.954243762288], [27043.777258400078], [27254.64182698178],
                [27321.379741976685], [27321.379741976685], [27396.20919627388],
                [27499.458570296378], [27532.136644157425]
            ]

        )

        expected_dict['altitudes'] = np.array(
            [
                [10.668000000000001], [235.4160647585811], [1280.7577161762033],
                [1789.2240867373423], [1789.2240867373423], [4166.117812322907],
                [6429.223292405268], [6899.915227304244], [6899.9152273042455],
                [8008.2327449168815], [9160.939977223055], [9434.801587464375],
                [9434.801587464375], [9935.29924394527], [10374.555901691507],
                [10452.8629164997], [10452.8629164997], [10548.73019817695],
                [10626.562722425131], [10638.043597267682], [10638.04359726768],
                [10647.591875522832], [10662.732889513021], [10668.0], [10668.0],
                [10668.0], [10668.0], [10668.0], [10668.0], [10626.3697117778],
                [10459.417512771737], [10380.121892016292], [10380.121892016292],
                [9838.94107534588], [8630.540312914369], [8136.5517903851705],
                [8136.551790385171], [6731.689477098073], [4835.959501956426],
                [4246.281358548518], [4246.281358548521], [3120.32996645691],
                [1583.2631997282303], [1100.7768687071518], [1100.7768687071516],
                [633.0094751176282], [224.47289606969227], [152.4000000000001]
            ]
        )

        expected_dict['masses'] = np.array(
            [
                [77818.91890875385], [77688.27501012207], [77524.23950674829],
                [77468.76019046773], [77468.76019046773], [77239.22039657168],
                [76990.73913506811], [76926.80036671055], [76926.80036671055], 
                [76754.33897227133], [76543.86405635109], [76483.7192616358],
                [76483.7192616358], [76354.17352305444], [76191.31940576647],
                [76143.67243567819], [76143.67243567819], [76062.24259563042],
                [75953.47376577006], [75919.91738782128], [75919.91738782128],
                [75883.75438725074], [75832.8423125292], [75816.45141024028],
                [75816.45141024028], [70515.89266171536], [63632.18566842271],
                [61543.65557306416], [61543.65557306416], [61512.493641635716],
                [61475.15980565045], [61464.41311500414], [61464.41311500414],
                [61421.476775837866], [61381.46504437594], [61373.33572196399],
                [61373.33572196399], [61354.67797641075], [61326.43296400884],
                [61316.4035933201], [61316.4035933201], [61294.97892209769],
                [61260.38645390502], [61247.776381418495], [61247.776381418495],
                [61232.816693577166], [61211.323048859755], [61204.500051063435]
            ]
        )

        expected_dict['ranges'] = np.array(
            [
                [1410.7246267645853], [7495.3855075223], [20669.444649304616],
                [25331.75944015018], [25331.75944015018], [48156.17760692068],
                [80276.4395807753], [90751.64747645857], [90751.64747645857],
                [123003.73379975135], [169609.2345943904], [184688.57605961812],
                [184688.57605961812], [219574.63609561243], [268234.4654645649],
                [283700.48601368495], [283700.48601368495], [311021.12460668734],
                [348717.6854537568], [360638.7338207807], [360638.7338207807],
                [373458.75114105834], [391178.95399686357], [396804.3683204999],
                [396804.3683204999], [2359117.1786281476], [5066704.90400833],
                [5923646.330821189], [5923646.330821189], [5940603.245546197],
                [5962922.359248149], [5969751.264817894], [5969751.264817894],
                [6000362.008167493], [6039449.319180842], [6051171.448664183],
                [6051171.448664183], [6081755.279179599], [6121391.20765825],
                [6133468.632684376], [6133468.632684376], [6156271.164261458],
                [6186727.665526107], [6196116.484579935], [6196116.484579935],
                [6206400.38654244], [6219220.724162429], [6222720.0]
            ]
        )

        expected_dict['velocities'] = np.array(
            [
                [70.41872863051675], [142.8426782139169], [190.85388104256134],
                [193.53336168014812], [193.5333616801481], [193.97130207062534],
                [200.66615582138846], [204.29712966333042], [204.2971296633304],
                [214.52894401730563], [223.53071066321067], [225.23147649896515],
                [225.23147649896515], [227.9612963574697], [229.9799417501587],
                [230.21437887976137], [230.21437887976137], [230.34923495762445],
                [230.1753140630945], [230.03381353938522], [230.0338135393852],
                [230.08808978229095], [230.9140856209351], [231.35686943957475],
                [231.35686943957478], [231.3566224727587], [231.35628170895902],
                [231.35617385838154], [231.35617385838154], [222.01301741383028],
                [210.6244396832895], [207.37442023458655], [207.3744202345866],
                [193.625244800323], [177.92188888735785], [173.6922435472287],
                [173.69224354722866], [163.6940205640424], [153.58581070698628],
                [151.343324702535], [151.343324702535], [147.41604989710945],
                [141.78284140393103], [139.9518515787329], [139.95185157873289],
                [133.84033631181276], [112.04151628392903], [101.91044471952507]
            ]
        )
        # fmt: on

        self.expected_dict = expected_dict

        phase_info = {
            'pre_mission': {'include_takeoff': True, 'optimize_mass': True},
            'climb': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'fix_initial': False,
                    'input_initial': True,
                    'optimize_mach': True,
                    'optimize_altitude': True,
                    'use_polynomial_control': False,
                    'num_segments': 6,
                    'order': 3,
                    'solve_for_distance': False,
                    'initial_mach': (0.2, 'unitless'),
                    'final_mach': (0.79, 'unitless'),
                    'mach_bounds': ((0.1, 0.8), 'unitless'),
                    'initial_altitude': (0.0, 'ft'),
                    'final_altitude': (35000.0, 'ft'),
                    'altitude_bounds': ((0.0, 36000.0), 'ft'),
                    'throttle_enforcement': 'path_constraint',
                    'constrain_final': False,
                    'fix_duration': False,
                    'initial_bounds': ((0.0, 0.0), 'min'),
                    'duration_bounds': ((5.0, 50.0), 'min'),
                    'no_descent': True,
                    'add_initial_mass_constraint': False,
                },
                'initial_guesses': {'time': ([0, 40.0], 'min')},
            },
            'cruise': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'optimize_mach': True,
                    'optimize_altitude': True,
                    'polynomial_control_order': 1,
                    'use_polynomial_control': True,
                    'num_segments': 1,
                    'order': 3,
                    'solve_for_distance': False,
                    'initial_mach': (0.79, 'unitless'),
                    'final_mach': (0.79, 'unitless'),
                    'mach_bounds': ((0.78, 0.8), 'unitless'),
                    'initial_altitude': (35000.0, 'ft'),
                    'final_altitude': (35000.0, 'ft'),
                    'altitude_bounds': ((35000.0, 35000.0), 'ft'),
                    'throttle_enforcement': 'boundary_constraint',
                    'fix_initial': False,
                    'constrain_final': False,
                    'fix_duration': False,
                    'initial_bounds': ((64.0, 192.0), 'min'),
                    'duration_bounds': ((60.0, 7200.0), 'min'),
                },
                'initial_guesses': {'time': ([128, 113], 'min')},
            },
            'descent': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'optimize_mach': True,
                    'optimize_altitude': True,
                    'use_polynomial_control': False,
                    'num_segments': 5,
                    'order': 3,
                    'solve_for_distance': False,
                    'initial_mach': (0.79, 'unitless'),
                    'final_mach': (0.3, 'unitless'),
                    'mach_bounds': ((0.2, 0.8), 'unitless'),
                    'initial_altitude': (35000.0, 'ft'),
                    'final_altitude': (500.0, 'ft'),
                    'altitude_bounds': ((0.0, 35000.0), 'ft'),
                    'throttle_enforcement': 'path_constraint',
                    'fix_initial': False,
                    'constrain_final': True,
                    'fix_duration': False,
                    'initial_bounds': ((120.5, 361.5), 'min'),
                    'duration_bounds': ((5.0, 30.0), 'min'),
                    'no_climb': True,
                },
                'initial_guesses': {'time': ([241, 30], 'min')},
            },
            'post_mission': {
                'include_landing': True,
                'constrain_range': True,
                'target_range': (3360.0, 'nmi'),
            },
        }

        self.phase_info = phase_info

        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_swap_1_GwFm_IPOPT(self):
        prob = run_aviary(
            'models/test_aircraft/aircraft_for_bench_GwFm.csv',
            self.phase_info,
            max_iter=100,
            optimizer='IPOPT',
            verbosity=0,
        )

        compare_against_expected_values(prob, self.expected_dict)

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_swap_1_GwFm_SNOPT(self):
        prob = run_aviary(
            'models/test_aircraft/aircraft_for_bench_GwFm.csv',
            self.phase_info,
            max_iter=50,
            optimizer='SNOPT',
            verbosity=0,
        )

        compare_against_expected_values(prob, self.expected_dict)


if __name__ == '__main__':
    test = ProblemPhaseTestCase()
    test.setUp()
    test.bench_test_swap_1_GwFm_IPOPT()
    test.bench_test_swap_1_GwFm_SNOPT()
