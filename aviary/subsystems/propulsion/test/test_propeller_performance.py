import unittest

import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from dymos.models.atmosphere import USatm1976Comp

from aviary.constants import TSLS_DEGR
from aviary.variable_info.variables import Aircraft
from aviary.subsystems.propulsion.propeller.propeller_performance import (
    PropellerPerformance, TipSpeedLimit,
)
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.options import get_option_defaults

# Setting up truth values from GASP
# test values now are slightly different due to setup - max tip speed was limited to test
# that it is being properly constrained (and that derivitives work across constraints)
# CT = np.array([0.27651, 0.20518, 0.13093, 0.10236, 0.10236, 0.19331,
#                0.10189, 0.10189, 0.18123, 0.08523, 0.06463, 0.02800])
CT = np.array(
    [
        0.27651,
        0.20518,
        0.13093,
        0.09694,
        0.09694,
        0.19331,
        0.10189,
        0.10189,
        0.18123,
        0.08523,
        0.06463,
        0.02800,
        0.27651,
        0.20518,
        0.13093,
    ]
)
XFT = np.array(
    [1.0, 1.0, 0.9976, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9976]
)
# CTX = np.array([0.27651, 0.20518, 0.13062, 0.10236, 0.10236, 0.19331,
#                0.10189, 0.10189, 0.18123, 0.08523, 0.06463, 0.02800])
CTX = np.array(
    [
        0.27651,
        0.20518,
        0.13062,
        0.09694,
        0.09693,
        0.19331,
        0.10189,
        0.10189,
        0.18123,
        0.08523,
        0.06463,
        0.02800,
        0.27651,
        0.20518,
        0.13062,
    ]
)
# three_quart_blade_angle = np.array(
#     [25.17, 29.67, 44.23, 31.94, 31.94, 17.44, 33.43, 33.43, 20.08, 30.28, 29.50, 28.10])
three_quart_blade_angle = np.array(
    [
        25.17,
        29.67,
        44.23,
        31.095,
        31.095,
        17.44,
        33.43,
        33.43,
        20.08,
        30.28,
        29.50,
        28.10,
        25.17,
        29.67,
        44.23,
    ]
)
# thrust = np.array([4634.8, 3415.9, 841.5, 1474.3, 1400.6, 3923.5,
#                    1467.6, 1394.2, 3678.3, 1210.4, 917.8, 397.7])
thrust = np.array(
    [
        4634.8,
        3415.9,
        841.5,
        1470.5,
        1397.0,
        3923.5,
        1467.6,
        1394.2,
        3678.3,
        1210.4,
        917.8,
        397.7,
        4634.8,
        3415.9,
        841.5,
    ]
)
# prop_eff = np.array([0.00078, 0.72352, 0.89202, 0.90586, 0.90586, 0.50750,
#                      0.90172, 0.90172, 0.47579, 0.83809, 0.76259, 0.49565])
prop_eff = np.array(
    [
        0.00078,
        0.72354,
        0.8865,
        0.90350,
        0.90350,
        0.50750,
        0.90172,
        0.90172,
        0.47579,
        0.83809,
        0.76259,
        0.49565,
        0.00078,
        0.72354,
        0.8865,
    ]
)
install_loss = np.array(
    [
        0.0133,
        0.02,
        0.034,
        0.0,
        0.05,
        0.05,
        0.0,
        0.05,
        0.05,
        0.0140,
        0.0140,
        0.0140,
        0.0133,
        0.02,
        0.034,
    ]
)
# install_eff = np.array([0.00077, 0.70904, 0.86171, 0.90586, 0.86056, 0.48213,
#                         0.90172, 0.85664, 0.45200, 0.82635, 0.75190, 0.48871])
install_eff = np.array(
    [
        0.00077,
        0.70904,
        0.86171,
        0.90350,
        0.85833,
        0.48213,
        0.90172,
        0.85664,
        0.45200,
        0.82635,
        0.75190,
        0.48871,
        0.00077,
        0.70904,
        0.86171,
    ]
)


class PropellerPerformanceTest(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.COMPUTE_PROPELLER_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES, val=4, units='unitless')

        prob = om.Problem()

        num_nodes = 3
        prob.model.add_subsystem(
            name='atmosphere',
            subsys=USatm1976Comp(num_nodes=num_nodes),
            promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                ('sos', Dynamic.Mission.SPEED_OF_SOUND),
                ('rho', Dynamic.Mission.DENSITY),
                ('temp', Dynamic.Mission.TEMPERATURE),
                ('pres', Dynamic.Mission.STATIC_PRESSURE),
            ],
        )

        prob.model.add_subsystem(
            'compute_mach',
            om.ExecComp(
                f'{Dynamic.Mission.MACH} = 0.00150933 * {Dynamic.Mission.VELOCITY} * ({TSLS_DEGR} / {Dynamic.Mission.TEMPERATURE})**0.5',
                mach={'units': 'unitless', 'val': np.zeros(num_nodes)},
                velocity={'units': 'knot', 'val': np.zeros(num_nodes)},
                temperature={'units': 'degR', 'val': np.zeros(num_nodes)},
                has_diag_partials=True,
            ),
            promotes=['*'],
        )

        pp = prob.model.add_subsystem(
            'pp',
            PropellerPerformance(num_nodes=num_nodes, aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=["*"],
        )

        pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        pp.set_input_defaults(
            Dynamic.Mission.PROPELLER_TIP_SPEED, 800 * np.ones(num_nodes), units="ft/s"
        )
        pp.set_input_defaults(
            Dynamic.Mission.VELOCITY, 100.0 * np.ones(num_nodes), units="knot"
        )
        num_blades = 4
        options.set_val(
            Aircraft.Engine.NUM_PROPELLER_BLADES, val=num_blades, units='unitless'
        )
        options.set_val(
            Aircraft.Engine.COMPUTE_PROPELLER_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        prob.setup()

        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless")
        prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.8875, units='ft')

        self.prob = prob
        self.options = options

    def compare_results(self, case_idx_begin, case_idx_end):
        p = self.prob
        cthr = p.get_val('thrust_coefficient')
        ctlf = p.get_val('comp_tip_loss_factor')
        tccl = p.get_val('thrust_coefficient_comp_loss')
        angb = p.get_val('blade_angle')
        thrt = p.get_val(Dynamic.Mission.THRUST)
        peff = p.get_val('propeller_efficiency')
        lfac = p.get_val('install_loss_factor')
        ieff = p.get_val('install_efficiency')

        tol = 5e-3

        for case_idx in range(case_idx_begin, case_idx_end):
            idx = case_idx - case_idx_begin
            assert_near_equal(cthr[idx], CT[case_idx], tolerance=tol)
            assert_near_equal(ctlf[idx], XFT[case_idx], tolerance=tol)
            assert_near_equal(tccl[idx], CTX[case_idx], tolerance=tol)
            assert_near_equal(
                angb[idx], three_quart_blade_angle[case_idx], tolerance=tol
            )
            assert_near_equal(thrt[idx], thrust[case_idx], tolerance=tol)
            assert_near_equal(peff[idx], prop_eff[case_idx], tolerance=tol)
            assert_near_equal(lfac[idx], install_loss[case_idx], tolerance=tol)
            assert_near_equal(ieff[idx], install_eff[case_idx], tolerance=tol)

    def test_case_0_1_2(self):
        # Case 0, 1, 2, to test installation loss factor computation.
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.0, 25000.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [0.10, 125.0, 300.0], units="knot")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1850.0, 1850.0, 900.0], units="hp")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_MACH_MAX, 1.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 800.0, units="ft/s")

        prob.run_model()
        self.compare_results(case_idx_begin=0, case_idx_end=2)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method="fd",
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=["*atmosphere*"],
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

    def test_case_3_4_5(self):
        # Case 3, 4, 5, to test normal cases.
        prob = self.prob
        options = self.options

        options.set_val(
            Aircraft.Engine.COMPUTE_PROPELLER_INSTALLATION_LOSS,
            val=False,
            units='unitless',
        )
        prob.setup()
        prob.set_val('install_loss_factor', [0.0, 0.05, 0.05], units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units="knot")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units="hp")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 769.70, units="ft/s")

        prob.run_model()
        self.compare_results(case_idx_begin=3, case_idx_end=5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method="fd",
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=["*atmosphere*"],
        )
        assert_check_partials(partial_data, atol=1.5e-4, rtol=1e-4)

    def test_case_6_7_8(self):
        # Case 6, 7, 8, to test odd number of blades.
        prob = self.prob
        options = self.options

        num_blades = 3
        options.set_val(
            Aircraft.Engine.NUM_PROPELLER_BLADES, val=num_blades, units='unitless'
        )
        options.set_val(
            Aircraft.Engine.COMPUTE_PROPELLER_INSTALLATION_LOSS,
            val=False,
            units='unitless',
        )
        prob.setup()
        prob.set_val('install_loss_factor', [0.0, 0.05, 0.05], units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT, 0.5, units="unitless"
        )
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units="knot")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units="hp")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 750.0, units="ft/s")

        prob.run_model()
        self.compare_results(case_idx_begin=6, case_idx_end=8)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method="fd",
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=["*atmosphere*"],
        )
        assert_check_partials(partial_data, atol=1e-4, rtol=1e-4)

    def test_case_9_10_11(self):
        # Case 9, 10, 11, to test CLI > 0.5
        prob = self.prob
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
            0.65,
            units="unitless",
        )
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 10000.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 200.0], units="knot")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [900.0, 750.0, 500.0], units="hp")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 750.0, units="ft/s")

        prob.run_model()
        self.compare_results(case_idx_begin=9, case_idx_end=11)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method="fd",
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=["*atmosphere*"],
        )
        # remove partial derivative of 'comp_tip_loss_factor' with respect to
        # 'aircraft:engine:propeller_integrated_lift_coefficient' from assert_check_partials
        partial_data_hs = partial_data['pp.hamilton_standard']
        key_pair = (
            'comp_tip_loss_factor',
            'aircraft:engine:propeller_integrated_lift_coefficient',
        )
        del partial_data_hs[key_pair]
        assert_check_partials(partial_data, atol=1.5e-3, rtol=1e-4)

    def test_case_12_13_14(self):
        # Case 12, 13, 14, to test mach limited tip speed.
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.0, 25000.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [0.10, 125.0, 300.0], units="knot")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1850.0, 1850.0, 900.0], units="hp")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_MACH_MAX, 0.8, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, 800.0, units="ft/s")

        prob.run_model()
        self.compare_results(case_idx_begin=12, case_idx_end=13)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method="fd",
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=["*atmosphere*"],
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)


class TipSpeedLimitTest(unittest.TestCase):
    def test_tipspeed(self):
        tol = 1e-5

        prob = om.Problem()
        prob.model.add_subsystem(
            "group",
            TipSpeedLimit(num_nodes=3),
            promotes=["*"],
        )
        prob.setup()
        prob.set_val(Dynamic.Mission.VELOCITY,
                     val=[0.16878, 210.97623, 506.34296], units='ft/s')
        prob.set_val(Dynamic.Mission.SPEED_OF_SOUND,
                     val=[1116.42671, 1116.42671, 1015.95467], units='ft/s')
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_MACH_MAX, val=[0.8], units='unitless')
        prob.set_val(Aircraft.Engine.PROPELLER_TIP_SPEED_MAX, val=[800], units='ft/s')
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, val=[10.5], units='ft')

        prob.run_model()

        tip_speed = prob.get_val(Dynamic.Mission.PROPELLER_TIP_SPEED, units='ft/s')
        rpm = prob.get_val('rpm', units='rpm')
        assert_near_equal(tip_speed, [800, 800, 635.7686], tolerance=tol)
        assert_near_equal(rpm, [1455.1309, 1455.1309, 1156.4082], tolerance=tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method="fd",
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
