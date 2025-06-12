import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.propeller.propeller_performance import (
    AdvanceRatio,
    AreaSquareRatio,
    OutMachs,
    PropellerPerformance,
    TipSpeed,
)
from aviary.variable_info.enums import OutMachType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Settings

# Setting up truth values from GASP (The first 12 are actual truth values, the rest are intelligent guesses)
# test values now are slightly different due to setup - max tip speed was limited to test
# that it is being properly constrained (and that derivatives work across constraints)
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
        0.09877,
        0.09877,
        0.18641,
    ]
)
XFT = np.array(
    [
        1.0,
        1.0,
        0.9976,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.9976,
        1.0,
        1.0,
        1.0,
    ]
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
        0.09877,
        0.09877,
        0.18641,
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
        1498.29,
        1423.38,
        3637.83130,
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
        0.92057,
        0.92057,
        0.47056,
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
        0.0,
        0.05,
        0.05,
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
        0.92057,
        0.87454,
        0.44703,
    ]
)


class PropellerPerformanceTest(unittest.TestCase):
    """Test computation of propeller performance test using Hamilton Standard model."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=4, units='unitless')
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, False)
        options.set_val(Settings.VERBOSITY, 0)

        prob = om.Problem()

        num_nodes = 3

        prob.model.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=num_nodes),
            promotes=['*'],
        )

        pp = prob.model.add_subsystem(
            'pp',
            PropellerPerformance(num_nodes=num_nodes, aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        pp.set_input_defaults(Aircraft.Engine.Propeller.DIAMETER, 10, units='ft')
        pp.set_input_defaults(
            Dynamic.Vehicle.Propulsion.PROPELLER_TIP_SPEED,
            800 * np.ones(num_nodes),
            units='ft/s',
        )
        pp.set_input_defaults(Dynamic.Mission.VELOCITY, 100.0 * np.ones(num_nodes), units='knot')
        num_blades = 4
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=num_blades, units='unitless')
        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=True,
            units='unitless',
        )

        setup_model_options(prob, options)

        prob.setup()

        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 10.5, units='ft')
        prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 114.0, units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless')
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.8875, units='ft')

        self.prob = prob
        self.options = options

    def compare_results(self, case_idx_begin, case_idx_end):
        p = self.prob
        cthr = p.get_val('thrust_coefficient')
        ctlf = p.get_val('comp_tip_loss_factor')
        tccl = p.get_val('thrust_coefficient_comp_loss')
        thrt = p.get_val(Dynamic.Vehicle.Propulsion.THRUST)
        peff = p.get_val('propeller_efficiency')
        lfac = p.get_val('install_loss_factor')
        ieff = p.get_val('install_efficiency')

        tol = 5e-3

        for case_idx in range(case_idx_begin, case_idx_end):
            idx = case_idx - case_idx_begin
            assert_near_equal(cthr[idx], CT[case_idx], tolerance=tol)
            assert_near_equal(ctlf[idx], XFT[case_idx], tolerance=tol)
            assert_near_equal(tccl[idx], CTX[case_idx], tolerance=tol)
            assert_near_equal(thrt[idx], thrust[case_idx], tolerance=tol)
            assert_near_equal(peff[idx], prop_eff[case_idx], tolerance=tol)
            assert_near_equal(lfac[idx], install_loss[case_idx], tolerance=tol)
            assert_near_equal(ieff[idx], install_eff[case_idx], tolerance=tol)

    def test_case_0_1_2(self):
        # Case 0, 1, 2, to test installation loss factor computation.
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.0, 25000.0], units='ft')
        prob.set_val(Dynamic.Mission.VELOCITY, [0.10, 125.0, 300.0], units='knot')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.RPM,
            [1455.13090827, 1455.13090827, 1455.13090827],
            units='rpm',
        )
        prob.set_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, [1850.0, 1850.0, 900.0], units='hp')
        prob.set_val(Aircraft.Engine.Propeller.TIP_MACH_MAX, 1.0, units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800.0, units='ft/s')

        prob.run_model()
        self.compare_results(case_idx_begin=0, case_idx_end=2)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=['*atmosphere*'],
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

    def test_case_3_4_5(self):
        # Case 3, 4, 5, to test normal cases.
        prob = self.prob
        options = self.options

        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=False,
            units='unitless',
        )

        setup_model_options(prob, options)

        prob.setup()
        prob.set_val('install_loss_factor', [0.0, 0.05, 0.05], units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 12.0, units='ft')
        prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 150.0, units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless')
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units='ft')
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units='knot')
        prob.set_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units='hp')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.RPM,
            [1225.02, 1225.02, 1225.02],
            units='rpm',
        )
        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 769.70, units='ft/s')

        prob.run_model()

        self.compare_results(case_idx_begin=3, case_idx_end=5)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=['*atmosphere*'],
        )
        assert_check_partials(partial_data, atol=1.5e-4, rtol=1e-4)

    def test_case_6_7_8(self):
        # Case 6, 7, 8, to test odd number of blades.
        prob = self.prob
        options = self.options

        num_blades = 3
        options.set_val(Aircraft.Engine.Propeller.NUM_BLADES, val=num_blades, units='unitless')
        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=False,
            units='unitless',
        )

        setup_model_options(prob, options)

        prob.setup()
        prob.set_val('install_loss_factor', [0.0, 0.05, 0.05], units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 12.0, units='ft')
        prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 150.0, units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT, 0.5, units='unitless')
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units='ft')
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units='knot')
        prob.set_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units='hp')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.RPM,
            [1193.66207319, 1193.66207319, 1193.66207319],
            units='rpm',
        )
        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 750.0, units='ft/s')

        prob.run_model()
        self.compare_results(case_idx_begin=6, case_idx_end=8)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=['*atmosphere*'],
        )
        assert_check_partials(partial_data, atol=1e-4, rtol=1e-4)

    def test_case_9_10_11(self):
        # Case 9, 10, 11, to test CLI > 0.5
        prob = self.prob
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 12.0, units='ft')
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.Propeller.ACTIVITY_FACTOR, 150.0, units='unitless')
        prob.set_val(
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT,
            0.65,
            units='unitless',
        )
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 10000.0], units='ft')
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 200.0], units='knot')
        prob.set_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, [900.0, 750.0, 500.0], units='hp')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.RPM,
            [1193.66207319, 1193.66207319, 1193.66207319],
            units='rpm',
        )

        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 750.0, units='ft/s')

        prob.run_model()
        self.compare_results(case_idx_begin=9, case_idx_end=11)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=['*atmosphere*'],
        )
        # remove partial derivative of 'comp_tip_loss_factor' with respect to
        # integrated lift coefficient from assert_check_partials
        partial_data_hs = partial_data['pp.hamilton_standard']
        key_pair = (
            'comp_tip_loss_factor',
            Aircraft.Engine.Propeller.INTEGRATED_LIFT_COEFFICIENT,
        )
        del partial_data_hs[key_pair]
        assert_check_partials(partial_data, atol=1.5e-3, rtol=1e-4)

    def test_case_12_13_14(self):
        # Case 12, 13, 14, to test mach limited tip speed.
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.0, 25000.0], units='ft')
        prob.set_val(Dynamic.Mission.VELOCITY, [0.10, 125.0, 300.0], units='knot')
        prob.set_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, [1850.0, 1850.0, 900.0], units='hp')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.RPM,
            [1455.1309082687574, 1455.1309082687574, 1156.4081529986502],
            units='rpm',
        )
        prob.set_val(Aircraft.Engine.Propeller.TIP_MACH_MAX, 0.8, units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 800.0, units='ft/s')

        prob.run_model()
        self.compare_results(case_idx_begin=12, case_idx_end=13)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            excludes=['*atmosphere*'],
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

    def test_case_15_16_17(self):
        # case 15, 16, 17, to test propeller map
        prob = self.prob
        options = self.options

        options.set_val(
            Aircraft.Engine.Propeller.COMPUTE_INSTALLATION_LOSS,
            val=False,
            units='unitless',
        )
        prop_file_path = 'models/engines/propellers/PropFan.prop'
        options.set_val(Aircraft.Engine.Propeller.DATA_FILE, val=prop_file_path, units='unitless')
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, val='slinear', units='unitless')

        setup_model_options(prob, options)

        prob.setup(force_alloc_complex=True)
        prob.set_val('install_loss_factor', [0.0, 0.05, 0.05], units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 12.0, units='ft')
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units='ft')
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units='knot')
        prob.set_val(Dynamic.Vehicle.Propulsion.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units='hp')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.RPM,
            [1225.0155969783186, 1225.0155969783186, 1225.0155969783186],
            units='rpm',
        )
        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, 769.70, units='ft/s')

        prob.run_model()
        self.compare_results(case_idx_begin=15, case_idx_end=17)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
            includes=['*selectedMach*'],
        )
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class OutMachsTest(unittest.TestCase):
    """
    Test the computation of OutMachs: Given two of Mach, helical Mach, and tip Mach,
    compute the other.
    """

    def test_helical_mach(self):
        # Given Mach and tip Mach, compute helical Mach.
        tol = 1e-5
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            OutMachs(num_nodes=2, output_mach_type=OutMachType.HELICAL_MACH),
            promotes=['*'],
        )
        prob.setup()
        prob.set_val('mach', val=[0.5, 0.7], units='unitless')
        prob.set_val('tip_mach', val=[0.5, 0.7], units='unitless')
        prob.run_model()
        y = prob.get_val('helical_mach')
        y_exact = np.sqrt([0.5 * 0.5 + 0.5 * 0.5, 0.7 * 0.7 + 0.7 * 0.7])

        assert_near_equal(y, y_exact, tolerance=tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=1e-4, rtol=1e-4)

    def test_mach(self):
        # Given helical Mach and tip Mach, compute Mach.
        tol = 1e-5
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            OutMachs(num_nodes=2, output_mach_type=OutMachType.MACH),
            promotes=['*'],
        )
        prob.setup()
        prob.set_val('helical_mach', val=[0.7, 0.8], units='unitless')
        prob.set_val('tip_mach', val=[0.5, 0.4], units='unitless')
        prob.run_model()
        y = prob.get_val('mach')
        y_exact = np.sqrt([0.7 * 0.7 - 0.5 * 0.5, 0.8 * 0.8 - 0.4 * 0.4])

        assert_near_equal(y, y_exact, tolerance=tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=1e-4, rtol=1e-4)

    def test_tip_mach(self):
        # Given helical Mach and Mach, compute tip Mach.
        tol = 1e-5
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            OutMachs(num_nodes=2, output_mach_type=OutMachType.TIP_MACH),
            promotes=['*'],
        )
        prob.setup()
        prob.set_val('helical_mach', val=[0.7, 0.8], units='unitless')
        prob.set_val('mach', val=[0.5, 0.4], units='unitless')
        prob.run_model()
        y = prob.get_val('tip_mach')
        y_exact = np.sqrt([0.7 * 0.7 - 0.5 * 0.5, 0.8 * 0.8 - 0.4 * 0.4])

        assert_near_equal(y, y_exact, tolerance=tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=1e-4, rtol=1e-4)


class TipSpeedLimitTest(unittest.TestCase):
    """Test computation of tip speed limit in TipSpeedLimit class."""

    def test_tipspeed(self):
        tol = 1e-5

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            TipSpeed(num_nodes=3),
            promotes=['*'],
        )
        prob.setup()
        prob.set_val(
            Dynamic.Mission.VELOCITY,
            val=[0.16878, 210.97623, 506.34296],
            units='ft/s',
        )
        prob.set_val(
            Dynamic.Atmosphere.SPEED_OF_SOUND,
            val=[1116.42671, 1116.42671, 1015.95467],
            units='ft/s',
        )
        prob.set_val(Aircraft.Engine.Propeller.TIP_MACH_MAX, val=[0.8], units='unitless')
        prob.set_val(Aircraft.Engine.Propeller.TIP_SPEED_MAX, val=[800], units='ft/s')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, val=[10.5], units='ft')

        prob.run_model()

        tip_speed = prob.get_val('propeller_tip_speed_limit', units='ft/s')
        assert_near_equal(tip_speed, [800, 800, 635.7686], tolerance=tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)


class SquareRatioTest(unittest.TestCase):
    """Test the computation of square ratio with a maximum."""

    def test_sqa_ratio_1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            AreaSquareRatio(num_nodes=2, smooth_sqa=False),
            promotes=['*'],
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('DiamNac', val=2.8875, units='ft')
        prob.set_val('DiamProp', val=10.0, units='ft')
        prob.run_model()

        sqa_ratio = prob.get_val('sqa_array', units='unitless')
        assert_near_equal(sqa_ratio, [0.08337656, 0.08337656], tolerance=1e-5)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_sqa_ratio_2(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            AreaSquareRatio(num_nodes=2, smooth_sqa=True),
            promotes=['*'],
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('DiamNac', val=2.8875, units='ft')
        prob.set_val('DiamProp', val=10.0, units='ft')
        prob.run_model()

        sqa_ratio = prob.get_val('sqa_array', units='unitless')
        assert_near_equal(sqa_ratio, [0.08337656, 0.08337656], tolerance=1e-5)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_sqa_ratio_3(self):
        """Smooth, above 0.5."""
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            AreaSquareRatio(num_nodes=2, smooth_sqa=True),
            promotes=['*'],
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('DiamNac', val=8, units='ft')
        prob.set_val('DiamProp', val=10.0, units='ft')
        prob.run_model()

        sqa_ratio = prob.get_val('sqa_array', units='unitless')
        assert_near_equal(sqa_ratio, [0.5, 0.5], tolerance=1e-5)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AdvanceRatioTest(unittest.TestCase):
    """Test the computation of advanced ratio with a maximum."""

    def test_zje_1(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            AdvanceRatio(num_nodes=4, smooth_zje=False),
            promotes=['*'],
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('vtas', val=[0.1, 125.0, 300.0, 1000.0], units='knot')
        prob.set_val('tipspd', val=[800.0, 800.0, 750.0, 500.0], units='ft/s')
        prob.set_val('sqa_array', val=[0.0756, 0.0756, 0.0756, 1.0], units='unitless')
        prob.run_model()

        equiv_adv_ratio = prob.get_val('equiv_adv_ratio', units='unitless')
        assert_near_equal(
            equiv_adv_ratio,
            [6.50074004e-04, 8.12592505e-01, 2.08023681e00, 5.0],
            tolerance=1e-5,
        )

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_zje_2(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            AdvanceRatio(num_nodes=4, smooth_zje=True),
            promotes=['*'],
        )
        prob.setup(force_alloc_complex=True)
        prob.set_val('vtas', val=[0.1, 125.0, 300.0, 1000.0], units='knot')
        prob.set_val('tipspd', val=[800.0, 800.0, 750.0, 500.0], units='ft/s')
        prob.set_val('sqa_array', val=[0.0756, 0.0756, 0.0756, 1.0], units='unitless')
        prob.run_model()

        equiv_adv_ratio = prob.get_val('equiv_adv_ratio', units='unitless')
        assert_near_equal(
            equiv_adv_ratio,
            [6.50074004e-04, 8.12592505e-01, 2.08023681e00, 5.0],
            tolerance=1e-5,
        )

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
    # test = PropellerPerformanceTest()
    # test.setUp()
    # test.test_case_3_4_5()
