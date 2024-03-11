import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.variable_info.variables import Aircraft
from aviary.subsystems.propulsion.prop_performance import PropPerf

from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.options import get_option_defaults

# Setting up truth values from GASP
CT = np.array([0.27651, 0.20518, 0.13093, 0.10236,
              0.10236, 0.19331, 0.10189, 0.10189, 0.18123])
XFT = np.array([1.0, 1.0, 0.9976, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
CTX = np.array([0.27651, 0.20518, 0.13062, 0.10236,
               0.10236, 0.19331, 0.10189, 0.10189, 0.18123])
three_quart_blade_angle = np.array(
    [25.17, 29.67, 44.23, 31.94, 31.94, 17.44, 33.43, 33.43, 20.08])
thrust = np.array([4634.8, 3415.9, 841.5, 1474.3, 1400.6,
                  3923.5, 1467.6, 1394.2, 3678.3])
prop_eff = np.array([0.0, 0.7235, 0.892, 0.9059, 0.9059, 0.5075, 0.9017, 0.9017, 0.4758])
install_loss = np.array([0.0133, 0.02, 0.034, 0.0, 0.05, 0.05, 0.0, 0.05, 0.05])
install_eff = np.array([0.0, 0.709, 0.8617, 0.9059, 0.8606,
                       0.4821, 0.9017, 0.8566, 0.452])


class PropPerformanceTest(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=4, units='unitless')

        prob = om.Problem()
        num_nodes = 3
        pp = prob.model.add_subsystem(
            'pp',
            PropPerf(num_nodes=num_nodes, aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=["*"],
        )

        pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        pp.set_input_defaults(Dynamic.Mission.PROPELLER_TIP_SPEED, 800*np.ones(num_nodes), units="ft/s")
        pp.set_input_defaults(Dynamic.Mission.VELOCITY, 0*np.ones(num_nodes), units="knot")
        num_blades = 4
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=num_blades, units='unitless')
        # pp.options.set(compute_installation_loss=True)
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        prob.setup()

        print()
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.275, units="unitless")
        prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.8875, units='ft')

        self.prob = prob
        self.options = options

    def compare_results(self, case_idx_begin, case_idx_end):
        p = self.prob
        cthr = p.get_val('thrust_coefficient')
        ctlf = p.get_val('comp_tip_loss_factor')
        tccl = p.get_val('thrust_coefficient_comp_loss')
        angb = p.get_val('ang_blade')
        thrt = p.get_val('Thrust')
        peff = p.get_val('propeller_efficiency')
        lfac = p.get_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR)
        ieff = p.get_val('install_efficiency')

        tol = 0.005

        for case_idx in range(case_idx_begin, case_idx_end):
            assert_near_equal(cthr[case_idx - case_idx_begin], CT[case_idx], tolerance=tol)
            assert_near_equal(ctlf[case_idx - case_idx_begin], XFT[case_idx], tolerance=tol)
            assert_near_equal(tccl[case_idx - case_idx_begin], CTX[case_idx], tolerance=tol)
            assert_near_equal(angb[case_idx - case_idx_begin], three_quart_blade_angle[case_idx], tolerance=tol)
            assert_near_equal(thrt[case_idx - case_idx_begin], thrust[case_idx], tolerance=tol)
            assert_near_equal(peff[case_idx - case_idx_begin], prop_eff[case_idx], tolerance=tol)
            assert_near_equal(lfac[case_idx - case_idx_begin], install_loss[case_idx], tolerance=tol)
            assert_near_equal(ieff[case_idx - case_idx_begin], install_eff[case_idx], tolerance=tol)

    def test_case_0_1_2(self):
        # Case 0, 1, 2
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.0, 25000.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [0.0, 125.0, 300.0], units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, [800.0, 800.0, 750.0], units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1850.0, 1850.0, 900.0], units="hp")

        prob.run_model()
        self.compare_results(case_idx_begin=0, case_idx_end=2)

        partial_data = prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)

    def Ttest_case_3_4_5(self):
        # Case 3, 4, 5
        prob = self.prob
        options = self.options

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, [0.0, 0.05, 0.05], units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, [750.0, 750.0, 785.0], units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units="hp")

        prob.run_model()
        self.compare_results(case_idx_begin=3, case_idx_end=5)

    def ttest_case_6_7_8(self):
        # Case 6, 7, 8
        prob = self.prob
        options = self.options

        num_blades = 3
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=num_blades, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, [0.0, 0.05, 0.05], units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, [10000.0, 10000.0, 0.0], units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, [200.0, 200.0, 50.0], units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, [750.0, 750.0, 785.0], units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1000.0, 1000.0, 1250.0], units="hp")

        prob.run_model()
        self.compare_results(case_idx_begin=6, case_idx_end=8)

if __name__ == "__main__":
    unittest.main()
