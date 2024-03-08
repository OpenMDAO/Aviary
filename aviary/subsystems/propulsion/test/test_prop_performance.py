import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.variable_info.variables import Aircraft
from aviary.subsystems.propulsion.prop_performance import PropPerf

from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.options import get_option_defaults

# Setting up truth values from GASP
CT = []
XFT = []
CTX = []
three_quart_blade_angle = []
thrust = []
prop_eff = []
install_loss = []
install_eff = []

# Case 1
CT.append(0.27651)
XFT.append(1.0000)
CTX.append(0.27651)
three_quart_blade_angle.append(25.17)
thrust.append(4634.8)
prop_eff.append(0.0000)
install_loss.append(0.0133)
install_eff.append(0.0000)

# Case 2
CT.append(0.20518)
XFT.append(1.0000)
CTX.append(0.20518)
three_quart_blade_angle.append(29.67)
thrust.append(3415.9)
prop_eff.append(0.7235)
install_loss.append(0.0200)
install_eff.append(0.7090)

# Case 3
CT.append(0.13093)
XFT.append(0.9976)
CTX.append(0.13062)
three_quart_blade_angle.append(44.23)
thrust.append(841.5)
prop_eff.append(0.8920)
install_loss.append(0.0340)
install_eff.append(0.8617)

# Case 4
CT.append(0.10236)
XFT.append(1.0000)
CTX.append(0.10236)
three_quart_blade_angle.append(31.94)
thrust.append(1474.3)
prop_eff.append(0.9059)
install_loss.append(0.0000)
install_eff.append(0.9059)

# Case 5
CT.append(0.10236)
XFT.append(1.0000)
CTX.append(0.10236)
three_quart_blade_angle.append(31.94)
thrust.append(1400.6)
prop_eff.append(0.9059)
install_loss.append(0.0500)
install_eff.append(0.8606)

# Case 6
CT.append(0.19331)
XFT.append(1.0000)
CTX.append(0.19331)
three_quart_blade_angle.append(17.44)
thrust.append(3923.5)
prop_eff.append(0.5075)
install_loss.append(0.0500)
install_eff.append(0.4821)

# Case 7
CT.append(0.10189)
XFT.append(1.0000)
CTX.append(0.10189)
three_quart_blade_angle.append(33.43)
thrust.append(1467.6)
prop_eff.append(0.9017)
install_loss.append(0.0000)
install_eff.append(0.9017)

# Case 8
CT.append(0.10189)
XFT.append(1.0000)
CTX.append(0.10189)
three_quart_blade_angle.append(33.43)
thrust.append(1394.2)
prop_eff.append(0.9017)
install_loss.append(0.0500)
install_eff.append(0.8566)

# Case 9
CT.append(0.18123)
XFT.append(1.0000)
CTX.append(0.18123)
three_quart_blade_angle.append(20.08)
thrust.append(3678.3)
prop_eff.append(0.4758)
install_loss.append(0.0500)
install_eff.append(0.4520)


class PropPerformanceTest(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=4, units='unitless')

        prob = om.Problem()
        pp = prob.model.add_subsystem(
            'pp',
            PropPerf(num_nodes=1, aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=["*"],
        )

        pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        pp.set_input_defaults(Dynamic.Mission.PROPELLER_TIP_SPEED, 800, units="ft/s")
        pp.set_input_defaults(Dynamic.Mission.VELOCITY, 0, units="knot")
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

    def compare_results(self, case_idx):
        p = self.prob
        cthr = float(p.get_val('thrust_coefficient')[0])
        ctlf = float(p.get_val('comp_tip_loss_factor')[0])
        tccl = float(p.get_val('thrust_coefficient_comp_loss')[0])
        angb = float(p.get_val('ang_blade')[0])
        thrt = float(p.get_val('Thrust')[0])
        peff = float(p.get_val('propeller_efficiency')[0])
        lfac = float(p.get_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR)[0])
        ieff = float(p.get_val('install_efficiency')[0])

        tol = 0.005
        assert_near_equal(cthr, CT[case_idx], tolerance=tol)
        assert_near_equal(ctlf, XFT[case_idx], tolerance=tol)
        assert_near_equal(tccl, CTX[case_idx], tolerance=tol)
        assert_near_equal(angb, three_quart_blade_angle[case_idx], tolerance=tol)
        assert_near_equal(thrt, thrust[case_idx], tolerance=tol)
        assert_near_equal(peff, prop_eff[case_idx], tolerance=tol)
        assert_near_equal(lfac, install_loss[case_idx], tolerance=tol)
        assert_near_equal(ieff, install_eff[case_idx], tolerance=tol)

    def test_case_0(self):
        # Case 0
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 0.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 800.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1850.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=0)

    def test_case_1(self):
        # Case 1
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 125.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 800.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1850.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=1)

    def test_case_2(self):
        # Case 2
        prob = self.prob
        prob.set_val(Dynamic.Mission.ALTITUDE, 25000.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 300.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 900.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=2)

    def test_case_3(self):
        # Case 3
        prob = self.prob
        options = self.options

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, 0.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 200.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1000.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=3)

    def test_case_4(self):
        # Case 4
        prob = self.prob
        options = self.options

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, 0.05, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 200.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1000.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=4)

    def test_case_5(self):
        # Case 5
        prob = self.prob
        options = self.options

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, 0.05, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 50.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 785.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1250.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=5)

    def test_case_6(self):
        # Case 6
        prob = self.prob
        options = self.options

        num_blades = 3
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=num_blades, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, 0.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 200.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1000.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=6)

    def test_case_7(self):
        # Case 7
        prob = self.prob
        options = self.options

        num_blades = 3
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=num_blades, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, 0.05, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, 10000.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 200.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 750.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1000.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=7)

    def test_case_8(self):
        # Case 8
        prob = self.prob
        options = self.options

        num_blades = 3
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=num_blades, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=False, units='unitless')
        prob.setup()
        prob.set_val(Dynamic.Mission.INSTALLATION_LOSS_FACTOR, 0.05, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 12.0, units="ft")
        # prob.set_val(Aircraft.Nacelle.AVG_DIAMETER, 2.4, units='ft')
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 150.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT,
                     0.5, units="unitless")
        # prob.set_val('DiamNac_DiamProp', 0.2, units="unitless")
        prob.set_val(Dynamic.Mission.ALTITUDE, 0.0, units="ft")
        prob.set_val(Dynamic.Mission.VELOCITY, 50.0, units="knot")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED, 785.0, units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, 1250.0, units="hp")

        prob.run_model()
        self.compare_results(case_idx=8)


if __name__ == "__main__":
    unittest.main()
