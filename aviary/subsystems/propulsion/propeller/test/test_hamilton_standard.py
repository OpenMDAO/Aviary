import unittest
import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.propulsion.propeller.hamilton_standard import (
    HamiltonStandard, PreHamiltonStandard, PostHamiltonStandard,
)
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.constants import RHO_SEA_LEVEL_ENGLISH


class PreHamiltonStandardTest(unittest.TestCase):
    """
    Test computation in PreHamiltonStandard class.
    """

    def setUp(self):
        prob = om.Problem()

        num_nodes = 3

        prob.model.add_subsystem(
            'prehs',
            PreHamiltonStandard(num_nodes=num_nodes),
            promotes_inputs=['*'],
            promotes_outputs=["*"],
        )

        prob.setup()
        self.prob = prob

    def test_preHS(self):
        prob = self.prob
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED,
                     [700.0, 750.0, 800.0], units="ft/s")
        prob.set_val(Dynamic.Mission.SHAFT_POWER, [1850.0, 1850.0, 900.0], units="hp")
        prob.set_val(Dynamic.Mission.DENSITY,
                     [0.00237717, 0.00237717, 0.00106526], units="slug/ft**3")
        prob.set_val(Dynamic.Mission.VELOCITY, [100.0, 100, 100], units="ft/s")
        prob.set_val(Dynamic.Mission.SPEED_OF_SOUND,
                     [661.46474547, 661.46474547, 601.93668333], units="knot")

        prob.run_model()

        tol = 5e-4
        assert_near_equal(prob.get_val("power_coefficient"),
                          [0.3871, 0.3147, 0.2815], tolerance=tol)
        assert_near_equal(
            prob.get_val("advance_ratio"),
            [0.44879895, 0.41887902, 0.39269908],
            tolerance=tol,
        )
        assert_near_equal(
            prob.get_val("tip_mach"), [0.6270004, 0.67178614, 0.78743671], tolerance=tol
        )

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
        assert_check_partials(partial_data, atol=0.0003, rtol=7e-7)


class HamiltonStandardTest(unittest.TestCase):
    """
    Test computation in HamiltonStandard class.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.NUM_PROPELLER_BLADES, val=4, units='unitless')

        prob = om.Problem()

        num_nodes = 3

        prob.model.add_subsystem(
            'hs',
            HamiltonStandard(num_nodes=num_nodes, aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=["*"],
        )

        prob.setup()
        self.prob = prob

    def test_HS(self):
        prob = self.prob
        prob.set_val("power_coefficient", [0.2352, 0.2352, 0.2553], units="unitless")
        prob.set_val("advance_ratio", [0.0066, 0.8295, 1.9908], units="unitless")
        prob.set_val(Dynamic.Mission.MACH, [0.001509, 0.1887, 0.4976], units="unitless")
        prob.set_val("tip_mach", [1.2094, 1.2094, 1.3290], units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR, 114.0, units="unitless")
        prob.set_val(Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICIENT,
                     0.5, units="unitless")

        prob.run_model()

        tol = 5e-4
        assert_near_equal(prob.get_val("thrust_coefficient"),
                          [0.2763, 0.2052, 0.1158], tolerance=tol)
        assert_near_equal(prob.get_val("comp_tip_loss_factor"),
                          [1.0, 1.0, 0.9818], tolerance=tol)

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
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


class PostHamiltonStandardTest(unittest.TestCase):
    """
    Test computation in PostHamiltonStandard class.
    """

    def setUp(self):
        prob = om.Problem()

        num_nodes = 3

        prob.model.add_subsystem(
            'posths',
            PostHamiltonStandard(num_nodes=num_nodes),
            promotes_inputs=['*'],
            promotes_outputs=["*"],
        )

        prob.setup()
        self.prob = prob

    def test_postHS(self):
        prob = self.prob
        prob.set_val("power_coefficient", [0.3871, 0.3147, 0.2815], units="unitless")
        prob.set_val("advance_ratio", [0.4494, 0.4194, 0.3932], units="unitless")
        prob.set_val(Dynamic.Mission.PROPELLER_TIP_SPEED,
                     [700.0, 750.0, 800.0], units="ft/s")
        prob.set_val(
            Dynamic.Mission.DENSITY,
            np.array([1.0001, 1.0001, 0.4482]) * RHO_SEA_LEVEL_ENGLISH,
            units="slug/ft**3",
        )
        prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.0, units="ft")
        prob.set_val("thrust_coefficient", [0.2765, 0.2052, 0.1158], units="unitless")
        prob.set_val("install_loss_factor", [0.0133, 0.0200, 0.0325], units="unitless")
        prob.set_val("comp_tip_loss_factor", [1.0, 1.0, 0.9819], units="unitless")

        prob.run_model()

        tol = 5e-4
        assert_near_equal(prob.get_val("thrust_coefficient_comp_loss"),
                          [0.2765, 0.2052, 0.1137], tolerance=tol)
        assert_near_equal(prob.get_val(Dynamic.Mission.THRUST),
                          [3218.9508, 2723.7294, 759.7543], tolerance=tol)
        assert_near_equal(prob.get_val("propeller_efficiency"),
                          [0.321, 0.2735, 0.1588], tolerance=tol)
        assert_near_equal(prob.get_val("install_efficiency"),
                          [0.3167, 0.2680, 0.15378], tolerance=tol)

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
    # test = HamiltonStandardTest()
    # test.setUp()
    # test.test_HS()
