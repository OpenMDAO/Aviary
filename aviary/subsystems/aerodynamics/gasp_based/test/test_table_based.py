import unittest
import os

import numpy as np
import openmdao
import openmdao.api as om
import pkg_resources
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging import version

from aviary.subsystems.aerodynamics.gasp_based.table_based import (
    CruiseAero, LowSpeedAero)
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft, Dynamic


class TestCruiseAero(unittest.TestCase):

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"),
                     "Older version of OpenMDAO does not properly skip Metamodel.")
    def test_climb(self):
        prob = om.Problem()

        fp = "subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_free.txt"
        prob.model = CruiseAero(num_nodes=8, aero_data=fp)

        prob.setup(force_alloc_complex=True)

        prob.set_val(
            Dynamic.Mission.MACH, [
                0.381, 0.384, 0.391, 0.399, 0.8, 0.8, 0.8, 0.8])
        prob.set_val("alpha", [5.19, 5.19, 5.19, 5.18, 3.58, 3.81, 4.05, 4.18])
        prob.set_val(
            Dynamic.Mission.ALTITUDE, [
                500, 1000, 2000, 3000, 35000, 36000, 37000, 37500])
        prob.run_model()

        cl_exp = np.array(
            [0.5968, 0.5975, 0.5974, 0.5974, 0.5566, 0.5833, 0.6113, 0.6257]
        )
        cd_exp = np.array(
            [0.0307, 0.0307, 0.0307, 0.0307, 0.0296, 0.0310, 0.0326, 0.0334]
        )

        assert_near_equal(prob["CL"], cl_exp, tolerance=0.005)
        assert_near_equal(prob["CD"], cd_exp, tolerance=0.009)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=4e-7, rtol=2e-7)

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"),
                     "Older version of OpenMDAO does not properly skip Metamodel.")
    def test_cruise(self):
        prob = om.Problem()
        fp = pkg_resources.resource_filename(
            "aviary", f"subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_free.txt")
        prob.model = CruiseAero(num_nodes=2, aero_data=fp)
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Mission.MACH, [0.8, 0.8])
        prob.set_val("alpha", [4.216, 3.146])
        prob.set_val(Dynamic.Mission.ALTITUDE, [37500, 37500])
        prob.run_model()

        cl_exp = np.array([0.6304, 0.5059])
        cd_exp = cl_exp / np.array([18.608, 18.425])

        assert_near_equal(prob["CL"], cl_exp, tolerance=0.005)
        assert_near_equal(prob["CD"], cd_exp, tolerance=0.005)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=9e-8, rtol=2e-7)


class TestLowSpeedAero(unittest.TestCase):

    # gear retraction start time at takeoff
    t_init_gear_to = 37.3
    # flap retraction start time at takeoff
    t_init_flaps_to = 47.6
    # takeoff flap deflection (deg)
    flap_defl_to = 10

    free_data = pkg_resources.resource_filename(
        "aviary", f"subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_free.txt")
    flaps_data = pkg_resources.resource_filename(
        "aviary", f"subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_flaps.txt")
    ground_data = pkg_resources.resource_filename(
        "aviary", f"subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_ground.txt")

    @skipIfMissingXDSM('rotation_specs/aero.json')
    def test_spec(self):
        comp = LowSpeedAero(free_aero_data=self.free_data,
                            flaps_aero_data=self.flaps_data,
                            ground_aero_data=self.ground_data,
                            extrapolate=True)
        assert_match_spec(comp, "rotation_specs/aero.json")

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"),
                     "Older version of OpenMDAO does not properly skip Metamodel.")
    def test_groundroll(self):
        # takeoff with flaps applied, gear down, zero alt
        prob = om.Problem()
        prob.model = LowSpeedAero(num_nodes=4,
                                  free_aero_data=self.free_data,
                                  flaps_aero_data=self.flaps_data,
                                  ground_aero_data=self.ground_data,
                                  extrapolate=True)
        prob.setup()

        prob.set_val("t_curr", [0.0, 1.0, 2.0, 3.0])
        prob.set_val(Dynamic.Mission.ALTITUDE, 0)
        prob.set_val(Dynamic.Mission.MACH, [0.0, 0.009, 0.018, 0.026])
        prob.set_val("alpha", 0)
        # TODO set q if we want to test lift/drag forces

        prob.set_val("flap_defl", self.flap_defl_to)
        prob.set_val("t_init_gear", self.t_init_gear_to)
        prob.set_val("t_init_flaps", self.t_init_flaps_to)
        prob.run_model()

        cl_exp = 0.5597 * np.ones(4)
        cd_exp = 0.0572 * np.ones(4)

        # TODO yikes @ tolerances
        assert_near_equal(prob["CL"], cl_exp, tolerance=0.1)
        assert_near_equal(prob["CD"], cd_exp, tolerance=0.3)

        partial_data = prob.check_partials(
            method="fd", out_stream=None
        )  # fd because there is a cs in the time ramp
        assert_check_partials(partial_data, atol=3e-7, rtol=6e-5)

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"),
                     "Older version of OpenMDAO does not properly skip Metamodel.")
    def test_takeoff(self):
        # takeoff crossing flap retraction and gear retraction points
        prob = om.Problem()
        prob.model = LowSpeedAero(num_nodes=8,
                                  free_aero_data=self.free_data,
                                  flaps_aero_data=self.flaps_data,
                                  ground_aero_data=self.ground_data,
                                  extrapolate=True)
        prob.setup()

        prob.set_val(
            "t_curr",
            [37.0, 38.0, 39.0, 40.0, 47.0, 48.0, 49.0, 50.0],
        )

        alts = [44.2, 62.7, 84.6, 109.7, 373.0, 419.4, 465.3, 507.8]
        prob.set_val(Dynamic.Mission.ALTITUDE, alts)
        prob.set_val(
            Dynamic.Mission.MACH, [
                0.257, 0.260, 0.263, 0.265, 0.276, 0.277, 0.279, 0.280])
        prob.set_val("alpha", [8.94, 8.74, 8.44, 8.24, 6.45, 6.34, 6.76, 7.59])
        # TODO set q if we want to test lift/drag forces

        prob.set_val(Aircraft.Wing.AREA, 1370)
        prob.set_val(Aircraft.Wing.SPAN, 117.8)

        prob.set_val("flap_defl", self.flap_defl_to)
        prob.set_val("t_init_gear", self.t_init_gear_to)
        prob.set_val("t_init_flaps", self.t_init_flaps_to)
        prob.run_model()

        cl_exp = np.array(
            [1.3734, 1.3489, 1.3179, 1.2979, 1.1356, 1.0645, 0.9573, 0.8876]
        )
        cd_exp = np.array(
            [0.1087, 0.1070, 0.1019, 0.0969, 0.0661, 0.0641, 0.0644, 0.0680]
        )

        # TODO yikes @ tolerances
        assert_near_equal(prob["CL"], cl_exp, tolerance=0.02)
        assert_near_equal(prob["CD"], cd_exp, tolerance=0.09)

        partial_data = prob.check_partials(
            method="fd", out_stream=None
        )  # fd because there is a cs in the time ramp
        assert_check_partials(
            partial_data, atol=0.255, rtol=5e-7
        )  # fd does very poorly with the t_curr, t_init, and duration values in the time
        # ramp because its step is so much bigger that cs. By decreasing the fd step
        # size you can see that the derivatives are right wrt these values


if __name__ == "__main__":
    unittest.main()
