import unittest
import os

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.mission.gasp_based.flight_conditions import FlightConditions
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class FlightConditionsTestCase1(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.TAS),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "rho", val=1.22 * np.ones(2), units="kg/m**3"
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=344 * np.ones(2),
            units="m/s")
        self.prob.model.set_input_defaults("TAS", val=344 * np.ones(2), units="m/s")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-5
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.DYNAMIC_PRESSURE], 1507.6 * np.ones(2), tol)
        assert_near_equal(self.prob[Dynamic.Mission.MACH], np.ones(2), tol)
        assert_near_equal(
            self.prob.get_val("EAS", units="m/s"), 343.3 * np.ones(2), tol
        )

        partial_data = self.prob.check_partials(out_stream=None, method="cs")

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('accel_specs/fc.json')
    def test_fc_spec1(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "accel_specs/fc.json")


class FlightConditionsTestCase2(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.EAS),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "rho", val=1.05 * np.ones(2), units="kg/m**3"
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=344 * np.ones(2),
            units="m/s")
        self.prob.model.set_input_defaults(
            "EAS", val=318.4821143 * np.ones(2), units="m/s"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-5
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.DYNAMIC_PRESSURE], 1297.54 * np.ones(2), tol)
        assert_near_equal(self.prob["TAS"], 1128.61 * np.ones(2), tol)
        assert_near_equal(self.prob[Dynamic.Mission.MACH], np.ones(2), tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('climb_specs/fc.json')
    def test_fc_spec2(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "climb_specs/fc.json")


class FlightConditionsTestCase3(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FlightConditions(num_nodes=2, input_speed_type=SpeedType.MACH),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "rho", val=1.05 * np.ones(2), units="kg/m**3"
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.SPEED_OF_SOUND,
            val=344 * np.ones(2),
            units="m/s")
        self.prob.model.set_input_defaults(
            Dynamic.Mission.MACH, val=np.ones(2), units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-5
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.DYNAMIC_PRESSURE], 1297.54 * np.ones(2), tol)
        assert_near_equal(self.prob["TAS"], 1128.61 * np.ones(2), tol)
        assert_near_equal(
            self.prob.get_val("EAS", units="m/s"), 318.4821143 * np.ones(2), tol
        )

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    @skipIfMissingXDSM('cruise_specs/fc.json')
    def test_fc_spec3(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "cruise_specs/fc.json")


if __name__ == "__main__":
    unittest.main()
