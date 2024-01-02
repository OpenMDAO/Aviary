import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.basic_calculations import \
    BasicFlapsCalculations
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft

"""
All data is from validation files using standalone flaps model
"""


class BasicFlapsCalculationsTestCase(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem(model=om.Group())

        self.prob.model.add_subsystem('BC', BasicFlapsCalculations(), promotes=['*'])

        self.prob.setup()

        # initial conditions
        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.41, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

    def test_case(self):

        self.prob.run_model()
        tol = 2.1e-4
        print()

        reg_data = 0.74444
        ans = self.prob["VLAM8"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.93578
        ans = self.prob["VDEL4"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.90761
        ans = self.prob["VDEL5"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.9975
        ans = self.prob["VLAM9"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.5
        ans = self.prob["slat_defl_ratio"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.89761
        ans = self.prob[Aircraft.Wing.SLAT_SPAN_RATIO]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.09239
        ans = self.prob["body_to_span_ratio"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.12679
        ans = self.prob["chord_to_body_ratio"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.79208
        ans = self.prob["VLAM12"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method="fd")
        assert_check_partials(data, atol=1e-6, rtol=4e-6)

    @skipIfMissingXDSM('flaps_specs/basic.json')
    def test_basic_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "flaps_specs/basic.json")


if __name__ == "__main__":
    unittest.main()
