import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.meta_model import \
    MetaModelGroup
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.enums import FlapType
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic

"""
All data is from validation files using standalone flaps model
"""


class MetaModelTestCasePlain(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE, val=FlapType.PLAIN, units='unitless')
        self.prob.model = LuTMMa = MetaModelGroup(aviary_options=options)
        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val("flap_defl_ratio", 40 / 60)
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)
        self.prob.set_val("slat_defl_ratio", 10 / 20)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89761)
        self.prob.set_val("reynolds", 164.78406)
        self.prob.set_val(Dynamic.Mission.MACH, 0.18368)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89761)
        self.prob.set_val("body_to_span_ratio", 0.09239)
        self.prob.set_val("chord_to_body_ratio", 0.12679)

    def test_case(self):

        self.prob.run_model()
        tol = 1e-4

        reg_data = 1
        ans = self.prob["VDEL1"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.55667
        ans = self.prob["VDEL2"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.76500
        ans = self.prob["VDEL3"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.05498
        ans = self.prob["fus_lift"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.97217
        ans = self.prob["VLAM1"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.09948
        ans = self.prob["VLAM2"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.97217
        ans = self.prob["VLAM3"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.19742
        ans = self.prob["VLAM4"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1
        ans = self.prob["VLAM5"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.80000
        ans = self.prob["VLAM6"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.73500
        ans = self.prob["VLAM7"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.74000
        ans = self.prob["VLAM10"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.84232
        ans = self.prob["VLAM11"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.03209
        ans = self.prob["VLAM13"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.99082
        ans = self.prob["VLAM14"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method="fd")
        assert_check_partials(data, atol=1e-4, rtol=1e-4)

    @skipIfMissingXDSM('flaps_specs/tables.json')
    def test_lookup_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "flaps_specs/tables.json")


class MetaModelTestCaseSingleSlotted(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE,
                        val=FlapType.SINGLE_SLOTTED, units='unitless')
        self.prob.model = LuTMMb = MetaModelGroup(aviary_options=options)
        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")

    def test_case(self):

        self.prob.run_model()
        tol = 1e-4

        reg_data = 1
        ans = self.prob["VDEL1"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.25725
        ans = self.prob["VLAM4"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1
        ans = self.prob["VLAM5"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.0
        ans = self.prob["VLAM6"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method="fd")
        assert_check_partials(data, atol=1e-4, rtol=1e-4)

    @skipIfMissingXDSM('flaps_specs/tables.json')
    def test_lookup_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "flaps_specs/tables.json")


class MetaModelTestCaseFowler(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE, val=FlapType.FOWLER, units='unitless')
        self.prob.model = LuTMMc = MetaModelGroup(aviary_options=options)
        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val("flap_defl", 40.0, units="deg")

    def test_case(self):

        self.prob.run_model()
        tol = 1e-4

        reg_data = 1.0
        ans = self.prob["VLAM5"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.11
        ans = self.prob["VLAM6"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method="fd")
        assert_check_partials(data, atol=1e-4, rtol=1e-4)

    @skipIfMissingXDSM('flaps_specs/tables.json')
    def test_lookup_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "flaps_specs/tables.json")


if __name__ == "__main__":
    unittest.main()
