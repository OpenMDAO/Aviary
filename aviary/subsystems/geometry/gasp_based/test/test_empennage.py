import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.empennage import (EmpennageSize,
                                                             TailSize,
                                                             TailVolCoef)
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft

tol = 5e-4
partial_tols = {"atol": 1e-8, "rtol": 1e-8}


class TestTailVolCoef(
    unittest.TestCase
):  # note, this component is not used in the large single aisle
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "hvc",
            TailVolCoef(vertical=False),
            promotes_inputs=[
                "aircraft:*",
            ],
        )
        self.prob.model.add_subsystem(
            "vvc",
            TailVolCoef(vertical=True),
            promotes_inputs=[
                "aircraft:*",
            ],
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults("hvc.cab_w", val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults("hvc.wing_ref", val=12.615, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units="unitless"
        )
        self.prob.model.set_input_defaults("vvc.cab_w", val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults("vvc.wing_ref", 117.8054, units="ft")
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_large_sinle_aisle_1_volcoefs(self):
        self.prob.run_model()
        assert_near_equal(
            self.prob["hvc.vol_coef"], 1.52233, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["vvc.vol_coef"], 0.11623, tol
        )  # not actual GASP value

    def test_partials(self):
        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, **partial_tols)


class TestTailComp(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem("tail", TailSize(), promotes=["*"])
        self.prob.setup(check=False, force_alloc_complex=True)

        # values for horizontal tail
        self.prob.model.set_input_defaults("vol_coef", val=1.189, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults("r_arm", val=0.2307, units="unitless")
        self.prob.model.set_input_defaults("wing_ref", val=12.615, units="ft")
        self.prob.model.set_input_defaults("ar", val=4.75, units="unitless")
        self.prob.model.set_input_defaults("tr", val=0.352, units="unitless")

    def test_large_sinle_aisle_1_htail(self):
        self.prob.run_model()

        assert_near_equal(self.prob["area"], 375.9, tol)
        assert_near_equal(self.prob["span"], 42.25, tol)
        assert_near_equal(
            self.prob["rchord"], 13.16130387591471, tol
        )  # (potentially not actual GASP value, it is calculated twice in different places)
        assert_near_equal(self.prob["chord"], 9.57573, tol)
        assert_near_equal(self.prob["arm"], 54.7, tol)

    def test_large_sinle_aisle_1_vtail(self):
        # override horizontal tail defaults for vertical tail
        self.prob.set_val("vol_coef", 0.145, units="unitless")
        self.prob.set_val(Aircraft.Wing.AREA, 1370.3, units="ft**2")
        self.prob.set_val("r_arm", 2.362, units="unitless")
        self.prob.set_val("wing_ref", 117.8, units="ft")
        self.prob.set_val("ar", 1.67, units="unitless")
        self.prob.set_val("tr", 0.801, units="unitless")

        self.prob.run_model()

        assert_near_equal(self.prob["area"], 469.3, tol)
        assert_near_equal(self.prob["span"], 28, tol)
        assert_near_equal(self.prob["rchord"], 18.61267549773935, tol)
        assert_near_equal(self.prob["chord"], 16.83022, tol)
        assert_near_equal(
            self.prob["arm"], 49.87526, tol
        )  # note: slightly different from GASP output value, likely truncation differences, this value is from Kenny

    def test_partials(self):
        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, **partial_tols)


class TestEmpennageGroup(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem("emp", EmpennageSize(), promotes=["*"])

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.615, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, val=0.145, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, val=2.362, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units="unitless"
        )

    def test_large_sinle_aisle_1_defaults(self):
        self.prob.model.emp.options["aviary_options"] = get_option_defaults()

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        assert_near_equal(self.prob[Aircraft.HorizontalTail.AREA], 375.9, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.SPAN], 42.25, tol)
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.16130387591471, tol
        )  # (potentially not actual GASP value, it is calculated twice in different places)
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.57573, tol
        )
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.7, tol)

        assert_near_equal(self.prob[Aircraft.VerticalTail.AREA], 469.3, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.SPAN], 28, tol)
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.ROOT_CHORD], 18.61267549773935, tol
        )
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.83022, tol)
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.MOMENT_ARM], 49.87526, tol
        )  # note: slightly different from GASP output value, likely numerical diff.s, this value is from Kenny

    @skipIfMissingXDSM('size_both1_specs/empennage.json')
    def test_io_emp_spec_defaults(self):
        self.prob.model.emp.options["aviary_options"] = get_option_defaults()

        self.prob.setup(check=False, force_alloc_complex=True)

        subsystem = self.prob.model

        assert_match_spec(subsystem, "size_both1_specs/empennage.json")

    def test_large_sinle_aisle_1_calc_volcoefs(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        self.prob.model.emp.options["aviary_options"] = options
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units="unitless")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")

        self.prob.run_model()

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.VOLUME_COEFFICIENT], 1.52233, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.VOLUME_COEFFICIENT], 0.11623, tol
        )  # not actual GASP value

    @skipIfMissingXDSM('size_both2_specs/empennage.json')
    def test_io_emp_spec_vol_coefs(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        self.prob.model.emp.options["aviary_options"] = options
        self.prob.setup(check=False, force_alloc_complex=True)

        subsystem = self.prob.model

        assert_match_spec(subsystem, "size_both2_specs/empennage.json")


if __name__ == "__main__":
    unittest.main()
