import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.non_dimensional_conversion import \
    DimensionalNonDimensionalInterchange
from aviary.variable_info.variables import Aircraft

from aviary.subsystems.geometry.gasp_based.non_dimensional_conversion import DimensionalNonDimensionalInterchange
from aviary.variable_info.options import get_option_defaults


class FoldOnlyTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=180.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=118.0, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS], 0.65556, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FoldOnlyTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=150.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDED_SPAN], 75.0, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class StrutOnlyTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=180.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=118.0, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS], 0.65556, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class StrutOnlyTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=150.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Strut.ATTACHMENT_LOCATION], 75.0, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FoldAndStrutTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=180.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=118.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0.5, units="unitless"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS], 0.65556, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.ATTACHMENT_LOCATION], 90.0, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FoldAndStrutTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=150.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, val=0.8, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=90.0, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDED_SPAN], 120.0, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS], 0.6, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FoldAndStrutTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem("dimensionless_calcs",
                                      DimensionalNonDimensionalInterchange(
                                          aviary_options=options),
                                      promotes_inputs=["aircraft:*"],
                                      promotes_outputs=["aircraft:*"]
                                      )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=180.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=118.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=108.0, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS], 0.65556, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS], .6, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
