import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.gasp_based.wing import (
    BWBWingFoldVolume,
    BWBWingGroup,
    BWBWingVolume,
    BWBWingVolume,
    ExposedWing,
    WingFoldArea,
    WingFoldVolume,
    WingGroup,
    WingParameters,
    WingSize,
    WingVolume,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings
from aviary.variable_info.variables import Aircraft, Settings


class WingSizeTestCase(unittest.TestCase):
    """Tests for WingSize."""

    def _make_prob(self, gross_mass, wing_loading, aspect_ratio):
        prob = om.Problem()
        prob.model.add_subsystem('size', WingSize(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, gross_mass, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, wing_loading, units='lbf/ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, aspect_ratio, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
        prob = self._make_prob(gross_mass=175400, wing_loading=128, aspect_ratio=10.13)
        prob.run_model()

        tol = 2e-4
        expected_values = {
            Aircraft.Wing.AREA: (1370.3, 'ft**2'),
            Aircraft.Wing.SPAN: (117.8, 'ft'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_bwb(self):
        prob = self._make_prob(gross_mass=150000.0, wing_loading=70.0, aspect_ratio=10.0)
        prob.run_model()

        tol = 1e-7
        expected_values = {
            Aircraft.Wing.AREA: (2142.8571, 'ft**2'),
            Aircraft.Wing.SPAN: (146.38501, 'ft'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingParametersTestCase(unittest.TestCase):
    """Tests for WingParameters."""

    def _make_prob(
        self, area, span, aspect_ratio, taper_ratio, sweep, tc_root, avg_diameter, tc_tip
    ):
        prob = om.Problem()
        prob.model.add_subsystem('parameters', WingParameters(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Wing.AREA, area, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, span, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, aspect_ratio, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, taper_ratio, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, sweep, units='deg')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, tc_root, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, avg_diameter, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, tc_tip, units='unitless'
        )

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
        prob = self._make_prob(
            area=1370.3,
            span=117.8,
            aspect_ratio=10.13,
            taper_ratio=0.33,
            sweep=25,
            tc_root=0.15,
            avg_diameter=13.1,
            tc_tip=0.12,
        )
        prob.run_model()

        tol = 5e-4
        # this is slightly different from the GASP output value, likely due to rounding error
        expected_values = {
            Aircraft.Wing.CENTER_CHORD: (17.49, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (12.615, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (16.41, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.1397, 'unitless'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_bwb(self):
        """Test BWB data for BWBWingParameters."""
        prob = self._make_prob(
            area=2142.85718,
            span=146.385,
            aspect_ratio=10.0,
            taper_ratio=0.27444,
            sweep=30.0,
            tc_root=0.165,
            avg_diameter=38.0,
            tc_tip=0.1,
        )
        prob.run_model()

        tol = 1e-7
        expected_values = {
            Aircraft.Wing.CENTER_CHORD: (22.97244663, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (16.2200537, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (20.33371818, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.13596576, 'unitless'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingVolumeTestCase(unittest.TestCase):
    """Tests for WingVolume and BWBWingVolume."""

    def _make_prob(
        self,
        component,
        area,
        span,
        aspect_ratio,
        taper_ratio,
        tc_root,
        avg_diameter,
        tc_tip,
        wing_fuel_fraction,
        has_fold=False,
        smooth_mass_discontinuities=False,
        extra_inputs=None,
    ):
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=has_fold, units='unitless')
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
            val=smooth_mass_discontinuities,
            units='unitless',
        )
        options.set_val(Settings.VERBOSITY, val=Verbosity.BRIEF)

        prob = om.Problem()
        prob.model.add_subsystem('wing_vol', component, promotes=['*'])

        if area is not None:
            prob.model.set_input_defaults(Aircraft.Wing.AREA, area, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, span, units='ft')
        if aspect_ratio is not None:
            prob.model.set_input_defaults(
                Aircraft.Wing.ASPECT_RATIO, aspect_ratio, units='unitless'
            )
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, taper_ratio, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, tc_root, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, avg_diameter, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, tc_tip, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, wing_fuel_fraction, units='unitless'
        )
        if extra_inputs:
            for name, (val, units) in extra_inputs.items():
                prob.model.set_input_defaults(name, val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
        prob = self._make_prob(
            component=WingVolume(),
            area=1370.3,
            span=117.8,
            aspect_ratio=10.13,
            taper_ratio=0.33,
            tc_root=0.15,
            avg_diameter=13.1,
            tc_tip=0.12,
            wing_fuel_fraction=0.6,
            has_fold=False,
        )
        prob.run_model()

        tol = 5e-4
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_bwb(self):
        """Test BWB data for BWBWingVolume."""
        prob = self._make_prob(
            component=BWBWingVolume(),
            area=None,
            span=146.385,
            aspect_ratio=None,
            taper_ratio=0.27444,
            tc_root=0.165,
            avg_diameter=38.0,
            tc_tip=0.1,
            wing_fuel_fraction=0.45,
            has_fold=False,
            smooth_mass_discontinuities=False,
            extra_inputs={
                Aircraft.LandingGear.MAIN_GEAR_LOCATION: (0.0, 'unitless'),
                Aircraft.Wing.CENTER_CHORD: (22.9724445, 'ft'),
            },
        )
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 783.6209, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class WingFoldAreaTestCase(unittest.TestCase):
    """Tests for WingFoldArea."""

    def _make_prob(self, choose_fold_location, taper_ratio, area, span, extra_inputs):
        options = AviaryValues()
        options.set_val(
            Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=choose_fold_location, units='unitless'
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            WingFoldArea(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, taper_ratio, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=area, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=span, units='ft')
        for name, (val, units) in extra_inputs.items():
            prob.model.set_input_defaults(name, val=val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_strut_y_no_choose_fold_location(self):
        prob = self._make_prob(
            choose_fold_location=False,  # toggled OFF for this test
            taper_ratio=0.33,
            area=1370.3,
            span=117.8,
            extra_inputs={'strut_y': (25, 'ft')},  # not actual GASP value
        )
        prob.run_model()

        tol = 1e-4
        assert_near_equal(
            prob[Aircraft.Wing.FOLDING_AREA], 620.04352246, tol
        )  # not actual GASP value

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_folded_span_choose_fold_location(self):
        prob = self._make_prob(
            choose_fold_location=True,
            taper_ratio=0.33,
            area=1370.3,
            span=117.8,
            extra_inputs={Aircraft.Wing.FOLDED_SPAN: (25, 'ft')},  # not actual GASP value
        )
        prob.run_model()

        tol = 1e-4
        assert_near_equal(
            prob[Aircraft.Wing.FOLDING_AREA], 964.0812219, tol
        )  # not actual GASP value

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    def test_bwb_folded_span_choose_fold_location(self):
        prob = self._make_prob(
            choose_fold_location=True,
            taper_ratio=0.27444,
            area=2142.85718,
            span=146.385,
            extra_inputs={Aircraft.Wing.FOLDED_SPAN: (118, 'ft')},
        )
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Wing.FOLDING_AREA], 224.82521003, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingFoldVolumeTestCase(unittest.TestCase):
    """Tests for WingFoldVolume."""

    def _make_prob(self, choose_fold_location, folding_area, extra_inputs):
        options = AviaryValues()
        options.set_val(
            Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=choose_fold_location, units='unitless'
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            WingFoldVolume(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, val=folding_area, units='ft**2')
        for name, (val, units) in extra_inputs.items():
            prob.model.set_input_defaults(name, val=val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_strut_y_no_choose_fold_location(self):
        prob = self._make_prob(
            choose_fold_location=False,  # toggled OFF for this test
            folding_area=620.0435,
            extra_inputs={'strut_y': (25, 'ft')},  # not actual GASP value
        )
        prob.run_model()

        tol = 1e-4
        # not actual GASP values
        expected_values = {
            'nonfolded_taper_ratio': 0.71561969,
            'nonfolded_wing_area': 750.25647754,
            'tc_ratio_mean_folded': 0.14363328,
            'nonfolded_AR': 3.33219382,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 712.3428037422319,
        }
        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)

    def test_folded_span_choose_fold_location(self):
        prob = self._make_prob(
            choose_fold_location=True,
            folding_area=964.0812219,
            extra_inputs={Aircraft.Wing.FOLDED_SPAN: (25, 'ft')},  # not actual GASP value
        )
        prob.run_model()

        tol = 1e-4
        # not actual GASP values
        expected_values = {
            'nonfolded_taper_ratio': 0.85780985,
            'nonfolded_wing_area': 406.2187781,
            'tc_ratio_mean_folded': 0.14681664,
            'nonfolded_AR': 1.53857978,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 406.64971668264957,
        }
        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class BWBWingFoldVolumeTestCase(unittest.TestCase):
    """Tests for BWBWingFoldVolume."""

    def _make_prob(self, choose_fold_location, extra_inputs):
        options = AviaryValues()
        options.set_val(
            Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=choose_fold_location, units='unitless'
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            BWBWingFoldVolume(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')
        prob.model.set_input_defaults('wing_volume_no_fold', val=783.6209, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        for name, (val, units) in extra_inputs.items():
            prob.model.set_input_defaults(name, val=val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_no_choose_fold_location(self):
        """
        Test against GASP BWB model, CHOOSE_FOLD_LOCATION = False
        This case should not be allowed, but it is tested anyway.
        """
        prob = self._make_prob(
            choose_fold_location=False,  # toggled OFF for this test
            extra_inputs={'strut_y': (59, 'ft')},
        )
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.90774, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)

    def test_choose_fold_location(self):
        """Test against GASP BWB model, CHOOSE_FOLD_LOCATION = True."""
        prob = self._make_prob(
            choose_fold_location=True,
            extra_inputs={Aircraft.Wing.FOLDED_SPAN: (118.0, 'ft')},
        )
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.90774, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


@use_tempdirs
class WingGroupTestCase(unittest.TestCase):
    """Tests for WingGroup."""

    def _make_prob(self, options, inputs):
        prob = om.Problem()
        prob.model.add_subsystem('group', WingGroup(), promotes=['*'])

        for name, (val, units) in inputs.items():
            prob.model.set_input_defaults(name, val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_no_fold_no_strut(self):
        """
        Actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix.

        HAS_FOLD = False
        HAS_STRUT = False
        CHOOSE_FOLD_LOCATION = True
        DIMENSIONAL_LOCATION_SPECIFIED = False
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
        """
        options = AviaryValues()

        inputs = {
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),
            Aircraft.Design.WING_LOADING: (128, 'lbf/ft**2'),
            Aircraft.Wing.ASPECT_RATIO: (10.13, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.33, 'unitless'),
            Aircraft.Wing.SWEEP: (25, 'deg'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.15, 'unitless'),
            Aircraft.Fuselage.AVG_DIAMETER: (13.1, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP: (0.12, 'unitless'),
            Aircraft.Fuel.WING_FUEL_FRACTION: (0.6, 'unitless'),
        }

        prob = self._make_prob(options, inputs)
        prob.run_model()

        tol = 5e-4

        # THICKNESS_TO_CHORD_UNWEIGHTED is slightly different from the GASP output value,
        # likely due to rounding error
        expected_values = {
            Aircraft.Wing.AREA: (1370.3, 'ft**2'),
            Aircraft.Wing.SPAN: (117.8, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (17.49, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (12.615, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (16.41, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.1397, 'unitless'),
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: (1114, 'ft**3'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    def test_fold_and_strut_dimensional_location(self):
        """
        Wing with both folds and struts, fold and strut dimensional location specified.

        The fold is at the strut connection.
        HAS_FOLD = True
        HAS_STRUT = True
        CHOOSE_FOLD_LOCATION = False
        DIMENSIONAL_LOCATION_SPECIFIED = True
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
        """
        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for the full WingGroup subsystem hierarchy.
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=False, units='unitless'
        )

        inputs = {
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),
            Aircraft.Design.WING_LOADING: (128, 'lbf/ft**2'),
            Aircraft.Wing.ASPECT_RATIO: (10.13, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.33, 'unitless'),
            Aircraft.Wing.SWEEP: (25, 'deg'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.15, 'unitless'),
            Aircraft.Fuselage.AVG_DIAMETER: (13.1, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP: (0.12, 'unitless'),
            Aircraft.Strut.AREA_RATIO: (0.02189, 'unitless'),  # not actual GASP value
            Aircraft.Strut.ATTACHMENT_LOCATION: (1.0, 'ft'),  # not actual GASP value
            Aircraft.Fuel.WING_FUEL_FRACTION: (0.6, 'unitless'),
        }

        prob = self._make_prob(options, inputs)
        prob.run_model()

        tol = 5e-4

        # THICKNESS_TO_CHORD_UNWEIGHTED is slightly different from the GASP output value,
        # likely due to rounding error
        expected_values = {
            Aircraft.Wing.AREA: (1370.3, 'ft**2'),
            Aircraft.Wing.SPAN: (117.8, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (17.49, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (12.615, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (16.41, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.1397, 'unitless'),
            Aircraft.Wing.FOLDING_AREA: (1352.8724859, 'ft**2'),  # not actual GASP value
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: (
                18.26837098,
                'ft**3',
            ),  # not actual GASP value
            Aircraft.Strut.LENGTH: (14.42957033, 'ft'),  # not actual GASP value
            Aircraft.Strut.CHORD: (1.03953199, 'ft'),  # not actual GASP value
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        # not actual GASP values
        expected_unitless = {
            'nonfolded_taper_ratio': 0.9943133,
            'nonfolded_wing_area': 17.4400141,
            'tc_ratio_mean_folded': 0.14987269,
            'nonfolded_AR': 0.0573394,
        }
        for var_name, expected in expected_unitless.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=7e-12, rtol=1e-12)

    def test_fold_dimensional_location(self):
        """
        Wing with folds which has dimensional location specified.

        HAS_FOLD = True
        HAS_STRUT = False
        CHOOSE_FOLD_LOCATION = True
        DIMENSIONAL_LOCATION_SPECIFIED = True
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
        """
        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for the full WingGroup subsystem hierarchy.
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=False, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )

        inputs = {
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),
            Aircraft.Design.WING_LOADING: (128, 'lbf/ft**2'),
            Aircraft.Wing.ASPECT_RATIO: (10.13, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.33, 'unitless'),
            Aircraft.Wing.SWEEP: (25, 'deg'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.15, 'unitless'),
            Aircraft.Fuselage.AVG_DIAMETER: (13.1, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP: (0.12, 'unitless'),
            Aircraft.Wing.FOLDED_SPAN: (25, 'ft'),  # not actual GASP value
            Aircraft.Fuel.WING_FUEL_FRACTION: (0.6, 'unitless'),
        }

        prob = self._make_prob(options, inputs)
        prob.run_model()

        tol = 5e-4

        # THICKNESS_TO_CHORD_UNWEIGHTED is slightly different from the GASP output value,
        # likely due to rounding error
        expected_values = {
            Aircraft.Wing.AREA: (1370.3, 'ft**2'),
            Aircraft.Wing.SPAN: (117.8, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (17.49, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (12.615, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (16.41, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.1397, 'unitless'),
            Aircraft.Wing.FOLDING_AREA: (964.14982163, 'ft**2'),  # not actual GASP value
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: (
                406.64971668264957,
                'ft**3',
            ),  # not actual GASP value
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        # not actual GASP values
        expected_unitless = {
            'nonfolded_taper_ratio': 0.85780985,
            'nonfolded_wing_area': 406.16267837,
            'tc_ratio_mean_folded': 0.14681715,
            'nonfolded_AR': 1.5387923,
        }
        for var_name, expected in expected_unitless.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    def test_fold_and_strut_all_dimensional_location(self):
        """
        Wing with both folds and struts, fold and strut dimensional location specified.

        HAS_FOLD = True
        HAS_STRUT = True
        CHOOSE_FOLD_LOCATION = True
        DIMENSIONAL_LOCATION_SPECIFIED = True
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = True
        """
        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for the full WingGroup subsystem hierarchy.
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')

        inputs = {
            Aircraft.Wing.FOLDED_SPAN: (1, 'ft'),
            Aircraft.Strut.ATTACHMENT_LOCATION: (0, 'ft'),
            Aircraft.Strut.AREA_RATIO: (0.2, 'unitless'),
            Aircraft.Fuselage.AVG_DIAMETER: (10.0, 'ft'),
            Aircraft.Design.GROSS_MASS: (152000.0, 'lbm'),
            Aircraft.Design.WING_LOADING: (128, 'lbf/ft**2'),
            Aircraft.Fuel.WING_FUEL_FRACTION: (0.6, 'unitless'),
            Aircraft.Wing.ASPECT_RATIO: (10.13, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.33, 'unitless'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.11, 'unitless'),
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP: (0.1, 'unitless'),
        }

        prob = self._make_prob(options, inputs)
        prob.run_model()
        tol = 5e-4

        expected_values = {
            Aircraft.Wing.AREA: (1187.5, 'ft**2'),
            Aircraft.Wing.SPAN: (109.6785, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (16.2814, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (11.7430, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (15.4789, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.1067, 'unitless'),
            Aircraft.Wing.FOLDING_AREA: (1171.2684, 'ft**2'),
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: (11.61131, 'ft**3'),
            Aircraft.Strut.LENGTH: (11.18034, 'ft'),
            Aircraft.Strut.CHORD: (10.62132, 'ft'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        expected_unitless = {
            'nonfolded_taper_ratio': 0.9939,
            'nonfolded_wing_area': 16.2316,
            'tc_ratio_mean_folded': 0.10995,
            'nonfolded_AR': 0.06161,
        }
        for var_name, expected in expected_unitless.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-12, rtol=1e-13)

    def test_strut_dimensional_location(self):
        """
        Wing with struts which has dimensional location specified.

        HAS_FOLD = False
        HAS_STRUT = True
        CHOOSE_FOLD_LOCATION = False
        DIMENSIONAL_LOCATION_SPECIFIED = True
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
        """
        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for the full WingGroup subsystem hierarchy.
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=False, units='unitless'
        )

        inputs = {
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),
            Aircraft.Design.WING_LOADING: (128, 'lbf/ft**2'),
            Aircraft.Wing.ASPECT_RATIO: (10.13, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.33, 'unitless'),
            Aircraft.Wing.SWEEP: (25, 'deg'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.15, 'unitless'),
            Aircraft.Fuselage.AVG_DIAMETER: (13.1, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP: (0.12, 'unitless'),
            Aircraft.Strut.ATTACHMENT_LOCATION: (1.0, 'ft'),  # not actual GASP value
            Aircraft.Strut.AREA_RATIO: (0.021893, 'unitless'),
            Aircraft.Fuel.WING_FUEL_FRACTION: (0.6, 'unitless'),
        }

        prob = self._make_prob(options, inputs)
        prob.run_model()

        tol = 5e-4

        # THICKNESS_TO_CHORD_UNWEIGHTED is slightly different from the GASP output value,
        # likely due to rounding error
        expected_values = {
            Aircraft.Wing.AREA: (1370.3, 'ft**2'),
            Aircraft.Wing.SPAN: (117.8, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (17.49, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (12.615, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (16.41, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.1397, 'unitless'),
            Aircraft.Strut.LENGTH: (14.42957033, 'ft'),  # not actual GASP value
            Aircraft.Strut.CHORD: (1.03953199, 'ft'),  # not actual GASP value
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class BWBWingGroupTestCase(unittest.TestCase):
    """Tests for BWBWingGroup."""

    def setUp(self):
        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for the full BWBWingGroup subsystem hierarchy.
        options = AviaryValues()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=False, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless')
        options.set_val(Aircraft.Design.TYPE, val='transport', units='unitless')
        options.set_val(Settings.VERBOSITY, val=Verbosity.BRIEF)
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', BWBWingGroup(), promotes=['*'])

        # Input values below match the "large single aisle 1" GASP reference case used elsewhere in this file; no _MetaData defaults are relied on.

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, 70.0, units='lbf/ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')

        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.385, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.45, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')

        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, 22.9724445, units='ft')

        prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118, units='ft')

        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix.

        HAS_FOLD = True
        HAS_STRUT = False
        CHOOSE_FOLD_LOCATION = True
        FOLD_DIMENSIONAL_LOCATION_SPECIFIED = True

        Testing GASP data case:
        Aircraft.Wing.AREA -- SW = 2142.9
        Aircraft.Wing.SPAN -- B = 146.4
        Aircraft.Wing.CENTER_CHORD -- CROOT = 23.3
        Aircraft.Wing.AVERAGE_CHORD -- CBARW = 16.22
        Aircraft.Wing.ROOT_CHORD -- CROOTW = 20.0657883
        Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED -- TCM = 0.124
        wing_volume_no_fold -- FVOLW_GEOMX = 783.6
        Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX -- FVOLW_GEOM = 605.9
        Aircraft.Wing.FOLDING_AREA -- SWFOLD = 224.8
        Aircraft.Wing.EXPOSED_AREA -- SW_EXP = 1352.1
        Note: CROOT in GASP matches with Aircraft.Wing.CENTER_CHORD.
        """
        prob = self.prob
        prob.run_model()

        tol = 1e-7

        expected_values = {
            Aircraft.Wing.AREA: (2142.85714286, 'ft**2'),
            Aircraft.Wing.SPAN: (146.38501094, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (22.97244452, 'ft'),
            Aircraft.Wing.AVERAGE_CHORD: (16.2200522, 'ft'),
            Aircraft.Wing.ROOT_CHORD: (20.33371617, 'ft'),
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: (0.13596576, 'unitless'),
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: (605.90781747, 'ft**3'),
            Aircraft.Wing.FOLDING_AREA: (224.82529025, 'ft**2'),
            Aircraft.Wing.EXPOSED_AREA: (1352.1135998, 'ft**2'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        with self.subTest(var='wing_volume_no_fold'):
            assert_near_equal(prob['wing_volume_no_fold'], 783.62100035, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class ExposedWingTestCase(unittest.TestCase):
    """Tests for ExposedWing."""

    def _make_prob(self, design_type, height_to_width_ratio):
        # Options below are set explicitly (hardcoded from _MetaData defaults) so this test
        # does not depend on any _MetaData drift for the ExposedWing component.
        options = AviaryValues()
        options.set_val(Aircraft.Design.TYPE, val=design_type, units='unitless')
        options.set_val(Settings.VERBOSITY, val=Verbosity.BRIEF)

        prob = om.Problem()
        prob.model.add_subsystem(
            'expo_wing',
            ExposedWing(),
            promotes=['*'],
        )

        # Input values below (hardcoded, not read from _MetaData).
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, height_to_width_ratio, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.274439991, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_bwb_middle(self):
        """BWB case."""
        prob = self._make_prob(design_type='BWB', height_to_width_ratio=0.25970)
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1352.11359987, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=5e-11)

    def test_case_middle(self):
        """Tube + Wing case, test in the range (epsilon, 1.0 - epsilon)."""
        prob = self._make_prob(design_type='transport', height_to_width_ratio=1.0)
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1352.1135998, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=5e-11)

    def test_case_left(self):
        """Tube + Wing case, test in the range (0.0, epsilon)."""
        prob = self._make_prob(design_type='transport', height_to_width_ratio=1.0)
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.049, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1781.29634277, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=5e-11)

    def test_case_right(self):
        """Tube + Wing case, test in the range (1.0 - epsilon, 1.0)."""
        prob = self._make_prob(design_type='transport', height_to_width_ratio=1.0)
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.951, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1781.29634277, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-10, rtol=5e-12)


if __name__ == '__main__':
    unittest.main()
