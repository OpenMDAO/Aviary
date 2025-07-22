import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.wing import (
    BWBWingVolume,
    BWBWingFoldVolume,
    BWBWingGroup,
    ExposedWing,
    WingFoldArea,
    WingFoldVolume,
    WingGroup,
    WingParameters,
    WingSize,
    WingVolume,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class WingSizeTestCase1(
    unittest.TestCase
):  # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('size', WingSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingSizeTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.geometry.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.geometry.gasp_based.wing as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('size', WingSize(), promotes=['*'])
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBWingSizeTestCase1(unittest.TestCase):
    """
    BWB model
    """

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('size', WingSize(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.LOADING, 70.0, units='lbf/ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Wing.AREA], 2142.8571, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 146.38501, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingParametersTestCase1(
    unittest.TestCase
):  # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('parameters', WingParameters(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingParametersTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('parameters', WingParameters(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBWingParametersTestCase1(unittest.TestCase):
    """Test BWB data for BWBWingParameters"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('parameters', WingParameters(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.385, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 22.97244663, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 16.2200537, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 20.33371818, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.13596576, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingVolumeTestCase1(
    unittest.TestCase
):  # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=False, units='unitless')
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_vol', WingVolume(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBWingVolumeTestCase(unittest.TestCase):
    """Test BWB data for BWBWingVolume"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=False, units='unitless')
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wing_vol', BWBWingVolume(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.385, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, 22.9724445, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.45, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 783.6209, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class WingFoldAreaTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingFoldArea(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults('strut_y', val=25, units='ft')  # not actual GASP value
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 620.04352246, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingFoldVolumeTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingFoldVolume(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults('strut_y', val=25, units='ft')  # not actual GASP value
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, val=620.0435, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob['nonfolded_taper_ratio'], 0.71561969, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['nonfolded_wing_area'], 750.25647754, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['tc_ratio_mean_folded'], 0.14363328, tol
        )  # not actual GASP value
        assert_near_equal(self.prob['nonfolded_AR'], 3.33219382, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 712.3428037422319, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class WingFoldAreaTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingFoldArea(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units='ft'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 964.0812219, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class BWBWingFoldAreaTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            WingFoldArea(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.385, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.FOLDING_AREA], 224.82521003, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingFoldVolumeTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingFoldVolume(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults('strut_y', val=25, units='ft')  # not actual GASP value
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, val=620.0435, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob['nonfolded_taper_ratio'], 0.71561969, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['nonfolded_wing_area'], 750.25647754, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['tc_ratio_mean_folded'], 0.14363328, tol
        )  # not actual GASP value
        assert_near_equal(self.prob['nonfolded_AR'], 3.33219382, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 712.3428037422319, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class WingFoldAreaTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingFoldArea(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units='ft'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 964.0812219, tol
        )  # not actual GASP value


class WingFoldVolumeTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingFoldVolume(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units='ft'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDING_AREA, 964.0812219, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob['nonfolded_taper_ratio'], 0.85780985, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['nonfolded_wing_area'], 406.2187781, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['tc_ratio_mean_folded'], 0.14681664, tol
        )  # not actual GASP value
        assert_near_equal(self.prob['nonfolded_AR'], 1.53857978, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 406.64971668264957, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class BWBWingFoldVolumeTestCase1(unittest.TestCase):
    """
    Test against GASP BWB model, CHOOSE_FOLD_LOCATION = False
    This case should not be allowed, but it is tested anyway.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            BWBWingFoldVolume(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')
        prob.model.set_input_defaults('strut_y', val=59, units='ft')
        prob.model.set_input_defaults('wing_volume_no_fold', val=783.6209, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')

        setup_model_options(prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-7

        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.90774, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class BWBWingFoldVolumeTestCase2(unittest.TestCase):
    """
    Test against GASP BWB model, CHOOSE_FOLD_LOCATION = True
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            BWBWingFoldVolume(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118.0, units='ft')
        prob.model.set_input_defaults('wing_volume_no_fold', val=783.6209, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        self.prob.run_model()

        tol = 1e-7

        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.90774, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class WingGroupTestCase1(unittest.TestCase):
    """
    actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    HAS_FOLD = False
    HAS_STRUT = False
    CHOOSE_FOLD_LOCATION = True
    DIMENSIONAL_LOCATION_SPECIFIED = False
    FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', WingGroup(), promotes=['*'])

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingGroupTestCase2(unittest.TestCase):
    """
    Wing with both folds and struts which has fold dimensional location and strut dimensional location specified
    with the fold at the strut connection.
    HAS_FOLD = True
    HAS_STRUT = True
    CHOOSE_FOLD_LOCATION = False
    DIMENSIONAL_LOCATION_SPECIFIED = True
    FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.02189, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=1.0, units='ft'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        assert_near_equal(
            self.prob['fold_vol.nonfolded_taper_ratio'], 0.9943133, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 1352.8724859, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['fold_vol.nonfolded_wing_area'], 17.4400141, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['fold_vol.tc_ratio_mean_folded'], 0.14987269, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['fold_vol.nonfolded_AR'], 0.0573394, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 18.26837098, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 14.42957033, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Strut.CHORD], 1.03953199, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=7e-12, rtol=1e-12)


class WingGroupTestCase3(unittest.TestCase):
    """
    Wing with folds which has dimensional location specified.
    HAS_FOLD = True
    HAS_STRUT = False
    CHOOSE_FOLD_LOCATION = True
    DIMENSIONAL_LOCATION_SPECIFIED = True
    FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units='ft'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        assert_near_equal(
            self.prob['fold_vol.nonfolded_taper_ratio'], 0.85780985, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 964.14982163, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['fold_vol.nonfolded_wing_area'], 406.16267837, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['fold_vol.tc_ratio_mean_folded'], 0.14681715, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob['fold_vol.nonfolded_AR'], 1.5387923, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 406.64971668264957, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingGroupTestCase4(unittest.TestCase):
    """
    Wing with both folds and struts which has fold dimensional location and strut dimensional location specified.
    HAS_FOLD = True
    HAS_STRUT = True
    CHOOSE_FOLD_LOCATION = True
    DIMENSIONAL_LOCATION_SPECIFIED = True
    FOLD_DIMENSIONAL_LOCATION_SPECIFIED = True
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, val=1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Strut.ATTACHMENT_LOCATION, val=0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Strut.AREA_RATIO, val=0.2, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=10.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=152000.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, val=0.1, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1187.5, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 109.6785, tol)
        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.2814, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 11.7430, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 15.4789, tol)
        assert_near_equal(self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1067, tol)
        assert_near_equal(self.prob['fold_vol.nonfolded_taper_ratio'], 0.9939, tol)
        assert_near_equal(self.prob[Aircraft.Wing.FOLDING_AREA], 1171.2684, tol)
        assert_near_equal(self.prob['fold_vol.nonfolded_wing_area'], 16.2316, tol)
        assert_near_equal(self.prob['fold_vol.tc_ratio_mean_folded'], 0.10995, tol)
        assert_near_equal(self.prob['fold_vol.nonfolded_AR'], 0.06161, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 11.61131, tol)
        assert_near_equal(self.prob[Aircraft.Strut.LENGTH], 11.18034, tol)
        assert_near_equal(self.prob[Aircraft.Strut.CHORD], 10.62132, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-12, rtol=1e-13)


class WingGroupTestCase5(unittest.TestCase):
    """
    Wing with struts which has dimentional location specified.
    HAS_FOLD = False
    HAS_STRUT = True
    CHOOSE_FOLD_LOCATION = False
    DIMENSIONAL_LOCATION_SPECIFIED = True
    FOLD_DIMENSIONAL_LOCATION_SPECIFIED = False
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            WingGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, 128, units='lbf/ft**2')

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.2, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=150, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=1.0, units='ft'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.021893, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 14.42957033, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Strut.CHORD], 1.03953199, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class BWBWingGroupTestCase1(unittest.TestCase):
    """
    actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    HAS_FOLD = True
    HAS_STRUT = False
    CHOOSE_FOLD_LOCATION = True
    FOLD_DIMENSIONAL_LOCATION_SPECIFIED = True
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', BWBWingGroup(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.LOADING, 70.0, units='lbf/ft**2')
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
        Note: CROOT in GASP matches with Aircraft.Wing.CENTER_CHORD
        """
        prob = self.prob
        prob.run_model()

        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.AREA], 2142.85714286, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 146.38501094, tol)
        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 22.97244452, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 16.2200522, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 20.33371617, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.13596576, tol)
        assert_near_equal(prob['wing_volume_no_fold'], 783.62100035, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.90781747, tol)
        assert_near_equal(prob[Aircraft.Wing.FOLDING_AREA], 224.82529025, tol)
        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1352.1135998, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class BWBExposedWingTestCase(unittest.TestCase):
    """BWB case."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'bwb_expo_wing',
            ExposedWing(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.274439991, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case_middle(self):
        prob = self.prob
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1352.11359987, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=5e-11)


class ExposedWingTestCase(unittest.TestCase):
    """Tube + Wing case."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='transport', units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'expo_wing',
            ExposedWing(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.274439991, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case_middle(self):
        """Test in the range (epsilon, 1.0 - epsilon)."""
        prob = self.prob
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1352.1135998, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=5e-11)

    def test_case_left(self):
        """Test in the range (0.0, epsilon)."""
        prob = self.prob
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.049, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1781.29634277, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=5e-11)

    def test_case_right(self):
        prob = self.prob
        """Test in the range (1.0 - epsilon, 1.0)."""
        prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.951, units='unitless')
        prob.run_model()
        tol = 1e-7

        assert_near_equal(prob[Aircraft.Wing.EXPOSED_AREA], 1781.29634277, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-10, rtol=5e-12)


if __name__ == '__main__':
    unittest.main()
