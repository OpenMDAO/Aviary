import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.design_load import (
    DesignLoadGroup,
    LiftCurveSlopeAtCruise,
    LoadFactors,
    LoadParameters,
    LoadSpeeds,
)
from aviary.subsystems.mass.gasp_based.design_load import (
    BWBDesignLoadGroup,
    BWBLoadFactors,
    BWBLoadSpeeds,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class LoadSpeedsTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 350, tol)  # bug fixed value
        assert_near_equal(self.prob['vel_c'], 350, tol)  # bug fixed value
        assert_near_equal(self.prob['max_maneuver_factor'], 2.5, tol)  # bug fixed value
        assert_near_equal(self.prob['min_dive_vel'], 420, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=0, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 346.75, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 306.15, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 3.8, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 407.94, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=1, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 401.6, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 315, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 4.4, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 472.5, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase4(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=2, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 320.17, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 294.27, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 6, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 376.67, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class LoadSpeedsTestCase5(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=4, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 350, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 350, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 4, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 420, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase6smooth(
    unittest.TestCase
):  # this is the large single aisle 1 V3 test case (LoadSpeedsTestCase1) with smooth functions
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 350, tol)  # bug fixed value
        assert_near_equal(self.prob['vel_c'], 350, tol)  # bug fixed value
        assert_near_equal(self.prob['max_maneuver_factor'], 2.5, tol)  # bug fixed value
        assert_near_equal(self.prob['min_dive_vel'], 420, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase7smooth(unittest.TestCase):  # TestCase2 with smooth functions
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=0, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 2e-3
        assert_near_equal(self.prob['max_airspeed'], 346.75, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 306.15, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 3.8, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 407.94, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-14, rtol=2e-13)


class LoadSpeedsTestCase8smooth(unittest.TestCase):  # TestCase3 with smooth functions
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=1, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 401.6, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 315, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 4.4, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 472.5, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-7, rtol=1e-6)


class LoadSpeedsTestCase9smooth(unittest.TestCase):  # TestCase4 with smooth functions
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=2, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 320.17, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 294.27, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 6, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 376.67, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-13)


class LoadSpeedsTestCase10smooth(unittest.TestCase):  # TestCase5 with smooth functions
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=4, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # not actual bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_airspeed'], 350, tol)  # not actual GASP value
        assert_near_equal(self.prob['vel_c'], 350, tol)  # not actual GASP value
        assert_near_equal(self.prob['max_maneuver_factor'], 4, tol)  # not actual GASP value
        assert_near_equal(self.prob['min_dive_vel'], 420, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadParametersTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('params', LoadParameters(), promotes=['*'])

        self.prob.model.set_input_defaults('vel_c', val=350, units='kn')  # bug fixed value
        self.prob.model.set_input_defaults('max_airspeed', val=350, units='kn')  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 4e-4
        assert_near_equal(self.prob['max_mach'], 0.9, tol)  # bug fixed value
        assert_near_equal(self.prob['density_ratio'], 0.533, tol)  # bug fixed value
        assert_near_equal(self.prob['V9'], 350, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadParametersTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=2, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=30000, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('params', LoadParameters(), promotes=['*'])

        self.prob.model.set_input_defaults('vel_c', val=350, units='mi/h')  # bug fixed value
        self.prob.model.set_input_defaults('max_airspeed', val=350, units='kn')  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_mach'], 0.824, tol)  # not actual GASP value
        assert_near_equal(self.prob['density_ratio'], 0.682, tol)  # not actual GASP value
        assert_near_equal(self.prob['V9'], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadParametersTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=4, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=22000, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('params', LoadParameters(), promotes=['*'])

        self.prob.model.set_input_defaults('vel_c', val=350, units='mi/h')  # bug fixed value
        self.prob.model.set_input_defaults('max_airspeed', val=350, units='kn')  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(self.prob['max_mach'], 0.7197, tol)  # not actual GASP value
        assert_near_equal(self.prob['density_ratio'], 0.6073, tol)  # not actual GASP value
        assert_near_equal(self.prob['V9'], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadParametersTestCase4smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'params',
            LoadParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('vel_c', val=350, units='kn')  # bug fixed value
        self.prob.model.set_input_defaults('max_airspeed', val=350, units='kn')  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 6e-4
        assert_near_equal(self.prob['max_mach'], 0.9, tol)  # bug fixed value
        assert_near_equal(self.prob['density_ratio'], 0.533, tol)  # bug fixed value
        assert_near_equal(self.prob['V9'], 350, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=5e-14)


class LoadParametersTestCase5smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=2, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=30000, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'params',
            LoadParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('vel_c', val=350, units='mi/h')  # bug fixed value
        self.prob.model.set_input_defaults('max_airspeed', val=350, units='kn')  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_mach'], 0.824, tol)  # not actual GASP value
        assert_near_equal(self.prob['density_ratio'], 0.682, tol)  # not actual GASP value
        assert_near_equal(self.prob['V9'], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadParametersTestCase6smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=4, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=22000, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'params',
            LoadParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('vel_c', val=350, units='mi/h')  # bug fixed value
        self.prob.model.set_input_defaults('max_airspeed', val=350, units='kn')  # bug fixed value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_mach'], 0.7197, tol)  # not actual GASP value
        assert_near_equal(self.prob['density_ratio'], 0.6073, tol)  # not actual GASP value
        assert_near_equal(self.prob['V9'], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-8)


class LiftCurveSlopeAtCruiseTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('factors', LiftCurveSlopeAtCruise(), promotes=['*'])
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=0.436, units='rad')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_slope(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Design.LIFT_CURVE_SLOPE], 6.3967, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadFactorsTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('factors', LoadFactors(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units='lbf/ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'density_ratio', val=0.533, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults('V9', val=350, units='kn')  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'max_maneuver_factor', val=2.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units='1/rad'
        )  # bug fixed value and original value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.9502, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class LoadFactorsTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('factors', LoadFactors(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            'density_ratio', val=0.5328, units='unitless'
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults(
            'V9', val=304.14, units='kn'
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults('max_maneuver_factor', val=2.5, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=12.615, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units='1/rad'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadFactorsTestCase3smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'factors',
            LoadFactors(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units='lbf/ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'density_ratio', val=0.533, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults('V9', val=350, units='kn')  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'max_maneuver_factor', val=2.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units='1/rad'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 4e-4
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.9502, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-13, rtol=2e-13)


class LoadFactorsTestCase4smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'factors',
            LoadFactors(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units='lbf/ft**2'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            'density_ratio', val=0.5328, units='unitless'
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults(
            'V9', val=304.14, units='kn'
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults('max_maneuver_factor', val=2.5, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=12.615, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units='1/rad'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class DesignLoadGroupTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'Dload',
            DesignLoadGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units='lbf/ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['max_mach'], 0.9, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class DesignLoadGroupTestCase2smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'Dload',
            DesignLoadGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )  # bug fixed and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units='lbf/ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=0.436, units='rad')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 6e-4
        assert_near_equal(self.prob['max_mach'], 0.9, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.7397, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-13, rtol=5e-12)


class BWBLoadSpeedsTestCATD3(unittest.TestCase):
    """PART25_STRUCTURAL_CATEGORY = 3."""

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')  # default
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            BWBLoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """GASP data"""

        self.options.set_val(
            Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
            val=3,
            units='unitless',
        )
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 2.5, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_case2(self):
        """Aviary enhanced algorithms"""

        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 2.5, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 2.5, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 2.5, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class BWBLoadSpeedsTestCATD0(unittest.TestCase):
    """PART25_STRUCTURAL_CATEGORY = 0."""

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=0, units='unitless')
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')  # default

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            BWBLoadSpeeds(),
            promotes=['*'],
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, 402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.1136, units='ft**2')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """GASP data"""

        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 336.68277925, tol)
        assert_near_equal(self.prob['vel_c'], 294.8987279, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 3.8, tol)
        assert_near_equal(self.prob['min_dive_vel'], 396.09738736, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_case2(self):
        """Aviary enhanced algorithms"""

        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 336.68277925, tol)
        assert_near_equal(self.prob['vel_c'], 294.8987279, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 3.8, tol)
        assert_near_equal(self.prob['min_dive_vel'], 396.09738736, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 374.85124973, tol)
        assert_near_equal(self.prob['vel_c'], 315.0010502, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 3.8, tol)
        assert_near_equal(self.prob['min_dive_vel'], 441.00147028, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=5e-14)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 336.72148918, tol)
        assert_near_equal(self.prob['vel_c'], 294.93269834, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 3.8, tol)
        assert_near_equal(self.prob['min_dive_vel'], 396.14292844, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=5e-14)


class BWBLoadSpeedsTestCATD1(unittest.TestCase):
    """PART25_STRUCTURAL_CATEGORY = 1."""

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=1, units='unitless')
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')  # default

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            BWBLoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, 402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.1136, units='ft**2')

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """GASP data"""

        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 333.25575724, tol)
        assert_near_equal(self.prob['vel_c'], 294.8987279, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.4, tol)
        assert_near_equal(self.prob['min_dive_vel'], 392.06559676, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_case2(self):
        """Aviary enhanced algorithms"""

        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 401.625, tol)
        assert_near_equal(self.prob['vel_c'], 315.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.4, tol)
        assert_near_equal(self.prob['min_dive_vel'], 472.5, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 401.626339, tol)
        assert_near_equal(self.prob['vel_c'], 315.0010502, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.4, tol)
        assert_near_equal(self.prob['min_dive_vel'], 472.5015753, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        # assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 333.29393309, tol)
        assert_near_equal(self.prob['vel_c'], 294.93269834, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.4, tol)
        assert_near_equal(self.prob['min_dive_vel'], 392.11050952, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=5e-14)


class BWBLoadSpeedsTestCATD2(unittest.TestCase):
    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=2, units='unitless')
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')  # default

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            BWBLoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, 402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.1136, units='ft**2')

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """GASP data"""

        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 326.68543372, tol)
        assert_near_equal(self.prob['vel_c'], 290.57871182, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 6.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 384.33580438, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_case2(self):
        """Aviary enhanced algorithms"""

        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 415.0125, tol)
        assert_near_equal(self.prob['vel_c'], 315.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 6.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 488.25, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 415.01250012, tol)
        assert_near_equal(self.prob['vel_c'], 315.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 6.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 488.25, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=5e-14)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 326.69562597, tol)
        assert_near_equal(self.prob['vel_c'], 290.5891973, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 6.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 384.34779526, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=5e-14)


class BWBLoadSpeedsTestCATD4(unittest.TestCase):
    def setUp(self):
        self.options = get_option_defaults()
        # In this case, the value of PART25_STRUCTURAL_CATEGORY is used as max_maneuver_factor
        self.options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, val=4, units='unitless')
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')  # default

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'speeds',
            BWBLoadSpeeds(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, 402.5, units='mi/h'
        )

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """GASP data"""

        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

    def test_case2(self):
        """Aviary enhanced algorithms"""

        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )  # default
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=True, units='unitless')
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_airspeed'], 350.0, tol)
        assert_near_equal(self.prob['vel_c'], 350.0, tol)
        assert_near_equal(self.prob['max_maneuver_factor'], 4.0, tol)
        assert_near_equal(self.prob['min_dive_vel'], 420.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class BWBLoadFactorsTestCaseNonsmooth(unittest.TestCase):
    """GASP data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem('factors', BWBLoadFactors(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.1136, units='ft**2')
        prob.model.set_input_defaults('density_ratio', 0.692859828, units='unitless')
        prob.model.set_input_defaults('V9', 350.0, units='kn')
        prob.model.set_input_defaults('min_dive_vel', 420, units='kn')
        prob.model.set_input_defaults('max_maneuver_factor', 2.5, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, 16.2200546, units='ft')
        prob.model.set_input_defaults(Aircraft.Design.LIFT_CURVE_SLOPE, 5.94851685, units='1/rad')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Test the simplest scenario
        """
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)

    def test_case2(self):
        """
        Test all other if-else branches comparing
        cruise_load_factor vs dive_load_factor, and gust_load_factor vs max_maneuver_factor
        """

        # Case 2A
        self.prob.set_val('density_ratio', 0.53281, units='unitless')
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.77353191, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)

        # Case 2B
        self.prob.set_val('density_ratio', 0.53281, units='unitless')
        self.prob.set_val('V9', 210.0, units='kn')
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)

        # Case 2C
        self.prob.set_val('density_ratio', 0.53281, units='unitless')
        self.prob.set_val('V9', 209.9, units='kn')
        self.prob.set_val('max_maneuver_factor', 1.9, units='unitless')
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 2.86411929, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class BWBLoadFactorsTestCaseSmooth(unittest.TestCase):
    """Test for smoothing technique"""

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('factors', BWBLoadFactors(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, val=1352.1136, units='ft**2')
        prob.model.set_input_defaults('density_ratio', val=0.692859828, units='unitless')
        prob.model.set_input_defaults('V9', val=350.0, units='kn')
        prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        prob.model.set_input_defaults('max_maneuver_factor', val=2.5, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=16.2200546, units='ft')
        prob.model.set_input_defaults(Aircraft.Design.LIFT_CURVE_SLOPE, val=5.94852, units='1/rad')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Test the simplest scenario
        """
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


@use_tempdirs
class BWBDesignLoadGroupTestCaseNonsmooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        prob = self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'Dload',
            BWBDesignLoadGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h')

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, val=1352.1136, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=12.71, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')

        setup_model_options(self.prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_mach'], 0.9, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


@use_tempdirs
class BWBDesignLoadGroupTestCaseSmooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        prob = self.prob = om.Problem()

        prob.model.add_subsystem(
            'Dload',
            BWBDesignLoadGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h')

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, val=1352.1136, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, val=12.71, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=0.436, units='rad')
        prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')

        setup_model_options(self.prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['max_mach'], 0.90046425, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.97744787, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-13, rtol=5e-12)


if __name__ == '__main__':
    unittest.main()
