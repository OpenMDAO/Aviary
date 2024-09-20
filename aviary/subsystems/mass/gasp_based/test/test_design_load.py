import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.design_load import (DesignLoadGroup,
                                                           LoadFactors,
                                                           LoadParameters,
                                                           LiftCurveSlopeAtCruise,
                                                           LoadSpeeds)
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class LoadSpeedsTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(aviary_options=get_option_defaults()),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_airspeed"], 350, tol)  # bug fixed value
        assert_near_equal(self.prob["vel_c"], 350, tol)  # bug fixed value
        assert_near_equal(self.prob["max_maneuver_factor"], 2.5, tol)  # bug fixed value
        assert_near_equal(self.prob["min_dive_vel"], 420, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=0, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob["max_airspeed"], 346.75, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 306.15, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 3.8, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["min_dive_vel"], 407.94, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=1, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob["max_airspeed"], 401.6, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 315, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 4.4, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["min_dive_vel"], 472.5, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase4(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=2, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob["max_airspeed"], 320.17, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 294.27, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 6, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["min_dive_vel"], 376.67, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class LoadSpeedsTestCase5(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=4, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_airspeed"], 350, tol)  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 350, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 4, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["min_dive_vel"], 420, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase6smooth(
    unittest.TestCase
):  # this is the large single aisle 1 V3 test case (LoadSpeedsTestCase1) with smooth functions
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(
                aviary_options=options
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_airspeed"], 350, tol)  # bug fixed value
        assert_near_equal(self.prob["vel_c"], 350, tol)  # bug fixed value
        assert_near_equal(self.prob["max_maneuver_factor"], 2.5, tol)  # bug fixed value
        assert_near_equal(self.prob["min_dive_vel"], 420, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadSpeedsTestCase7smooth(unittest.TestCase):  # TestCase2 with smooth functions
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=0, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(
                aviary_options=options
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 2e-3
        assert_near_equal(
            self.prob["max_airspeed"], 346.75, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 306.15, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 3.8, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["min_dive_vel"], 407.94, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class LoadSpeedsTestCase8smooth(unittest.TestCase):  # TestCase3 with smooth functions
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=1, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob["max_airspeed"], 401.6, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 315, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 4.4, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["min_dive_vel"], 472.5, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=5e-7, rtol=1e-6)


class LoadSpeedsTestCase9smooth(unittest.TestCase):  # TestCase4 with smooth functions
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=2, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob["max_airspeed"], 320.17, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 294.27, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 6, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["min_dive_vel"], 376.67, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=2e-14)


class LoadSpeedsTestCase10smooth(unittest.TestCase):  # TestCase5 with smooth functions
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.LOADING_ABOVE_20, val=False, units='unitless')
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=4, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "speeds",
            LoadSpeeds(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # not actual bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_airspeed"], 350, tol)  # not actual GASP value
        assert_near_equal(self.prob["vel_c"], 350, tol)  # not actual GASP value
        assert_near_equal(
            self.prob["max_maneuver_factor"], 4, tol
        )  # not actual GASP value
        assert_near_equal(self.prob["min_dive_vel"], 420, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadParametersTestCase1(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "params", LoadParameters(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            "vel_c", val=350, units="kn"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "max_airspeed", val=350, units="kn"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 4e-4
        assert_near_equal(self.prob["max_mach"], 0.9, tol)  # bug fixed value
        assert_near_equal(self.prob["density_ratio"], 0.533, tol)  # bug fixed value
        assert_near_equal(self.prob["V9"], 350, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadParametersTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=2, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=30000, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "params", LoadParameters(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            "vel_c", val=350, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "max_airspeed", val=350, units="kn"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_mach"], 0.824, tol)  # not actual GASP value
        assert_near_equal(self.prob["density_ratio"], 0.682,
                          tol)  # not actual GASP value
        assert_near_equal(self.prob["V9"], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadParametersTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=4, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=22000, units='ft')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "params", LoadParameters(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            "vel_c", val=350, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "max_airspeed", val=350, units="kn"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(self.prob["max_mach"], 0.7197, tol)  # not actual GASP value
        assert_near_equal(self.prob["density_ratio"], 0.6073,
                          tol)  # not actual GASP value
        assert_near_equal(self.prob["V9"], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadParametersTestCase4smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "params",
            LoadParameters(aviary_options=options,),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "vel_c", val=350, units="kn"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "max_airspeed", val=350, units="kn"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 6e-4
        assert_near_equal(self.prob["max_mach"], 0.9, tol)  # bug fixed value
        assert_near_equal(self.prob["density_ratio"], 0.533, tol)  # bug fixed value
        assert_near_equal(self.prob["V9"], 350, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class LoadParametersTestCase5smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=2, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=30000, units='ft')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "params",
            LoadParameters(aviary_options=options,),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "vel_c", val=350, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "max_airspeed", val=350, units="kn"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_mach"], 0.824, tol)  # not actual GASP value
        assert_near_equal(self.prob["density_ratio"], 0.682,
                          tol)  # not actual GASP value
        assert_near_equal(self.prob["V9"], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


class LoadParametersTestCase6smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY,
                        val=4, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=22000, units='ft')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "params",
            LoadParameters(aviary_options=options,),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            "vel_c", val=350, units="mi/h"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "max_airspeed", val=350, units="kn"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_mach"], 0.7197, tol)  # not actual GASP value
        assert_near_equal(self.prob["density_ratio"], 0.6073,
                          tol)  # not actual GASP value
        assert_near_equal(self.prob["V9"], 304.14, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-8)


class LiftCurveSlopeAtCruiseTest(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "factors", LiftCurveSlopeAtCruise(), promotes=["*"])
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SWEEP, val=0.436, units="rad"
        )
        self.prob.model.set_input_defaults(
            Mission.Design.MACH, val=0.8, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_slope(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Design.LIFT_CURVE_SLOPE], 6.3967, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadFactorsTestCase1(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "factors", LoadFactors(aviary_options=get_option_defaults()), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units="lbf/ft**2"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "density_ratio", val=0.533, units="unitless"
        )  # bug fixed value
        self.prob.model.set_input_defaults("V9", val=350, units="kn")  # bug fixed value
        self.prob.model.set_input_defaults(
            "min_dive_vel", val=420, units="kn"
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            "max_maneuver_factor", val=2.5, units="unitless"
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units="ft"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units="1/rad"
        )  # bug fixed value and original value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.9502, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-14)


class LoadFactorsTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "factors", LoadFactors(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            "density_ratio", val=0.5328, units="unitless"
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults(
            "V9", val=304.14, units="kn"
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults("min_dive_vel", val=420, units="kn")
        self.prob.model.set_input_defaults(
            "max_maneuver_factor", val=2.5, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.615, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units="1/rad")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class LoadFactorsTestCase3smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "factors",
            LoadFactors(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units="lbf/ft**2"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            "density_ratio", val=0.533, units="unitless"
        )  # bug fixed value
        self.prob.model.set_input_defaults("V9", val=350, units="kn")  # bug fixed value
        self.prob.model.set_input_defaults(
            "min_dive_vel", val=420, units="kn"
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            "max_maneuver_factor", val=2.5, units="unitless"
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units="ft"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units="1/rad"
        )  # bug fixed value and original value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 4e-4
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.9502, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=5e-14, rtol=2e-13)


class LoadFactorsTestCase4smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "factors",
            LoadFactors(aviary_options=options,),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            "density_ratio", val=0.5328, units="unitless"
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults(
            "V9", val=304.14, units="kn"
        )  # may not be actual GASP value
        self.prob.model.set_input_defaults("min_dive_vel", val=420, units="kn")
        self.prob.model.set_input_defaults(
            "max_maneuver_factor", val=2.5, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.615, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765, units="1/rad")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class DesignLoadGroupTestCase1(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            "Dload",
            DesignLoadGroup(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units="lbf/ft**2"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units="ft"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["max_mach"], 0.9, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-15, rtol=1e-15)


# this is the large single aisle 1 V3 test case
class DesignLoadGroupTestCase2smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            "Dload",
            DesignLoadGroup(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units="mi/h"
        )  # bug fixed and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=126, units="lbf/ft**2"
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AVERAGE_CHORD, val=12.71, units="ft"
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 6e-4
        assert_near_equal(self.prob["max_mach"], 0.9, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.7397, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-14, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
