import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission

# this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix


@use_tempdirs
class SizeGroupTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, val=1.028233)
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Fuselage.AVG_DIAMETER: 157.2,
            'cabin_height': 13.1,
            'cabin_len': 72.1,
            'nose_height': 8.6,
            Aircraft.Fuselage.LENGTH: 129.5,  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
            Aircraft.Fuselage.WETTED_AREA: 4639.57,
            Aircraft.TailBoom.LENGTH: 129.5,  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
            Aircraft.Wing.AREA: 1370.3,
            Aircraft.Wing.SPAN: 117.8,
            Aircraft.Wing.CENTER_CHORD: 17.49,
            Aircraft.Wing.AVERAGE_CHORD: 12.615,
            Aircraft.Wing.ROOT_CHORD: 16.41,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.1397,  # not exact GASP value, likely due to rounding error
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 1114,
            Aircraft.HorizontalTail.AREA: 375.9,
            Aircraft.HorizontalTail.SPAN: 42.25,
            Aircraft.HorizontalTail.ROOT_CHORD: 13.16130387591471,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 9.57573,
            Aircraft.HorizontalTail.MOMENT_ARM: 54.7,
            Aircraft.VerticalTail.AREA: 469.3,
            Aircraft.VerticalTail.SPAN: 28,
            Aircraft.VerticalTail.ROOT_CHORD: 18.61267549773935,
            Aircraft.VerticalTail.AVERAGE_CHORD: 16.83022,
            Aircraft.VerticalTail.MOMENT_ARM: 49.9,
            Aircraft.Nacelle.AVG_DIAMETER: 7.35,
            Aircraft.Nacelle.AVG_LENGTH: 14.7,
            Aircraft.Nacelle.SURFACE_AREA: 339.58,
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class SizeGroupTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Strut.ATTACHMENT_LOCATION, val=12, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.021893, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, val=1.028233)
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        expected_values = {
            Aircraft.Fuselage.AVG_DIAMETER: 157.2,  # not actual GASP value
            'cabin_height': 13.1,  # not actual GASP value
            'cabin_len': 72.09722222,  # not actual GASP value
            'nose_height': 8.6,  # not actual GASP value
            Aircraft.Fuselage.LENGTH: 129.5,  # not actual GASP value
            Aircraft.Fuselage.WETTED_AREA: 4639.57,  # not actual GASP value
            Aircraft.TailBoom.LENGTH: 129.5,  # not actual GASP value
            Aircraft.Wing.AREA: 1370.3125,  # not actual GASP value
            Aircraft.Wing.SPAN: 117.81878299,  # not actual GASP value
            Aircraft.Wing.CENTER_CHORD: 17.48974356,  # not actual GASP value
            Aircraft.Wing.AVERAGE_CHORD: 12.61453233,  # not actual GASP value
            Aircraft.Wing.ROOT_CHORD: 16.40711451,  # not actual GASP value
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.13965584,  # not actual GASP value
            'nonfolded_taper_ratio': 0.93175961,  # not actual GASP value
            Aircraft.Wing.FOLDING_AREA: 1167.5966191,  # not actual GASP value
            'nonfolded_wing_area': 202.7158809,  # not actual GASP value
            'tc_ratio_mean_folded': 0.14847223,  # not actual GASP value
            'nonfolded_AR': 0.71035382,  # not actual GASP value
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 208.08091725,  # not actual GASP value
            'strut_y': 6,  # not actual GASP value
            Aircraft.Strut.LENGTH: 13.11154072,  # not actual GASP value
            Aircraft.Strut.CHORD: 1.14403031,  # not actual GASP value
            Aircraft.HorizontalTail.AREA: 375.87987047,  # not actual GASP value
            Aircraft.HorizontalTail.SPAN: 42.25434161,  # not actual GASP value
            Aircraft.HorizontalTail.ROOT_CHORD: 13.15924684,  # not actual GASP value
            Aircraft.HorizontalTail.AVERAGE_CHORD: 9.57681709,  # not actual GASP value
            Aircraft.HorizontalTail.MOMENT_ARM: 54.67937726,  # not actual GASP value
            Aircraft.VerticalTail.AREA: 469.31832812,  # not actual GASP value
            Aircraft.VerticalTail.SPAN: 27.99574268,  # not actual GASP value
            Aircraft.VerticalTail.ROOT_CHORD: 18.61623295,  # not actual GASP value
            Aircraft.VerticalTail.AVERAGE_CHORD: 16.83214111,  # not actual GASP value
            Aircraft.VerticalTail.MOMENT_ARM: 49.88094115,  # not actual GASP value
            Aircraft.Nacelle.AVG_DIAMETER: 7.35163168,  # may not be actual GASP value
            Aircraft.Nacelle.AVG_LENGTH: 14.70326336,  # may not be actual GASP value
            Aircraft.Nacelle.SURFACE_AREA: 339.58410134,  # may not be actual GASP value
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


@use_tempdirs
class SizeGroupTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, val=25, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, val=1.028233)
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        expected_values = {
            Aircraft.Fuselage.AVG_DIAMETER: 56.2,  # not actual GASP value
            'cabin_height': 9.18333,  # not actual GASP value
            'cabin_len': 435,  # not actual GASP value
            'nose_height': 4.68333,  # not actual GASP value
            Aircraft.Fuselage.LENGTH: 476.7333,  # not actual GASP value
            Aircraft.Fuselage.WETTED_AREA: 13400.44,  # not actual GASP value
            Aircraft.TailBoom.LENGTH: 476.7333,  # not actual GASP value
            Aircraft.Wing.AREA: 1370.3125,  # not actual GASP value
            Aircraft.Wing.SPAN: 117.81878299,  # not actual GASP value
            Aircraft.Wing.CENTER_CHORD: 17.48974356,  # not actual GASP value
            Aircraft.Wing.AVERAGE_CHORD: 12.61453233,  # not actual GASP value
            Aircraft.Wing.ROOT_CHORD: 16.988,  # not actual GASP value
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.14151,  # not actual GASP value
            'nonfolded_taper_ratio': 0.85783252,  # not actual GASP value
            Aircraft.Wing.FOLDING_AREA: 964.14982163,  # not actual GASP value
            'nonfolded_wing_area': 406.16267837,  # not actual GASP value
            'tc_ratio_mean_folded': 0.14681715,  # not actual GASP value
            'nonfolded_AR': 1.5387923,  # not actual GASP value
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 406.53567274,  # not actual GASP value
            Aircraft.HorizontalTail.AREA: 298.484,  # not actual GASP value
            Aircraft.HorizontalTail.SPAN: 37.654,  # not actual GASP value
            Aircraft.HorizontalTail.ROOT_CHORD: 11.7265,  # not actual GASP value
            Aircraft.HorizontalTail.AVERAGE_CHORD: 8.5341,  # not actual GASP value
            Aircraft.HorizontalTail.MOMENT_ARM: 54.67937726,  # not actual GASP value
            Aircraft.VerticalTail.AREA: 297.003,  # not actual GASP value
            Aircraft.VerticalTail.SPAN: 22.2709,  # not actual GASP value
            Aircraft.VerticalTail.ROOT_CHORD: 14.8094,  # not actual GASP value
            Aircraft.VerticalTail.AVERAGE_CHORD: 13.3902,  # not actual GASP value
            Aircraft.VerticalTail.MOMENT_ARM: 49.88094115,  # not actual GASP value
            Aircraft.Nacelle.AVG_DIAMETER: 7.35163168,  # may not be actual GASP value
            Aircraft.Nacelle.AVG_LENGTH: 14.70326336,  # may not be actual GASP value
            Aircraft.Nacelle.SURFACE_AREA: 339.58410134,  # may not be actual GASP value
            Aircraft.Electrical.HYBRID_CABLE_LENGTH: 50.6032,  # not actual GASP value
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-12)


@use_tempdirs
class SizeGroupTestCase4(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=False, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.021893, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, val=1.028233)
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        expected_values = {
            Aircraft.Fuselage.AVG_DIAMETER: 56.2,  # not actual GASP value
            'cabin_height': 9.18333,  # not actual GASP value
            'cabin_len': 435,  # not actual GASP value
            'nose_height': 4.68333,  # not actual GASP value
            Aircraft.Fuselage.LENGTH: 476.7333,  # not actual GASP value
            Aircraft.Fuselage.WETTED_AREA: 13400.44,  # not actual GASP value
            Aircraft.TailBoom.LENGTH: 476.7333,  # not actual GASP value
            Aircraft.Wing.AREA: 1370.3125,  # not actual GASP value
            Aircraft.Wing.SPAN: 117.81878299,  # not actual GASP value
            Aircraft.Wing.CENTER_CHORD: 17.48974356,  # not actual GASP value
            Aircraft.Wing.AVERAGE_CHORD: 12.61453233,  # not actual GASP value
            Aircraft.Wing.ROOT_CHORD: 16.988,  # not actual GASP value
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.14151,  # not actual GASP value
            'strut_y': 0,  # not actual GASP value
            Aircraft.Strut.LENGTH: 5.2361,  # not actual GASP value
            Aircraft.Strut.CHORD: 2.8647,  # not actual GASP value
            Aircraft.HorizontalTail.AREA: 298.484,  # not actual GASP value
            Aircraft.HorizontalTail.SPAN: 37.654,  # not actual GASP value
            Aircraft.HorizontalTail.ROOT_CHORD: 11.7265,  # not actual GASP value
            Aircraft.HorizontalTail.AVERAGE_CHORD: 8.5341,  # not actual GASP value
            Aircraft.HorizontalTail.MOMENT_ARM: 54.67937726,  # not actual GASP value
            Aircraft.VerticalTail.AREA: 297.003,  # not actual GASP value
            Aircraft.VerticalTail.SPAN: 22.2709,  # not actual GASP value
            Aircraft.VerticalTail.ROOT_CHORD: 14.8094,  # not actual GASP value
            Aircraft.VerticalTail.AVERAGE_CHORD: 13.3902,  # not actual GASP value
            Aircraft.VerticalTail.MOMENT_ARM: 49.88094115,  # not actual GASP value
            Aircraft.Nacelle.AVG_DIAMETER: 7.35163168,  # may not be actual GASP value
            Aircraft.Nacelle.AVG_LENGTH: 14.70326336,  # may not be actual GASP value
            Aircraft.Nacelle.SURFACE_AREA: 339.58410134,  # may not be actual GASP value
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class BWBSizeGroupTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, val=False, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 18)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 36, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 32, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 21, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, 70.0, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, 1.705, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, 0.366, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, 0.366, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.45, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, 0.000001, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.015, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 5.0, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 0.6, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.75, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, 0.5463, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 5.2615, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, 1.705, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2205, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.3588, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0.0, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, 0.0, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Testing GASP data case:
        Aircraft.Fuselage.AVG_DIAMETER -- SWF = 38.00
        fuselage.cabin_height -- HC = 9.87
        fuselage.cabin_len -- LC = 43.8
        fuselage.parameters1.nose_height -- HN = 4.87
        Aircraft.Fuselage.LENGTH -- ELF = 71.52
        Aircraft.Fuselage.WETTED_AREA -- SF = 4574.
        Aircraft.TailBoom.LENGTH -- ELFFC = 71.5
        Aircraft.Wing.AREA -- SW = 2142.9
        Aircraft.Wing.SPAN -- B = 146.4
        Aircraft.Wing.CENTER_CHORD -- CROOT = 23.3
        Aircraft.Wing.AVERAGE_CHORD -- CBARW = 16.22
        Aircraft.Wing.ROOT_CHORD -- CROOTW = 20.0657883
        Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED TCM = 0.124
        wing.wing_volume_no_fold -- FVOLW_GEOMX = 783.6
        Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX -- FVOLW_GEOM = 605.9
        Aircraft.Wing.EXPOSED_AREA -- SW_EXP = 1352.1
        Aircraft.HorizontalTail.AREA -- SHT = 0.
        Aircraft.HorizontalTail.SPAN -- BHT = 0.04
        Aircraft.HorizontalTail.ROOT_CHORD -- CRCLHT = 0.0383644775
        Aircraft.HorizontalTail.AVERAGE_CHORD -- CBARHT = 0.03
        Aircraft.HorizontalTail.MOMENT_ARM -- ELTH = 29.7
        Aircraft.VerticalTail.AREA -- SVT = 169.1
        Aircraft.VerticalTail.SPAN -- BVT = 16.98
        Aircraft.VerticalTail.ROOT_CHORD -- CRCLVT = 14.5819016
        Aircraft.VerticalTail.AVERAGE_CHORD -- CBARVT = 10.67
        Aircraft.VerticalTail.MOMENT_ARM -- ELTV = 27.8
        Aircraft.Nacelle.AVG_DIAMETER -- DBARN = 6.95
        Aircraft.Nacelle.AVG_LENGTH -- ELN = 9.44
        Aircraft.Nacelle.SURFACE_AREA -- SN = 205.965 (for one engine)
        """
        self.prob.run_model()

        tol = 1e-4

        # BWBFuselageGroup
        expected_values = {
            Aircraft.Fuselage.AVG_DIAMETER: 38,
            'cabin_height': 9.86859989,
            'nose_height': 4.86859989,
            Aircraft.Fuselage.LENGTH: 71.5245514,
            Aircraft.Fuselage.WETTED_AREA: 4573.42578,
            Aircraft.TailBoom.LENGTH: 71.5245514,
            Aircraft.Wing.AREA: 2142.85714286,
            Aircraft.Wing.SPAN: 146.38501094,
            Aircraft.Wing.CENTER_CHORD: 22.97244452,
            Aircraft.Wing.AVERAGE_CHORD: 16.2200522,
            Aircraft.Wing.ROOT_CHORD: 20.33371617,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.13596576,
            'wing_volume_no_fold': 783.62100035,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 605.90781747,
            Aircraft.Wing.EXPOSED_AREA: 1352.1135998,
            Aircraft.HorizontalTail.AREA: 0.00117064,
            Aircraft.HorizontalTail.SPAN: 0.04467601,
            Aircraft.HorizontalTail.ROOT_CHORD: 0.03836448,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 0.02808445,
            Aircraft.HorizontalTail.MOMENT_ARM: 29.69074172,
            Aircraft.VerticalTail.AREA: 169.11964286,
            Aircraft.VerticalTail.SPAN: 16.98084188,
            Aircraft.VerticalTail.ROOT_CHORD: 14.58190052,
            Aircraft.VerticalTail.AVERAGE_CHORD: 10.67457744,
            Aircraft.VerticalTail.MOMENT_ARM: 27.82191598,
            Aircraft.Nacelle.AVG_DIAMETER: 5.33382144,
            Aircraft.Nacelle.AVG_LENGTH: 7.24759657,
            Aircraft.Nacelle.SURFACE_AREA: 121.44575974,
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=3e-9)


if __name__ == '__main__':
    unittest.main()
