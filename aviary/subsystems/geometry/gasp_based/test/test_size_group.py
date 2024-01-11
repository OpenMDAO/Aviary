import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft, Mission

# this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix


class SizeGroupTestCase1(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM,
                        val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "size",
            SizeGroup(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALE_FACTOR, val=1.028233
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.FINENESS, 2, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(self.prob["fuselage.cabin_height"], 13.1, tol)
        assert_near_equal(self.prob["fuselage.cabin_len"], 72.1, tol)
        assert_near_equal(self.prob["fuselage.nose_height"], 8.6, tol)

        # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, tol)
        # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact GASP value, likely due to rounding error
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114, tol)

        assert_near_equal(self.prob[Aircraft.HorizontalTail.AREA], 375.9, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.SPAN], 42.25, tol)
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.16130387591471, tol
        )
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
        assert_near_equal(self.prob[Aircraft.VerticalTail.MOMENT_ARM], 49.9, tol)

        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.35, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 339.58, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)

    @skipIfMissingXDSM('mass_and_sizing_basic_specs/size.json')
    def test_io_wing_group_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "mass_and_sizing_basic_specs/size.json")


class SizeGroupTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Fuselage.PROVIDE_SURFACE_AREA,
                        val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM,
                        val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "size",
            SizeGroup(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=12, units='ft'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.021893, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALE_FACTOR, val=1.028233
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.FINENESS, 2, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.cabin_height"], 13.1, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.cabin_len"], 72.09722222, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.nose_height"], 8.6, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 18558260.55555555, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Wing.AREA], 1370.3125, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.SPAN], 117.81878299, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Wing.CENTER_CHORD], 17.48974356, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.61453233, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.ROOT_CHORD], 16.40711451, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.13965584, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob["wing.fold.nonfolded_taper_ratio"], 0.93175961, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 1167.5966191, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["wing.fold.nonfolded_wing_area"], 202.7158809, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["wing.fold.tc_ratio_mean_folded"], 0.14847223, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["wing.fold.nonfolded_AR"], 0.71035382, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 208.08091725, tol
        )  # not actual GASP value

        assert_near_equal(self.prob["wing.strut_y"], 6, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 13.11154072, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.CHORD], 1.14403031, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AREA], 375.87987047, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.SPAN], 42.25434161, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.15924684, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.57681709, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.67937726, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AREA], 469.31832812, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.SPAN], 27.99574268, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.ROOT_CHORD], 18.61623295, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.83214111, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.MOMENT_ARM], 49.88094115, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.35163168, tol
        )  # may not be actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.70326336, tol
        )  # may not be actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Nacelle.SURFACE_AREA], 339.58410134, tol
        )  # may not be actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)

    @skipIfMissingXDSM('mass_and_sizing_both_specs/size.json')
    def test_io_wing_group_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "mass_and_sizing_both_specs/size.json")


class SizeGroupTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Fuselage.PROVIDE_SURFACE_AREA,
                        val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "size",
            SizeGroup(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALE_FACTOR, val=1.028233
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.FINENESS, 2, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.cabin_height"], 9.18333, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.cabin_len"], 435, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.nose_height"], 4.68333, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 476.7333, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 53601769, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 476.7333, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Wing.AREA], 1370.3125, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.SPAN], 117.81878299, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Wing.CENTER_CHORD], 17.48974356, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.61453233, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.ROOT_CHORD], 16.988, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.14151, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob["wing.fold.nonfolded_taper_ratio"], 0.85783252, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 964.14982163, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["wing.fold.nonfolded_wing_area"], 406.16267837, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["wing.fold.tc_ratio_mean_folded"], 0.14681715, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["wing.fold.nonfolded_AR"], 1.5387923, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 406.53567274, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AREA], 298.484, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.SPAN], 37.654, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 11.7265, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 8.5341, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.67937726, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AREA], 297.003, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.SPAN], 22.2709, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.ROOT_CHORD], 14.8094, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 13.3902, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.MOMENT_ARM], 49.88094115, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.35163168, tol
        )  # may not be actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.70326336, tol
        )  # may not be actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Nacelle.SURFACE_AREA], 339.58410134, tol
        )  # may not be actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Electrical.HYBRID_CABLE_LENGTH], 50.6032, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-12)


class SizeGroupTestCase4(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF,
                        val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=False, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM,
                        val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "size",
            SizeGroup(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, val=128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.021893, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALE_FACTOR, val=1.028233
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.FINENESS, 2, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.cabin_height"], 9.18333, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.cabin_len"], 435, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fuselage.nose_height"], 4.68333, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 476.7333, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 476.7333, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Wing.AREA], 1370.3125, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.SPAN], 117.81878299, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Wing.CENTER_CHORD], 17.48974356, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.61453233, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.ROOT_CHORD], 16.988, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.14151, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob["wing.strut.strut_y"], 0, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 5.2361, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.CHORD], 2.8647, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AREA], 298.484, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.SPAN], 37.654, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.ROOT_CHORD], 11.7265, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 8.5341, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.67937726, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AREA], 297.003, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.SPAN], 22.2709, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.ROOT_CHORD], 14.8094, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 13.3902, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.MOMENT_ARM], 49.88094115, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.35163168, tol
        )  # may not be actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.70326336, tol
        )  # may not be actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Nacelle.SURFACE_AREA], 339.58410134, tol
        )  # may not be actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
