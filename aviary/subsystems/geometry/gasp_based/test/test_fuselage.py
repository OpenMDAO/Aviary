import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.geometry.gasp_based.fuselage import (
    BWBCabinLayout,
    BWBFuselageGroup,
    BWBFuselageParameters1,
    BWBFuselageParameters2,
    BWBFuselageSize,
    FuselageGroup,
    FuselageParameters,
    FuselageSize,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Settings


@use_tempdirs
class FuselageParametersTestCase1(unittest.TestCase):
    """this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180)
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 6)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('parameters', FuselageParameters(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(prob['cabin_height'], 13.1, tol)
        assert_near_equal(prob['cabin_len'], 72.1, tol)
        assert_near_equal(prob['nose_height'], 8.6, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageParametersTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('parameters', FuselageParameters(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case2(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol)
        assert_near_equal(prob['cabin_height'], 9.183, tol)  # not actual GASP value
        assert_near_equal(prob['cabin_len'], 72.5, tol)  # not actual GASP value
        assert_near_equal(prob['nose_height'], 4.683, tol)  # not actual GASP value

        partial_data2 = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data2, atol=1e-8, rtol=1e-8)


class FuselageSizeTestCase1(unittest.TestCase):
    """this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('size', FuselageSize(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        prob.model.set_input_defaults('nose_height', 8.6, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')
        prob.model.set_input_defaults('cabin_len', 72.1, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        prob.model.set_input_defaults('cabin_height', 13.1, units='ft')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(
            prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4639.68, tol)
        assert_near_equal(
            prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 1068.96, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageSizeTestCase2(unittest.TestCase):
    """this is the GASP test case for V3.6 advanced tube and wing."""

    def setUp(self):
        options = get_option_defaults()

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('parameters', FuselageSize(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        prob.model.set_input_defaults('nose_height', 8.6, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')
        prob.model.set_input_defaults('cabin_len', 61.6, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        prob.model.set_input_defaults('cabin_height', 13.1, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA_SCALER, 1, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 3e-4
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 119.03, tol)  # not actual GASP value
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4209, tol)  # not actual GASP value
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 119.03, tol)  # not actual GASP value
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 931.41, tol)

        partial_data2 = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data2, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase1(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 6)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', FuselageGroup(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(
            prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4639.57, tol)
        assert_near_equal(
            prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase2(unittest.TestCase):
    def setUp(self):
        """default values are not actual GASP values"""
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 6)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', FuselageGroup(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """outputs are not actual GASP value"""
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 129.5, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4639.565, tol)
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 129.5, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase3(unittest.TestCase):
    def setUp(self):
        """default values are not actual GASP value"""
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', FuselageGroup(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """outputs are not actual GASP value"""
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 114.23, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 2947.51, tol)
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 114.23, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase4(unittest.TestCase):
    def setUp(self):
        """default values are not actual GASP value"""
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', FuselageGroup(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.2, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """outputs are not actual GASP value"""
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 114.23, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 2947.51, tol)
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 114.23, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class BWBFuselageParameters1TestCase(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 18)
        aviary_options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        aviary_options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 32, units='inch')
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 150)
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 18)
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        prob.model.add_subsystem(
            'bwb_fuselage_parameters1', BWBFuselageParameters1(), promotes=['*']
        )

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 21, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 5.0, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 0.6, units='unitless')

        setup_model_options(prob, aviary_options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 38.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.HYDRAULIC_DIAMETER], 19.36509231, tol)
        assert_near_equal(prob['cabin_height'], 9.86859989, tol)
        assert_near_equal(prob['nose_height'], 4.86859989, tol)
        assert_near_equal(prob['nose_length'], 2.92115998, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


@use_tempdirs
class BWBLayoutTestCase(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        aviary_options = self.aviary_options = AviaryValues()

        aviary_options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        aviary_options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 32, units='inch')
        aviary_options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 36, units='inch')
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 150)
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11)
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        prob.model.add_subsystem('bwb_cabin_layout', BWBCabinLayout(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_FIRST, 28.0, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 21.0, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        prob.model.set_input_defaults(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults('nose_length', 2.92115998, units='ft')

        setup_model_options(prob, aviary_options)

        prob.setup()

    def test_case1(self):
        """Testing GASP data case: first class + economy class"""
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob['fuselage_station_aft'], 54.25449, tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)

    def test_case2(self):
        """Testing case: economy class only"""
        self.aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
            val=0,
            units='unitless',
        )
        prob = self.prob
        setup_model_options(prob, self.aviary_options)
        prob.setup()

        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob['fuselage_station_aft'], 51.25449, tol)

    def test_case3(self):
        """Testing case: first class + business class + economy class"""
        self.aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, val=20, units='unitless'
        )
        prob = self.prob
        setup_model_options(prob, self.aviary_options)
        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_BUSINESS, val=25.0, units='inch')
        prob.setup()

        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob['fuselage_station_aft'], 59.83782665, tol)


class BWBFuselageParameters2TestCase(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        prob.model.add_subsystem(
            'bwb_fuselage_parameters2', BWBFuselageParameters2(), promotes=['*']
        )

        prob.model.set_input_defaults(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.75, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults('nose_length', 2.92115998, units='ft')
        prob.model.set_input_defaults('cabin_height', 9.86859989, units='ft')
        prob.model.set_input_defaults('fuselage_station_aft', 54.254501, units='ft')

        setup_model_options(prob, self.aviary_options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 1943.76594, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 1283.52497, tol)
        assert_near_equal(prob['cabin_len'], 43.83334, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


class BWBFuselageSizeTestCase(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        prob.model.add_subsystem('bwb_fuselage_size', BWBFuselageSize(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft')
        prob.model.set_input_defaults('cabin_height', 9.86859989, units='ft')
        prob.model.set_input_defaults('forebody_len', 40.7456322, units='ft')
        prob.model.set_input_defaults('fuselage_station_aft', 54.254501, units='ft')
        prob.model.set_input_defaults('nose_area', 3.97908521, units='ft**2')
        prob.model.set_input_defaults('aftbody_len', 17.27005, units='ft')
        prob.model.set_input_defaults('nose_length', 2.921159934, units='ft')
        prob.model.set_input_defaults('cabin_len', 43.8333397, units='ft')

        setup_model_options(prob, self.aviary_options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        prob = self.prob
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4573.42578, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 71.5245514, tol)
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 71.5245514, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


@use_tempdirs
class BWBFuselageGroupTestCase(unittest.TestCase):
    """this is the GASP test case."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_ECONOMY, 18)
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_ECONOMY, 32, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 36, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 150)
        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11)

        options.set_val(Settings.VERBOSITY, 1, units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', BWBFuselageGroup(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_FIRST, 28, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 21, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 5.0, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 0.6, units='unitless')
        prob.model.set_input_defaults(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.75, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        prob = self.prob
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 38.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.HYDRAULIC_DIAMETER], 19.36509, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 1283.52497, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 1943.76594, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 71.5245514, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4573.42510, tol)
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 71.5245514, tol)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
            minimum_step=1e-12,
            abs_err_tol=5.0e-4,
            rel_err_tol=5.0e-5,
        )
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
