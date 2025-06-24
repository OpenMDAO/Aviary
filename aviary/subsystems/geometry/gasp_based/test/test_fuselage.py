import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

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


class FuselageParametersTestCase1(unittest.TestCase):
    """this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            FuselageParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(self.prob['cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['nose_height'], 8.6, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageParametersTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            FuselageParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case2(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol)
        assert_near_equal(self.prob['cabin_height'], 9.183, tol)  # not actual GASP value
        assert_near_equal(self.prob['cabin_len'], 72.5, tol)  # not actual GASP value
        assert_near_equal(self.prob['nose_height'], 4.683, tol)  # not actual GASP value

        partial_data2 = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data2, atol=1e-8, rtol=1e-8)


class FuselageSizeTestCase1(unittest.TestCase):
    """this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('size', FuselageSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults('nose_height', 8.6, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults('cabin_len', 72.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults('cabin_height', 13.1, units='ft')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4639.68, tol)
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(self.prob[Aircraft.Fuselage.CABIN_AREA], 1068.96, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageSizeTestCase2(unittest.TestCase):
    """this is the GASP test case for V3.6 advanced tube and wing."""

    def setUp(self):
        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem('parameters', FuselageSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults('nose_height', 8.6, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults('cabin_len', 61.6, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults('cabin_height', 13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 3e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.LENGTH], 119.03, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 4209, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.TailBoom.LENGTH], 119.03, tol)  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Fuselage.CABIN_AREA], 931.41, tol)

        partial_data2 = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data2, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase1(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FuselageGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')

        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4639.57, tol)
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')  # not actual GASP value

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FuselageGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 4639.565, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')  # not actual GASP value

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FuselageGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Fuselage.LENGTH], 114.23, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 2947.51, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.TailBoom.LENGTH], 114.23, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase4(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')  # not actual GASP value

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FuselageGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless'
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )  # not actual GASP value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Fuselage.LENGTH], 114.23, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 2947.51, tol
        )  # not actual GASP value
        assert_near_equal(self.prob[Aircraft.TailBoom.LENGTH], 114.23, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class BWBFuselageParameters1TestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 18)
        self.aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 21, units='inch')
        self.aviary_options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        self.aviary_options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        self.aviary_options.set_val(Aircraft.Fuselage.SEAT_PITCH, 32, units='inch')
        self.aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 150)
        self.aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 18)
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        self.prob.model.add_subsystem(
            'bwb_fuselage_parameters1', BWBFuselageParameters1(), promotes=['*']
        )

        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 5.0, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 0.6, units='unitless')

        setup_model_options(self.prob, self.aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 38.0, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.HYDRAULIC_DIAMETER], 19.36509231, tol)
        assert_near_equal(self.prob['cabin_height'], 9.86859989, tol)
        assert_near_equal(self.prob['nose_height'], 4.86859989, tol)
        assert_near_equal(self.prob['nose_length'], 2.92115998, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


class BWBLayoutTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.aviary_options = AviaryValues()

        self.aviary_options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 18)
        self.aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 21, units='inch')
        self.aviary_options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        self.aviary_options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        self.aviary_options.set_val(Aircraft.Fuselage.SEAT_PITCH, 32, units='inch')
        self.aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 150)
        self.aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11)
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        self.prob.model.add_subsystem('bwb_cabin_layout', BWBCabinLayout(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        self.prob.model.set_input_defaults('nose_length', 2.92115998, units='ft')

        setup_model_options(self.prob, self.aviary_options)

        self.prob.setup()

    def test_case1(self):
        """Testing GASP data case."""
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['fuselage_station_aft'], 54.25449, tol)

        partial_data = self.prob.check_partials(
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
        """Testing 0 First Class case."""
        self.aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
            val=0,
            units='unitless',
        )
        setup_model_options(self.prob, self.aviary_options)
        self.prob.setup()

        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['fuselage_station_aft'], 51.25449, tol)

        partial_data = self.prob.check_partials(
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


class BWBFuselageParameters2TestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        self.prob.model.add_subsystem(
            'bwb_fuselage_parameters2', BWBFuselageParameters2(), promotes=['*']
        )

        self.prob.model.set_input_defaults(
            Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.75, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        self.prob.model.set_input_defaults('nose_length', 2.92115998, units='ft')
        self.prob.model.set_input_defaults('cabin_height', 9.86859989, units='ft')
        self.prob.model.set_input_defaults('fuselage_station_aft', 54.254501, units='ft')

        setup_model_options(self.prob, self.aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Fuselage.PLANFORM_AREA], 1943.76594, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.CABIN_AREA], 1283.52497, tol)
        assert_near_equal(self.prob['cabin_len'], 43.83334, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


class BWBFuselageSizeTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        self.prob.model.add_subsystem('bwb_fuselage_size', BWBFuselageSize(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft'
        )
        self.prob.model.set_input_defaults('cabin_height', 9.86859989, units='ft')
        self.prob.model.set_input_defaults('forebody_len', 40.7456322, units='ft')
        self.prob.model.set_input_defaults('fuselage_station_aft', 54.254501, units='ft')
        self.prob.model.set_input_defaults('nose_area', 3.97908521, units='ft**2')
        self.prob.model.set_input_defaults('aftbody_len', 17.27005, units='ft')
        self.prob.model.set_input_defaults('nose_length', 2.921159934, units='ft')
        self.prob.model.set_input_defaults('cabin_len', 43.8333397, units='ft')

        setup_model_options(self.prob, self.aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4573.42578, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.LENGTH], 71.5245514, tol)
        assert_near_equal(self.prob[Aircraft.TailBoom.LENGTH], 71.5245514, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-5, rtol=1e-5)


class BWBFuselageGroupTestCase(unittest.TestCase):
    """this is the GASP test case."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 18)
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 21, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 32, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 150)
        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11)

        options.set_val(Settings.VERBOSITY, 1, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            BWBFuselageGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 5.0, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.25970, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.75, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """Testing GASP data case."""
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 38.0, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.HYDRAULIC_DIAMETER], 19.36509, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.CABIN_AREA], 1283.52497, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.PLANFORM_AREA], 1943.76594, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.LENGTH], 71.5245514, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4573.42510, tol)
        assert_near_equal(self.prob[Aircraft.TailBoom.LENGTH], 71.5245514, tol)

        partial_data = self.prob.check_partials(
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
    # unittest.main()
    test = FuselageSizeTestCase2()
    test.setUp()
    test.test_case1()
