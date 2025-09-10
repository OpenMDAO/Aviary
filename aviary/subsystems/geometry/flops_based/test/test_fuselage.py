import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.flops_based.fuselage import (
    BWBDetailedCabinLayout,
    BWBSimpleCabinLayout,
    DetailedCabinLayout,
    SimpleCabinLayout,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Settings


class SimpleCabinLayoutTest(unittest.TestCase):
    """Test simple cabin layout computation."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        prob.model.add_subsystem(
            'layout', SimpleCabinLayout(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=125.0, units='ft')
        prob.run_model()

        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 86.99047784, tolerance=1e-9)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


class DetailedCabinLayoutTest(unittest.TestCase):
    """Test simple cabin layout computation."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        options = self.aviary_options = AviaryValues()
        options.set_val(Settings.VERBOSITY, 1, units='unitless')

        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 158, units='unitless')
        # options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, units='unitless')
        # options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, units='unitless')
        # options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, units='unitless')
        # options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, [2], units='unitless')

        prob.model.add_subsystem(
            'layout', DetailedCabinLayout(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=125.0, units='ft')
        prob.run_model()

        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 107.53947222, tolerance=1e-9)
        fuselage_length = prob.get_val(Aircraft.Fuselage.LENGTH)
        assert_near_equal(fuselage_length, 147.53947222, tolerance=1e-9)

        fuselage_width = prob.get_val(Aircraft.Fuselage.MAX_WIDTH)
        assert_near_equal(fuselage_width, 14.84, tolerance=1e-9)

        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 15.74, tolerance=1e-9)


class BWBSimpleBWBCabinLayoutTest(unittest.TestCase):
    """Test simple cabin layout computation."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        prob.model.add_subsystem(
            'layout', BWBSimpleCabinLayout(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=137.5, units='ft')
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=64.58, units='ft')
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, val=45.0, units='deg')
        prob.set_val(Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, val=0.11, units='unitless')
        prob.run_model()

        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 96.25, tolerance=1e-9)
        root_chord = prob.get_val(Aircraft.Wing.ROOT_CHORD)
        assert_near_equal(root_chord, 63.96019518, tolerance=1e-9)
        area_cabin = prob.get_val(Aircraft.Fuselage.CABIN_AREA)
        assert_near_equal(area_cabin, 5173.1872025, tolerance=1e-9)
        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 15.125, tolerance=1e-9)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


class BWBDetailedCabinLayoutTest(unittest.TestCase):
    """Test simple cabin layout computation."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        options = self.aviary_options = AviaryValues()
        options.set_val(Settings.VERBOSITY, 1, units='unitless')

        options.set_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS, 100.0, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 28.0, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 340.0, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_BUSINESS, 4, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, 4, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_BUSINESS, 39.0, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 61.0, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 32.0, units='inch')

        prob.model.add_subsystem(
            'layout', BWBDetailedCabinLayout(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, val=45.0, units='deg')
        prob.set_val(Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, val=0.11, units='unitless')
        prob.set_val('Rear_spar_percent_chord', val=0.7, units='unitless')
        prob.run_model()

        fuselage_length = prob.get_val(Aircraft.Fuselage.LENGTH)
        assert_near_equal(fuselage_length, 90.2906901, tolerance=1e-9)

        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 63.20348307, tolerance=1e-9)

        fuselage_width = prob.get_val(Aircraft.Fuselage.MAX_WIDTH)
        assert_near_equal(fuselage_width, 144.0, tolerance=1e-9)

        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 9.93197591, tolerance=1e-9)

        cabin_area = prob.get_val(Aircraft.Fuselage.CABIN_AREA)
        assert_near_equal(cabin_area, 3917.33289811, tolerance=1e-9)


if __name__ == '__main__':
    # unittest.main()
    test = BWBDetailedCabinLayoutTest()
    test.setUp()
    test.test_case1()
