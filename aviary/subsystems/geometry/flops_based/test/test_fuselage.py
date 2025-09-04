import unittest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.flops_based.fuselage import DetailedCabinLayout, SimpleCabinLayout
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
        prob.set_val(Aircraft.Fuselage.LENGTH, val=125.0)
        prob.run_model()

        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 86.99047784, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


class DetailedCabinLayoutTest(unittest.TestCase):
    """Test simple cabin layout computation."""

    def setUp(self):
        self.prob = om.Problem()

    def test_case1(self):
        prob = self.prob
        self.aviary_options = AviaryValues()
        self.aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')

        self.aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11, units='unitless'
        )
        self.aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, 158, units='unitless'
        )
        # self.aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_FIRST, units='unitless')
        # self.aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, units='unitless')
        # self.aviary_options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, units='unitless')
        # self.aviary_options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, units='unitless')
        self.aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [2], units='unitless')

        prob.model.add_subsystem(
            'layout', DetailedCabinLayout(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        setup_model_options(self.prob, self.aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.LENGTH, val=125.0)
        prob.run_model()

        pax_compart_length = prob.get_val(Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH)
        assert_near_equal(pax_compart_length, 107.53947222, tolerance=1e-10)
        fuselage_length = prob.get_val(Aircraft.Fuselage.LENGTH)
        assert_near_equal(fuselage_length, 147.53947222, tolerance=1e-10)

        fuselage_width = prob.get_val(Aircraft.Fuselage.MAX_WIDTH)
        assert_near_equal(fuselage_width, 14.84, tolerance=1e-10)

        fuselage_height = prob.get_val(Aircraft.Fuselage.MAX_HEIGHT)
        assert_near_equal(fuselage_height, 15.74, tolerance=1e-10)


if __name__ == '__main__':
    unittest.main()
