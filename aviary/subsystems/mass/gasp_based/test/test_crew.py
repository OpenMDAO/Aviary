import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.crew import CabinCrewMass, FlightCrewMass
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class CrewMassTestCase(unittest.TestCase):
    """Tests for CabinCrewMass and FlightCrewMass."""

    def _make_prob(self, engine_type, num_passengers):
        options = AviaryValues()
        options.set_val(Aircraft.Engine.TYPE, val=[engine_type], units='unitless')
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=num_passengers, units='unitless'
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            'non_flight_crew',
            CabinCrewMass(),
            promotes=['*'],
        )

        prob.model.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        prob = self._make_prob(
            engine_type=GASPEngineType.TURBOJET,  # arbitrarily set
            num_passengers=180,  # large_single_aisle_1_GASP.csv
        )
        prob.run_model()

        tol = 1e-7
        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 800.0, tol)
            assert_near_equal(prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 492.0, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_case3(self):
        """BWB Parameters."""
        prob = self._make_prob(
            engine_type=GASPEngineType.RECIP_CARB,  # arbitrarily set
            num_passengers=150,  # large_single_aisle_1_GASP.csv
        )
        prob.run_model()

        tol = 1e-7
        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 600.0, tol)
            assert_near_equal(prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 492.0, tol)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
