import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.crew import FlightCrewMass, NonFlightCrewMass

from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class CrewTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.TURBOJET], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'non_flight_crew',
            NonFlightCrewMass(),
            promotes=['*'],
        )

        self.prob.model.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 800.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 492.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class CrewTestCase2(unittest.TestCase):
    """Gravity Modification Test"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.TURBOJET], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'non_flight_crew',
            NonFlightCrewMass(),
            promotes=['*'],
        )

        self.prob.model.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.crew as crew

        crew.GRAV_ENGLISH_LBM = 1.0

        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )  # large_single_aisle_1_GASP.csv

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.crew as crew

        crew.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 800.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 492.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class CrewTestCase3(unittest.TestCase):
    """BWB Parameters"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.Engine.TYPE, val=[GASPEngineType.RECIP_CARB], units='unitless'
        )  # arbitrarily set
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless'
        )  # large_single_aisle_1_GASP.csv

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'non_flight_crew',
            NonFlightCrewMass(),
            promotes=['*'],
        )

        self.prob.model.add_subsystem(
            'flight_crew',
            FlightCrewMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 600.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 492.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
