"""
Test preprocessing as part of the level 2 interface.
"""

from copy import deepcopy
import unittest

from openmdao.utils.assert_utils import assert_warning
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.aircraft.advanced_single_aisle.phase_info import phase_info
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TestPrePreprocessing(unittest.TestCase):
    def test_crew_preprocessing(self):
        # Test that flight-crew preprocesses correctly.
        prob = AviaryProblem()
        local_phase_info = deepcopy(phase_info)

        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()
        aviary_inputs = prob.aviary_inputs

        num_flight_crew = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        self.assertEqual(num_flight_crew, 3)

        num_flight_attendants = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS)
        self.assertEqual(num_flight_attendants, 4)

        num_galley_crew = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW)
        self.assertEqual(num_galley_crew, 1)

    def test_missing_passengers_warnings(self):
        # Test that verifies Aviary's behavior when the number of passengers in each clase are
        # not consistent with given totals.
        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )
        prob.aviary_inputs.delete(Aircraft.CrewPayload.NUM_ECONOMY_CLASS)
        prob.aviary_inputs.delete(Aircraft.Engine.SCALED_SLS_THRUST)

        msg = (
            'Sum of all passenger classes (36) does not equal total number of '
            'passengers provided for current mission (154). Setting '
            'Aircraft.CrewPayload.NUM_PASSENGERS with the sum of passenger classes for flown '
            'mission (36).'
        )

        with assert_warning(UserWarning, msg):
            prob.check_and_preprocess_inputs()

        num_pass = prob.aviary_inputs.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
        self.assertEqual(num_pass, 36)

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )
        prob.aviary_inputs.delete(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        prob.aviary_inputs.delete(Aircraft.Engine.SCALED_SLS_THRUST)
        prob.aviary_inputs.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, 154)

        msg = (
            'Sum of all passenger classes (138) does not equal total number of '
            'passengers provided for aircraft design (154). Overriding '
            'Aircraft.CrewPayload.Design.NUM_PASSENGERS with the sum of '
            'passenger classes for design (138).'
        )

        with assert_warning(UserWarning, msg):
            prob.check_and_preprocess_inputs()

        num_pass = prob.aviary_inputs.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        self.assertEqual(num_pass, 138)

    def test_simple_cabin_layout(self):
        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
            local_phase_info,
        )
        prob.aviary_inputs.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, False)

        prob.check_and_preprocess_inputs()


if __name__ == '__main__':
    # unittest.main()
    test = TestPrePreprocessing()
    test.setUp()
    test.test_missing_passengers_warnings()
