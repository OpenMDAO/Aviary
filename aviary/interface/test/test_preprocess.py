"""
Test preprocessing as part of the level 2 interface.
"""

from copy import deepcopy
import unittest

from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.N3CC.phase_info import phase_info
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TestLevel2Preprocessing(unittest.TestCase):
    def test_crew_preprocessing(self):
        # Test that flight-crew preprocesses correctly.
        prob = AviaryProblem()
        local_phase_info = deepcopy(phase_info)

        prob.load_inputs('models/N3CC/N3CC_FLOPS.csv', local_phase_info)
        prob.check_and_preprocess_inputs()
        aviary_inputs = prob.aviary_inputs

        num_flight_crew = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        self.assertEqual(num_flight_crew, 3)

        num_flight_attendants = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS)
        self.assertEqual(num_flight_attendants, 4)

        num_galley_crew = aviary_inputs.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW)
        self.assertEqual(num_galley_crew, 1)


if __name__ == '__main__':
    unittest.main()
