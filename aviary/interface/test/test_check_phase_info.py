import unittest
from aviary.interface.utils.check_phase_info import check_phase_info
from aviary.interface.default_phase_info.flops import phase_info as phase_info_flops
from aviary.interface.default_phase_info.gasp import phase_info as phase_info_gasp
import copy


class TestCheckInputs(unittest.TestCase):
    def test_correct_input_flops(self):
        # This should pass without any issue as it's the same valid dict as before
        self.assertTrue(check_phase_info(phase_info_flops, mission_method="FLOPS"))

    def test_incorrect_tuple(self):
        # Let's replace a tuple with a float in the 'climb' phase
        incorrect_tuple_phase_info = copy.deepcopy(phase_info_flops)
        incorrect_tuple_phase_info['climb']['user_options']['initial_altitude'] = 10.668e3
        with self.assertRaises(ValueError):
            check_phase_info(incorrect_tuple_phase_info, mission_method="FLOPS")

    def test_incorrect_unit(self):
        # Let's replace a valid unit with an invalid one in the 'climb' phase
        incorrect_unit_phase_info = copy.deepcopy(phase_info_flops)
        incorrect_unit_phase_info['climb']['user_options']['initial_altitude'] = (
            10.668e3, 'invalid_unit')
        with self.assertRaisesRegex(ValueError,
                                    "Key initial_altitude in phase climb has an invalid unit invalid_unit."):
            check_phase_info(incorrect_unit_phase_info, mission_method="FLOPS")

    def test_correct_input_gasp(self):
        # This should pass without any issue as it's the same valid dict as before
        self.assertTrue(check_phase_info(phase_info_gasp, mission_method="GASP"))


if __name__ == '__main__':
    # unittest.main()
    test = TestCheckInputs()
    test.test_incorrect_unit()
