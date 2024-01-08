import unittest

from aviary.utils.test_utils.variable_test import (
    DuplicateHierarchy, assert_metadata_alphabetization, assert_no_duplicates,
    assert_structure_alphabetization, get_names_from_hierarchy)
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission


class MetaDataTest(unittest.TestCase):
    def test_duplicate_names_FLOPS(self):

        flops_names = [var["historical_name"]['FLOPS'] for var in _MetaData.values()]
        assert_no_duplicates(flops_names)

    def test_duplicate_names_LEAPS1(self):

        leaps1_names = [var["historical_name"]['LEAPS1'] for var in _MetaData.values()]
        assert_no_duplicates(leaps1_names)

    def test_alphabetization(self):

        # TODO currently excluding Dynamic variables that do not have proper full
        #      names mirroring the hierarchy
        metadata_var_names = [key for key in _MetaData if ':' in key]

        assert_metadata_alphabetization(metadata_var_names)


class VariableStructureTest(unittest.TestCase):
    def test_duplicate_names_Aviary(self):

        aviary_names = get_names_from_hierarchy(
            Aircraft) + get_names_from_hierarchy(Mission)

        assert_no_duplicates(aviary_names)

    def test_alphabetization(self):

        assert_structure_alphabetization("variable_info/variables.py")


class TestTheTests(unittest.TestCase):
    def test_duplication_check(self):

        with self.assertRaises(ValueError) as cm:
            duplicated_names = get_names_from_hierarchy(DuplicateHierarchy)
            assert_no_duplicates(duplicated_names)
        self.assertEqual(str(
            cm.exception), "The variables ['mission:design:cruise_mach'] are duplicates in the provided list.")


if __name__ == "__main__":
    unittest.main()
