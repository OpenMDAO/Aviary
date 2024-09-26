import unittest

from copy import deepcopy

from aviary.utils.test_utils.variable_test import (
    assert_metadata_alphabetization, assert_no_duplicates,
    assert_structure_alphabetization, get_names_from_hierarchy)
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings


class DuplicateHierarchy:
    """A sample data set with a duplicate name in two different classes."""

    stuff = 'nothing'

    class Design:
        CRUISE_MACH = 'mission:design:cruise_mach'
        RANGE = 'mission:design:range'

    class OperatingLimits:
        MAX_MACH = 'mission:design:cruise_mach'  # this is a duplicate


class MetaDataTest(unittest.TestCase):
    """
    Tests for variable_meta_data.py: check for duplicate legacy code names, alphabetization, and any missing names from variable hierarchy.
    """

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

    def test_missing_names(self):
        # Test that all variables inside the metadata exist in the hierarchy, and vice-versa
        var_names = \
            get_names_from_hierarchy(Aircraft)\
            + get_names_from_hierarchy(Mission)\
            + get_names_from_hierarchy(Dynamic)\
            + get_names_from_hierarchy(Settings)

        metadata_dict = deepcopy(_MetaData)
        for var in var_names:
            try:
                metadata_dict.pop(var)
            except (TypeError, KeyError):
                raise Exception(f"Variable {var} is present in variables.py but is not "
                                'defined in metadata')
        if metadata_dict:
            # This will only happen if a variable in the metadata wasn't using the hierarchy
            raise Exception(f'Variables {[*metadata_dict.keys()]} are present in metadata, but are'
                            ' not defined in variables.py')


class VariableStructureTest(unittest.TestCase):
    """
    Tests for variables.py: check for duplicates and alphabetization
    """

    def test_duplicate_names_Aviary(self):

        aviary_names = get_names_from_hierarchy(Aircraft)\
            + get_names_from_hierarchy(Mission)\
            + get_names_from_hierarchy(Dynamic)\
            + get_names_from_hierarchy(Settings)

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
