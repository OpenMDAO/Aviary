import unittest

from copy import deepcopy

from aviary.utils.test_utils.variable_test import (
    DuplicateHierarchy, assert_metadata_alphabetization, assert_no_duplicates,
    assert_structure_alphabetization, get_names_from_hierarchy)
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Dynamic, Settings


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
        metadata_var_names = [catey for catey in _MetaData if ':' in catey]

        assert_metadata_alphabetization(metadata_var_names)

    def test_missing_names(self):
        """
        Test that all variables inside the metadata exist in the hierarchy, and vice-versa
        """
        # NOTE: This is messy due to the fact we are dealing with attributes inside nested classes
        var_names = \
            [(var_name, var) for cat_name, cat in Aircraft.__dict__.items() if not cat_name.startswith('__')
                for var_name, var in cat.__dict__.items() if not var_name.startswith('__')]\
            + [(var_name, var) for cat_name, cat in Mission.__dict__.items() if not cat_name.startswith('__')
                for var_name, var in cat.__dict__.items() if not var_name.startswith('__')]\
            + [(var_name, var) for cat_name, cat in Dynamic.__dict__.items() if not cat_name.startswith('__')
                for var_name, var in cat.__dict__.items() if not var_name.startswith('__')]\
            + [(var_name, var) for var_name, var in Settings.__dict__.items()
                if not var_name.startswith('__')]

        metadata_dict = deepcopy(_MetaData)
        for var in var_names:
            try:
                metadata_dict.pop(var[1])
            except (TypeError, KeyError):
                raise Exception(f"Variable {var[0]} ('{var[1]}') is present in variables.py but is not "
                                'defined in metadata')
        if metadata_dict:
            # This will only happen if a variable in the metadata wasn't using the hierarchy
            raise Exception(f'Variables {[*metadata_dict.keys()]} are present in metadata, but are'
                            ' not defined in variables.py')


class VariableStructureTest(unittest.TestCase):
    def test_duplicate_names_Aviary(self):

        aviary_names = get_names_from_hierarchy(Aircraft)\
            + get_names_from_hierarchy(Mission)\
            + get_names_from_hierarchy(Dynamic)\
            + get_names_from_hierarchy(Settings)

        assert_no_duplicates(aviary_names)

    def test_alphabetization(self):

        assert_structure_alphabetization("variable_info/variables.py")


class TestTheTests(unittest.TestCase):
    def test_duplication_checcat(self):

        with self.assertRaises(ValueError) as cm:
            duplicated_names = get_names_from_hierarchy(DuplicateHierarchy)
            assert_no_duplicates(duplicated_names)
        self.assertEqual(str(
            cm.exception), "The variables ['mission:design:cruise_mach'] are duplicates in the provided list.")


if __name__ == "__main__":
    unittest.main()
