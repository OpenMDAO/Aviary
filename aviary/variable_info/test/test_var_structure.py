import unittest
from copy import deepcopy

from aviary.utils.test_utils.variable_test import (
    assert_metadata_alphabetization,
    assert_no_duplicates,
    assert_structure_alphabetization,
    get_names_from_hierarchy,
)
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


class DuplicateHierarchy:
    """A sample data set with a duplicate name in two different classes."""

    stuff = 'nothing'

    class Design:
        CRUISE_MACH = 'mission:design:cruise_mach'
        RANGE = 'mission:design:range'

    class OperatingLimits:
        MAX_MACH = 'mission:design:cruise_mach'  # this is a duplicate


class MetaDataTest(unittest.TestCase):
    """Tests for variable_meta_data.py: check for duplicate legacy code names, alphabetization, and any missing names from variable hierarchy."""

    def test_duplicate_names_FLOPS(self):
        flops_names = [var['historical_name']['FLOPS'] for var in _MetaData.values()]
        assert_no_duplicates(flops_names)

    def test_duplicate_names_LEAPS1(self):
        leaps1_names = [var['historical_name']['LEAPS1'] for var in _MetaData.values()]
        assert_no_duplicates(leaps1_names)

    def test_alphabetization(self):
        # TODO currently excluding Dynamic variables that do not have proper full
        #      names mirroring the hierarchy
        metadata_var_names = [key for key in _MetaData if ':' in key]

        assert_metadata_alphabetization(metadata_var_names)

    def test_missing_names(self):
        # Test that all variables inside the metadata exist in the hierarchy, and vice-versa
        var_names = (
            get_names_from_hierarchy(Aircraft)
            + get_names_from_hierarchy(Mission)
            + get_names_from_hierarchy(Dynamic)
            + get_names_from_hierarchy(Settings)
        )

        metadata_dict = deepcopy(_MetaData)
        for var in var_names:
            try:
                metadata_dict.pop(var)
            except (TypeError, KeyError):
                raise Exception(
                    f'Variable {var} is present in variables.py but is not defined in metadata'
                )
        if metadata_dict:
            # This will only happen if a variable in the metadata wasn't using the hierarchy
            raise Exception(
                f'Variables {[*metadata_dict.keys()]} are present in metadata, but are'
                ' not defined in variables.py'
            )


class VariableStructureTest(unittest.TestCase):
    """Tests for variables.py: check for duplicates and alphabetization."""

    def test_duplicate_names_Aviary(self):
        aviary_names = (
            get_names_from_hierarchy(Aircraft)
            + get_names_from_hierarchy(Mission)
            + get_names_from_hierarchy(Dynamic)
            + get_names_from_hierarchy(Settings)
        )

        assert_no_duplicates(aviary_names)

    def test_alphabetization(self):
        assert_structure_alphabetization('variable_info/variables.py')

    # This test was written using AI
    def test_variable_names_match_strings(self):
        """
        Test that the last section of variable names match their associated string values.
        For example, Aircraft.APU.MASS should have 'mass' as the last section of 'aircraft:apu:mass'.
        Only the last section (after the last colon) is checked, not the full path.
        """

        def check_hierarchy(hierarchy, path_prefix='', hierarchy_name='', display_name=''):
            """
            Recursively check that variable names match their string values.

            Parameters
            ----------
            hierarchy : class
                The class hierarchy to check
            path_prefix : str
                The prefix path for building expected strings (e.g., 'aircraft:apu')
            hierarchy_name : str
                The name of the hierarchy root (e.g., 'aircraft', 'mission', 'settings').
                If empty, no prefix is used (for Dynamic variables).
            display_name : str
                The name to use in error messages (e.g., 'Dynamic', 'Aircraft')
            """
            mismatches = []

            # Get all attributes that don't start with __
            attrs = [attr for attr in dir(hierarchy) if not attr.startswith('__')]

            for attr_name in attrs:
                attr_value = getattr(hierarchy, attr_name)

                # Skip if it's a method or other non-string, non-class attribute
                if callable(attr_value) and not isinstance(attr_value, type):
                    continue

                # If it's a string, check if it matches the expected pattern
                if isinstance(attr_value, str):
                    # Only check the last section of the variable name
                    # Extract the last part after the last colon (or the whole string if no colon)
                    actual_last_section = attr_value.split(':')[-1]

                    # Expected last section is the attribute name in lowercase
                    expected_last_section = attr_name.lower()

                    # Compare only the last sections
                    if actual_last_section != expected_last_section:
                        # Build the full path for error message
                        if path_prefix:
                            full_path = f'{path_prefix}.{attr_name}'
                        else:
                            full_path = f'{display_name}.{attr_name}'
                        mismatches.append(
                            f'{full_path} = {attr_value!r}, expected last section '
                            f'{expected_last_section!r}'
                        )

                # If it's a class (nested class), recurse
                elif isinstance(attr_value, type):
                    # Build new path prefix
                    class_name_lower = attr_name.lower()
                    if hierarchy_name:
                        # Use full path pattern
                        if path_prefix:
                            new_prefix = f'{path_prefix}:{class_name_lower}'
                        else:
                            new_prefix = f'{hierarchy_name}:{class_name_lower}'
                    else:
                        # No prefix pattern (for Dynamic variables)
                        new_prefix = ''

                    # Build display name for error messages
                    if path_prefix:
                        new_display_name = f'{path_prefix}.{attr_name}'
                    else:
                        new_display_name = f'{display_name}.{attr_name}'

                    # Recursively check nested class
                    nested_mismatches = check_hierarchy(
                        attr_value, new_prefix, hierarchy_name, new_display_name
                    )
                    mismatches.extend(nested_mismatches)

            return mismatches

        all_mismatches = []

        aircraft_mismatches = check_hierarchy(Aircraft, '', 'aircraft', 'Aircraft')
        all_mismatches.extend(aircraft_mismatches)

        mission_mismatches = check_hierarchy(Mission, '', 'mission', 'Mission')
        all_mismatches.extend(mission_mismatches)

        # Note: Dynamic variables don't have the full path prefix, they're just the variable name
        # Pass empty hierarchy_name to indicate no prefix should be used

        # TODO: Should dynamic variables be updated to match hierarchy names?
        # dynamic_mismatches = check_hierarchy(Dynamic, '', '', 'Dynamic')
        # all_mismatches.extend(dynamic_mismatches)

        settings_mismatches = check_hierarchy(Settings, '', 'settings', 'Settings')
        all_mismatches.extend(settings_mismatches)

        if all_mismatches:
            error_msg = 'Variable name mismatches found:\n' + '\n'.join(all_mismatches)
            self.fail(error_msg)


class TestTheTests(unittest.TestCase):
    def test_duplication_check(self):
        with self.assertRaises(ValueError) as cm:
            duplicated_names = get_names_from_hierarchy(DuplicateHierarchy)
            assert_no_duplicates(duplicated_names)
        self.assertEqual(
            str(cm.exception),
            "The variables ['mission:design:cruise_mach'] are duplicates in the provided list.",
        )


if __name__ == '__main__':
    # unittest.main()
    test = VariableStructureTest()
    test.test_variable_names_match_strings()
