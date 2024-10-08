import unittest

import aviary.api as av
from aviary.utils.merge_variable_metadata import merge_meta_data

dict1 = av.CoreMetaData.copy()
# this is the baseline Aviary-core metadata
dict2 = av.CoreMetaData.copy()
# This is the baseline Aviary-core metadata with a few variables added onto it
dict3 = av.CoreMetaData.copy()
# This is the baseline Aviary-core metadata with a few variables added onto it, including some duplicates from dict2 that have the same metadata
dict4 = av.CoreMetaData.copy()
# This is the baseline Aviary-core metadata with a few variables added onto it, include a duplicate from dict2 that has different metadata
merged_dicts_23 = av.CoreMetaData.copy()
# This is the expected result of merging dict2 with dict3

############################################ Make dict2 ############################################
av.add_meta_data(
    'aircraft:CG',
    units='ft',
    desc='Center of gravity',
    default_value=0,
    meta_data=dict2,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:wing:character',
    units=None,
    desc='Character of wing',
    default_value=0,
    meta_data=dict2,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:landing_gear:shape',
    units='ft',
    desc='Shape of landing gear',
    default_value=0,
    meta_data=dict2,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:engine:color',
    units=None,
    desc='Color of engine',
    default_value='red',
    meta_data=dict2,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:fuselage:eccentricity',
    units=None,
    desc='Eccentricity of cross section of fuselage',
    default_value=.5,
    meta_data=dict2,
    historical_name=None,
)


############################################ Make dict3 ############################################
av.add_meta_data(
    'aircraft:CG',  # This is a duplicate from dict2 with the same metadata
    units='ft',
    desc='Center of gravity',
    default_value=0,
    meta_data=dict3,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:wing:temperature',
    units='degK',
    desc='Temperature of the wing on the ground',
    default_value=273,
    meta_data=dict3,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:strut:superiority',
    units=None,
    desc='Amount by which the strut is better than having no strut',
    default_value=2,
    meta_data=dict3,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:engine:color',  # This is a duplicate from dict2 with the same metadata
    units=None,
    desc='Color of engine',
    default_value='red',
    meta_data=dict3,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:overall_color',
    units=None,
    desc='The predominant color of the vehicle',
    default_value='purple',
    meta_data=dict3,
    historical_name=None,
)


############################################ Make dict4 ############################################
av.add_meta_data(
    'aircraft:CG',  # This is a duplicate from dict2 with the same metadata
    units='ft',
    desc='Center of gravity',
    default_value=0,
    meta_data=dict4,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:wing:temperature',
    units='degK',
    desc='Temperature of the wing on the ground',
    default_value=273,
    meta_data=dict4,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:strut:superiority',
    units=None,
    desc='Amount by which the strut is better than having no strut',
    default_value=2,
    meta_data=dict4,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:engine:color',  # This is a duplicate from dict2 with different metadata
    units=None,
    desc='Color of enigne',  # note there is a subtle typo in the description here
    default_value='red',
    meta_data=dict4,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:overall_color',
    units=None,
    desc='The predominant color of the vehicle',
    default_value='purple',
    meta_data=dict4,
    historical_name=None,
)


############################################ Make merge of dicts 2&3 ############################################
av.add_meta_data(
    'aircraft:CG',
    units='ft',
    desc='Center of gravity',
    default_value=0,
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:wing:character',
    units=None,
    desc='Character of wing',
    default_value=0,
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:landing_gear:shape',
    units='ft',
    desc='Shape of landing gear',
    default_value=0,
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:engine:color',
    units=None,
    desc='Color of engine',
    default_value='red',
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:fuselage:eccentricity',
    units=None,
    desc='Eccentricity of cross section of fuselage',
    default_value=.5,
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:wing:temperature',
    units='degK',
    desc='Temperature of the wing on the ground',
    default_value=273,
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:strut:superiority',
    units=None,
    desc='Amount by which the strut is better than having no strut',
    default_value=2,
    meta_data=merged_dicts_23,
    historical_name=None,
)

av.add_meta_data(
    'aircraft:overall_color',
    units=None,
    desc='The predominant color of the vehicle',
    default_value='purple',
    meta_data=merged_dicts_23,
    historical_name=None,
)
############################################ Make merge of dicts 1&2&3 ############################################
# This is the expected result of merging dict1 with dict2 and dict3
merged_dicts_123 = merged_dicts_23.copy()
############################################ Make merge of dicts 1&2 ############################################
merged_dicts_12 = dict2.copy()  # This is the expected result of merging dict1 with dict2
############################################ Make merge of dicts 1&3 ############################################
merged_dicts_13 = dict3.copy()  # This is the expected result of merging dict1 with dict3
############################################ Make expected error message for attempted merge of dicts 2&4 ############################################
# This is the expected error message resulting from mergin dict2 with dict4
merged_dicts_24_msg = 'You have attempted to merge metadata dictionaries that contain the same variable with different metadata. The offending variable present in multiple dictionaries is "aircraft:engine:color".'

############################################ Make dict5 ############################################
dict5 = av.CoreMetaData.copy()

av.add_meta_data(
    'aircraft:fuselage:eccentricity',
    units=None,
    desc='Eccentricity of cross section of fuselage',
    default_value=0.500000001,  # a small but acceptable difference
    meta_data=dict5,
    historical_name=None,
)

############################################ Make dict6 ############################################
dict6 = av.CoreMetaData.copy()

av.add_meta_data(
    'aircraft:fuselage:eccentricity',
    units=None,
    desc='Eccentricity of cross section of fuselage',
    default_value=0.50001,  # a larger difference
    meta_data=dict6,
    historical_name=None,
)

# expected error message for merging dict2 and dict6
merged_dicts_26_msg = 'You have attempted to merge metadata dictionaries that contain the same variable with different metadata. The offending variable present in multiple dictionaries is "aircraft:fuselage:eccentricity".'


class MergeMetaDataTest(unittest.TestCase):
    """
    Test functionality of merge_meta_data function.
    """

    def test_match_merge(self):
        merge23 = merge_meta_data([dict2, dict3])
        merge123 = merge_meta_data([dict1, dict2, dict3])
        merge12 = merge_meta_data([dict1, dict2])
        merge13 = merge_meta_data([dict1, dict3])
        # test the small, acceptable difference
        merge25 = merge_meta_data([dict2, dict5])

        self.assertEqual(merge23, merged_dicts_23)
        self.assertEqual(merge123, merged_dicts_123)
        self.assertEqual(merge12, merged_dicts_12)
        self.assertEqual(merge13, merged_dicts_13)
        self.assertEqual(merge25, dict2)  # dict5 is essentially the same as dict2

    # ensure that the correct error message is generated when two dictionaries are merged that have the same variable with different metadata
    def test_mismatch_error(self):
        with self.assertRaises(ValueError) as cm:
            merge_meta_data([dict2, dict4])
        self.assertEqual(str(cm.exception), merged_dicts_24_msg)

        with self.assertRaises(ValueError) as cm:
            merge_meta_data([dict2, dict6])  # test the larger difference
        self.assertEqual(str(cm.exception), merged_dicts_26_msg)


if __name__ == '__main__':
    unittest.main()
