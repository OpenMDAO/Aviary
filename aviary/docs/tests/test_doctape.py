import unittest
import numpy as np

from openmdao.utils.assert_utils import assert_near_equal, assert_equal_numstrings, assert_equal_arrays

import aviary.docs.tests.utils as doctape


class DocTAPETests(unittest.TestCase):
    """
    Testing the DocTAPE functions to make sure they all run in all supported Python versions
    Docs are only built with latest, but these test will be run with latest and dev as well
    """

    def test_gramatical_list(self):
        string = doctape.gramatical_list(['a', 'b', 'c'])
        assert_equal_numstrings(string, 'a, b, and c')

    def test_check_value(self):
        doctape.check_value(1, 1.0)

    def test_check_contains(self):
        doctape.check_contains(1, [1, 2, 3])

    def test_check_args(self):
        doctape.check_args(doctape.check_args, 'func')

    def test_run_command_no_file_error(self):
        doctape.run_command_no_file_error('python -c "print()"')

    def test_get_attribute_name(self):
        class dummy_object:
            attr1 = 1
        name = doctape.get_attribute_name(dummy_object, 1)
        assert_equal_numstrings(name, 'attr1')

    def test_get_all_keys(self):
        keys = doctape.get_all_keys({'d1': {'d2': 2}})
        assert_equal_arrays(np.array(keys), np.array(['d1', 'd2']))

    def test_get_value(self):
        val = doctape.get_value({'d1': {'d2': 2}}, 'd1.d2')
        assert_near_equal(val, 2)

    def test_get_previous_line(self):
        something = "something_else"
        line1 = doctape.get_previous_line()
        line2 = doctape.get_previous_line(2)
        assert_equal_numstrings(line1, 'something = "something_else"')
        assert_equal_numstrings(line2[1].strip(), 'line1 = doctape.get_previous_line()')

    def test_get_variable_name(self):
        var = 7
        name = doctape.get_variable_name(var)
        assert_equal_numstrings(name, 'var')

    # requires IPython shell
    # def test_glue_variable(self):
    #     doctape.glue_variable('plain_text')

    # requires IPython shell
    # def test_glue_keys(self):
    #     doctape.glue_keys({'d1':{'d2':2}})


if __name__ == '__main__':
    unittest.main()