import unittest

import numpy as np
from openmdao.utils.assert_utils import (
    assert_equal_arrays,
    assert_equal_numstrings,
    assert_near_equal,
)

from aviary.utils.doctape import (
    check_args,
    check_contains,
    check_value,
    get_all_keys,
    get_attribute_name,
    get_previous_line,
    get_value,
    get_variable_name,
    glue_keys,
    glue_variable,
    gramatical_list,
    run_command_no_file_error,
)


class DocTAPETests(unittest.TestCase):
    """
    Testing the DocTAPE functions to make sure they all run in all supported Python versions
    Docs are only built with latest, but these test will be run with latest and dev as well.
    """

    def test_gramatical_list(self):
        string = gramatical_list(['a', 'b', 'c'])
        assert_equal_numstrings(string, 'a, b, and c')

    def test_check_value(self):
        check_value(1, 1.0)

    def test_check_contains(self):
        check_contains(1, [1, 2, 3])

    def test_check_args(self):
        check_args(check_args, 'func')

    def test_run_command_no_file_error(self):
        run_command_no_file_error('python -c "print()"')

    def test_get_attribute_name(self):
        class dummy_object:
            attr1 = 1

        name = get_attribute_name(dummy_object, 1)
        assert_equal_numstrings(name, 'attr1')

    def test_get_all_keys(self):
        keys = get_all_keys({'d1': {'d2': 2}})
        assert_equal_arrays(np.array(keys), np.array(['d1', 'd2']))

    def test_get_value(self):
        val = get_value({'d1': {'d2': 2}}, 'd1.d2')
        assert_near_equal(val, 2)

    def test_get_previous_line(self):
        line1 = get_previous_line()
        line2 = get_previous_line(2)
        assert_equal_numstrings(line2[0].strip(), line1)
        assert_equal_numstrings(line2[1].strip(), 'line1 = get_previous_line()')

    def test_get_variable_name(self):
        var = 7
        name = get_variable_name(var)
        assert_equal_numstrings(name, 'var')

    # requires IPython shell
    def test_glue_variable(self):
        glue_variable('plain_text', display=False)

    # requires IPython shell
    def test_glue_variable_non_str(self):
        glue_variable((9, 'ft'), display=False)

    # requires IPython shell
    def test_glue_keys(self):
        glue_keys({'d1': {'d2': 2}}, display=False)


if __name__ == '__main__':
    unittest.main()
