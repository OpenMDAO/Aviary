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
    glue_class_functions,
    glue_class_options,
    glue_keys,
    glue_variable,
    gramatical_list,
    run_command_no_file_error,
    get_all_non_aviary_names,
)

try:
    import myst_nb
except ImportError:
    myst_nb = False


@unittest.skipIf(
    myst_nb is False,
    'Skipping because myst_nb is not installed for doc testing.',
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

    def test_glue_class_functions(self):
        from aviary.interface.methods_for_level2 import AviaryProblem

        curr_glued = []
        glue_class_functions(AviaryProblem, curr_glued, prefix='zz')

        self.assertTrue('load_inputs' in curr_glued)
        self.assertTrue('load_inputs()' in curr_glued)
        self.assertTrue('zz.load_inputs' in curr_glued)
        self.assertTrue('zz.load_inputs()' in curr_glued)

    def test_glue_class_options_attributes(self):
        from aviary.core.aviary_group import AviaryGroup

        curr_glued = []
        glue_class_options(AviaryGroup, curr_glued)

        self.assertTrue('auto_order' in curr_glued)
        self.assertTrue('mission_info' in curr_glued)

    def test_get_all_non_aviary_names(self):
        from aviary.subsystems.aerodynamics.gasp_based.gaspaero import UFac

        names = get_all_non_aviary_names(UFac)
        expected_names = ['lift_ratio', 'bbar_alt', 'sigma', 'sigstr', 'ufac']
        assert_equal_arrays(np.array(names), np.array(expected_names))


if __name__ == '__main__':
    unittest.main()
    # test = DocTAPETests()
    # test.test_get_all_non_aviary_names()
