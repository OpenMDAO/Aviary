import unittest

from openmdao.utils.assert_utils import assert_near_equal

from aviary.utils.aviary_options_dict import AviaryOptionsDictionary


class TestAviaryOptionsDict(unittest.TestCase):
    """Test conversion of aero table from GASP and FLOPS data format to Aviary format."""

    def test_declare_options(self):
        opts = AviaryOptionsDictionary()

        opts.declare('simple', default=3.5)
        self.assertEqual(opts['simple'], 3.5)

        opts.declare('simple_units', default=5.5, units='ft')
        self.assertEqual(opts['simple_units'], (5.5, 'ft'))

        opts.declare('bounds_units', default=(1.2, 4.3), units='ft')
        self.assertEqual(opts['bounds_units'], ((1.2, 4.3), 'ft'))

    def test_get_val(self):
        opts = AviaryOptionsDictionary()

        opts.declare('simple_units', default=2.0, units='ft')
        assert_near_equal(opts.get_val('simple_units', units='inch'), 24.0)

        opts.declare('bounds_units', default=(5.0, 6.0), units='ft')
        assert_near_equal(opts.get_val('bounds_units', units='inch'), (60.0, 72.0))

    def test_load_on_init(self):
        class TestOptions(AviaryOptionsDictionary):
            def declare_options(self):
                self.declare(
                    'p1',
                    default=3.0,
                )

                self.declare('p1_units', default=3.0, units='ft')

        data = {
            'p1': 5.5,
            'p1_units': (12.0, 'inch'),
        }

        opts = TestOptions(data)
        self.assertEqual(opts['p1'], 5.5)
        self.assertEqual(opts['p1_units'], (1.0, 'ft'))

        # Support for legacy unitless.
        data = {
            'p1': (6.5, 'unitless'),
            'p1_units': (12.0, 'inch'),
        }

        opts = TestOptions(data)
        self.assertEqual(opts['p1'], 6.5)

    def test_errors(self):
        opts = AviaryOptionsDictionary(parent_name='builder')

        opts.declare('simple', default=3.5)
        with self.assertRaises(AttributeError) as cm:
            opts.get_val('simple', units='ft')

        err_text = "builder: Option 'simple' does not have declared units."
        self.assertEqual(str(cm.exception), err_text)


if __name__ == '__main__':
    unittest.main()
