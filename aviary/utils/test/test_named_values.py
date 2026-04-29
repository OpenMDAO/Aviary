"""Unit test cases for class NamedValues."""

import unittest

from aviary.utils.named_values import NamedValues
from aviary.variable_info.variables import Aircraft, Mission


class NamedValuesTest(unittest.TestCase):
    """Test NamedValues class."""

    def test_init(self):
        d = NamedValues()
        self._do_test_full_equal(d, _empty, [_data1, _data2, _data3])
        a = NamedValues(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues(_empty)
        self._do_test_full_equal(d, _empty, [_data1, _data2, _data3])
        a = NamedValues(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues(**_empty)
        self._do_test_full_equal(d, _empty, [_data1, _data2, _data3])
        a = NamedValues(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues(_data1)
        self._do_test_full_equal(d, _data1, [_empty, _data2, _data3])
        a = NamedValues(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues(**_data2)
        self._do_test_full_equal(d, _data2, [_empty, _data1, _data3])
        a = NamedValues(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues(_data1, **_data2)
        self._do_test_full_equal(d, _data3, [_empty, _data1, _data2])
        a = NamedValues(d)
        self._do_test_full_equal(d, a, ())

    def test_update(self):
        d = NamedValues()
        d.update()
        self._do_test_full_equal(d, _empty, [_data1, _data2, _data3])
        a = NamedValues()
        a.update(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues()
        d.update(_empty)
        self._do_test_full_equal(d, _empty, [_data1, _data2, _data3])
        a = NamedValues()
        a.update(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues()
        d.update(**_empty)
        self._do_test_full_equal(d, _empty, [_data1, _data2, _data3])
        a = NamedValues()
        t = tuple(d)
        a.update(t)
        self._do_test_full_equal(d, a, ())

        d = NamedValues()
        d.update(_data1)
        self._do_test_full_equal(d, _data1, [_empty, _data2, _data3])
        a = NamedValues()
        a.update(d)
        self._do_test_full_equal(d, a, ())

        d = NamedValues()
        d.update(**_data2)
        self._do_test_full_equal(d, _data2, [_empty, _data1, _data3])
        a = NamedValues()
        t = tuple(d)
        a.update(t)
        self._do_test_full_equal(d, a, ())

        d = NamedValues()
        d.update(_data1, **_data2)
        self._do_test_full_equal(d, _data3, [_empty, _data1, _data2])
        a = NamedValues()
        t = tuple(d)
        a.update(t)
        self._do_test_full_equal(d, a, ())

    def test_units(self):
        a = NamedValues()
        d = NamedValues(_data1)

        self._do_test_full_equal(d, _data1, (a, _empty, _data2, _data3))

        for key in _data1:
            a.set_val(key, *(d.get_item(key)))

        self._do_test_full_equal(a, d, (_empty, _data2, _data3))
        self._do_test_full_equal(d, _data1, (_empty, _data2, _data3))

        for key in _data1:
            # units support
            val, units = _data1[key]
            aval, aunits = a.get_item(key)

            self.assertEqual(val, aval)
            self.assertEqual(units, aunits)

        self.assertFalse(_nokey in _data1)

        try:
            aval, aunits = a.get_item(_nokey)
        except KeyError as e:
            if str(e) != "'key not found: ## no key ##'":
                print(str(e))
                raise AssertionError(
                    'Did not receive expected error message for empty key in get_items()'
                )

        val = 42
        units = 'min'
        original_val = 42
        original_units = 'min'

        try:
            aval, aunits = a.get_item(_nokey)
        except KeyError:
            pass
        else:
            raise AssertionError(
                'Did not receive expected error message for nonexistent key in get_items()'
            )

        self.assertEqual(val, original_val)
        self.assertEqual(units, original_units)

        key = 'elapsed_time'
        a.set_val(key, val, units)

        val = a.get_val(key, units)

        self.assertEqual(val, original_val)

        new_val = 42 * 60  # min -> s

        val = a.get_val(key, 's')

        self.assertEqual(val, new_val)

        # unit conversion is local; item is unchanged
        val, units = a.get_item(key)

        self.assertEqual(val, original_val)
        self.assertEqual(units, original_units)

        # still missing
        try:
            aval, aunits = a.get_item(_nokey)
        except KeyError as e:
            if str(e) != "'key not found: ## no key ##'":
                print(str(e))
                raise AssertionError(
                    'Did not receive expected error message for empty key in get_items()'
                )

        key = _nokey
        val = 314159
        # default units ('unitless')

        a.set_val(key, val)
        aval, aunits = a.get_item(key)

        self.assertEqual(val, aval)
        self.assertEqual(aunits, 'unitless')

    def test_collection(self):
        self.assertNotEqual(len(_data1), 0)
        d = NamedValues(_data1)

        self._do_test_full_equal(d, _data1, (_empty, _data2, _data3))

        # Container
        for key in _data1:
            self.assertTrue(key in d)

        self.assertTrue(_nokey not in d)
        self.assertFalse(_nokey in d)

        # Iterable
        a = tuple(_data1.items())
        b = tuple(iter(d))
        self.assertEqual(a, b)

        b = tuple(d)
        self.assertEqual(a, b)

        b = tuple(d.items())
        self.assertEqual(a, b)

        a = tuple(iter(_data1))
        b = tuple(d.keys())
        self.assertEqual(a, b)

        a = tuple(_data1.values())
        b = tuple(d.values())
        self.assertEqual(a, b)

        # Sized
        self.assertEqual(len(_data1), len(d))

    def test_delete(self):
        # create NamedValues collection a from dictionary _data1.
        a = NamedValues(_data1)
        self._do_test_full_equal(a, _data1, ())

        # remove an item from a.
        a.delete('NUM_FUSELAGES')
        # remove the same item from the dictionary and then make NamedValues collection b
        _data4 = _data1.copy()
        _data4.pop('NUM_FUSELAGES')
        b = NamedValues(_data4)
        # make sure a and b are the same.
        self._do_test_full_equal(a, b, ())

        # make sure it is deleted
        try:
            aval, aunits = a.get_item('NUM_FUSELAGES')
        except KeyError:
            pass
        else:
            raise AssertionError(
                'Did not receive expected error message for nonexistent key in get_items()'
            )

        # size
        self.assertEqual(len(_data4), len(a))
        self.assertEqual(len(_data4), len(b))

        # delete all
        for key, _ in b:
            a.delete(key)
        self._do_test_full_equal(a, _empty, ())

    def _do_test_full_equal(self, d, eq, ne):
        self.assertEqual(d._mapping, eq)
        self.assertEqual(d, eq)
        self.assertEqual(eq, d)

        for a in ne:
            self.assertNotEqual(d, a)
            self.assertNotEqual(a, d)


_nokey = '## no key ##'

_empty = {}

_data1 = {
    'NUM_FUSELAGES': (1, 'unitless'),
    'NUM_ENGINES': (2, 'unitless'),
    Aircraft.CrewPayload.BAGGAGE_MASS: (7500, 'lbm'),
    Aircraft.Design.RANGE: (3500, 'NM'),
}


def _nounits(a):
    d = {}

    for key, (val, _) in a.items():
        d[key] = val

    return d


_data1_nounits = _nounits(_data1)

_data2 = {
    'NUM_FUSELAGES': (2, 'unitless'),
    'NUM_ENGINES': (4, 'unitless'),
    'PASSENGER_MASS_TOTAL': (200, 'lbm'),
}

# _data2_nounits = _nounits(_data2)

_data3 = dict(_data1, **_data2)
# _data3_nounits = _nounits(_data3)


if __name__ == '__main__':
    unittest.main()
