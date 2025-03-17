'''
Define utilities for using aviary values with associated units and testing
for compatibility with aviary metadata dictionary.

Utilities
---------
Units : type alias
    define a type hint for associated units

ValueAndUnits : type alias
    define a type hint for a single value paired with its associated units

OptionalValueAndUnits : type alias
    define a type hint for an optional single value paired with its associated units

class AviaryValues
    define a collection of named values with associated units
'''
from enum import Enum

import numpy as np
from openmdao.utils.units import convert_units as _convert_units

from aviary.utils.named_values import NamedValues, get_items, get_keys, get_values
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.utils.utils import isiterable, enum_setter


class AviaryValues(NamedValues):
    '''
    Define a collection of aviary values with associated units and aviary tests.
    '''

    def set_val(self, key, val, units='unitless', meta_data=_MetaData):
        '''
        Update the named value and its associated units.

        Note, specifying units of `None` or units of any type other than `str` will raise
        `Typerror`.

        Parameters
        ----------
        key : str
            the name of the item

        val : Any
            the new value of the item

        units : str ('unitless')
            the units associated with the new value, if any

        Raises
        ------
        TypeError
            if units of `None` were specified or units of any type other than `str`
        '''
        if key in meta_data.keys():
            expected_types = meta_data[key]['types']
            if not isinstance(expected_types, tuple):
                expected_types = (expected_types, )

            # If provided val is not in expected types, see if it can be casted to one of
            # them (e.g. cast int to float).
            if not isinstance(val, expected_types):
                # Prefer casting to Enum if possible
                # Special handling to access an Enum member from either the member name
                # or its value.
                is_enum = False
                for _type in expected_types:
                    if issubclass(_type, Enum):
                        is_enum = True
                        break
                if is_enum:
                    val = enum_setter(meta_data[key], val)
                else:
                    for _type in expected_types:
                        try:
                            if isiterable(val):
                                if isinstance(val, np.ndarray):
                                    val = np.array([_type(item) for item in val])
                                else:
                                    val = type(val)([_type(item) for item in val])
                            # Don't cast things to bool, most types will convert and we
                            # don't actually want that
                            elif _type is not bool:
                                if _type is np.ndarray:
                                    val = np.array([val])
                                else:
                                    val = _type(val)
                        except (ValueError, TypeError):
                            # try another value in expected_types
                            pass
                        else:
                            break

            self._check_type(key, val, meta_data=meta_data)
            self._check_units_compatibility(key, val, units, meta_data=meta_data)

        super().set_val(key=key, val=val, units=units)

    def _check_type(self, key, val, meta_data=_MetaData):
        """
        Check that provided val is the correct type. If val is iterable, also check each
        individual index
        """
        # make a copy of val, so we do not modify it
        input_val = val
        expected_types = meta_data[key]['types']
        if expected_types is None:
            # MetaData item has no type requirement.
            return

        # If data is iterable, check that it is allowed to be.
        # Variables flagged multivalue can be lists or numpy arrays even if not specified
        # in `types`
        if isiterable(input_val):
            types = expected_types
            if meta_data[key]['multivalue']:
                if isinstance(expected_types, tuple):
                    types = (list, np.ndarray, tuple, *expected_types)
                else:
                    types = (list, np.ndarray, tuple, expected_types)
            if not isinstance(input_val, types):
                raise TypeError(
                    f'{key} is of type(s) {types} but you have provided a value of type '
                    f'{type(input_val)}.')

        # Numpy arrays have special typings. Convert to list using standard Python types
        # Numpy arrays do not allow mixed types, only have to check one entry
        # Empty arrays do not need this step
        if isinstance(input_val, np.ndarray) and len(input_val) > 0:
            input_val = input_val.tolist()
            while isiterable(input_val):
                input_val = input_val[0]

        # if val is not iterable, make it a list (checks assume val is iterable)
        if not isiterable(input_val):
            input_val = [input_val]
        # if val is an iterable, flatten it so we can easily loop over every entry
        else:
            input_val = _flatten_iters(input_val)

        for item in input_val:
            has_bool = False  # needs some fancy shenanigans because bools will register as ints
            if (isinstance(expected_types, type)):
                if expected_types is bool:
                    has_bool = True
            elif bool in expected_types:
                has_bool = True
            if (not isinstance(item, expected_types)) or (
                    (has_bool == False) and (isinstance(item, bool))):
                raise TypeError(
                    f'{key} is of type(s) {meta_data[key]["types"]} but you '
                    f'have provided a value of type {type(item)}.')

    def _check_units_compatibility(self, key, val, units, meta_data=_MetaData):
        expected_units = meta_data[key]['units']

        try:
            # NOTE the value here is unimportant, we only care if OpenMDAO will
            # convert the units
            _convert_units(10, expected_units, units)
        except ValueError:
            raise ValueError(
                f'The units {units} which you have provided for {key} are invalid.')
        except TypeError:
            raise TypeError(
                f'The base units of {key} are {expected_units}, and you have tried to '
                f'set {key} with units of {units}, which are not compatible.')
        except BaseException:
            raise KeyError('There is an unknown error with your units.')


def _flatten_iters(iterable):
    """Flattens iterables of any type and dimension"""
    for item in iterable:
        try:
            yield from iter(item)
        except TypeError:
            yield item
