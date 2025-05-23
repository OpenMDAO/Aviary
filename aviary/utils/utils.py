"""
A home for helper functions that are used in multiple files in aviary/utils
Helps to avoid circular imports. These functions do not rely on imports from other
utility files.
"""

from copy import deepcopy
from enum import Enum

import numpy as np
from openmdao.utils.units import convert_units

from aviary.variable_info.variable_meta_data import _MetaData


def isiterable(val, valid_iterables: tuple = (list, np.ndarray, tuple)):
    """
    Checks if provided value is an iterable, as defined by the _valid_iterables global
    variable.

    Parameters
    ----------
    val : any object
        The object that will be checked if it is a supported iterable type
    valid_iterables : tuple of types
        Which iterable types val will be checked against. By default, lists, numpy arrays
        and tuples are supported types
    """
    return isinstance(val, valid_iterables)


def wrapped_convert_units(val_unit_tuple, new_units):
    """
    Wrapper for OpenMDAO's convert_units function. Can handle iterable values.

    Parameters
    ----------
    val_unit_tuple : tuple
        Tuple of the form (value, units) where value is a float and units is a
        string.
    new_units : str
        New units to convert to.

    Returns
    -------
    value: float, list, np.ndarray, tuple
        Value converted to new units, as the same type as provided
    """
    value, units = deepcopy(val_unit_tuple)

    # can't convert units on None; return None
    if value is None:
        return None

    if isiterable(value):
        # tuples are immutable, so we have to convert to list to modify each index
        if isinstance(value, tuple):
            istuple = True
            value = list(value)
        else:
            istuple = False

        for i, item in enumerate(value):
            # Any entry may be none.
            if value[i] is not None:
                value[i] = convert_units(item, units, new_units)

        if istuple:
            value = tuple(value)
    else:
        value = convert_units(value, units, new_units)

    return value


def enum_setter(opt_meta, value):
    """
    Support setting the option with a string or int and converting it to the
    proper enum object.

    Parameters
    ----------
    opt_meta : dict
        Dictionary of entries for the option.
    value : any
        New value for the option.

    Returns
    -------
    any
        Post processed value to set into the option.
    """
    types = opt_meta['types']
    if not isinstance(types, tuple):
        types = (types,)
    for type_ in types:
        if issubclass(type_, Enum):
            enum_class = type_
            break

    if isinstance(value, Enum):
        return value

    elif isinstance(value, int):
        return enum_class(value)

    elif isinstance(value, str):
        try:
            return enum_class(value)
        except ValueError:
            return getattr(enum_class, value.upper())

    elif isiterable(value):
        # Numpy arrays have unique typing (float64, etc.), convert to list of standard
        # python types
        if isinstance(value, np.ndarray):
            value_iter = value.tolist()
        else:
            value_iter = value
        values = []
        for val in value_iter:
            if isinstance(val, Enum):
                values.append(val)
            elif isinstance(val, int):
                values.append(enum_class(val))
            elif isinstance(val, str):
                try:
                    # see if str maps to ENUM value
                    return enum_class(val)
                except ValueError:
                    # str instead maps to ENUM name
                    return getattr(enum_class, val.upper())
            else:
                break
        else:
            # maintain the same type of iterable
            if isinstance(value, np.ndarray):
                values = np.array(values)
            else:
                values = type(value)(values)

        return values

    msg = f"Value '{value}' not valid for option with types {enum_class}"
    raise TypeError(msg)


def check_type(key, val, meta_data=_MetaData):
    """
    Check that provided val is the correct type. If val is iterable, also check each
    individual index.
    """

    def _flatten_iters(iterable):
        """Flattens iterables of any type and dimension."""
        for item in iterable:
            try:
                yield from iter(item)
            except TypeError:
                yield item

    # make a copy of val, so we do not modify it
    input_val = deepcopy(val)
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
                f'{type(input_val)}.'
            )

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
        if isinstance(expected_types, type):
            if expected_types is bool:
                has_bool = True
        elif bool in expected_types:
            has_bool = True
        if (not isinstance(item, expected_types)) or (
            (has_bool is False) and (isinstance(item, bool))
        ):
            raise TypeError(
                f'{key} is of type(s) {meta_data[key]["types"]} but you '
                f'have provided a value of type {type(item)}.'
            )


def cast_type(key, val, meta_data=_MetaData):
    """
    Attempts to cast val into an accepted type in meta_data. If a valid type cast is
    found, val is changed to that type. If no compatible type is found, val is returned
    as is. Typing preference given by order of types defined in meta_data.
    """
    if key not in meta_data:
        return val

    cast_val = deepcopy(val)

    expected_types = meta_data[key]['types']
    if not isinstance(expected_types, tuple):
        expected_types = (expected_types,)

    # If provided val is not in expected types, see if it can be casted to one of
    # them (e.g. cast int to float).
    if not isinstance(cast_val, expected_types):
        # Prefer casting to Enum if possible
        # Special handling to access an Enum member from either the member name
        # or its value.
        is_enum = False
        for _type in expected_types:
            if issubclass(_type, Enum):
                is_enum = True
                break
        if is_enum:
            cast_val = enum_setter(meta_data[key], cast_val)
        else:
            for _type in expected_types:
                try:
                    if isiterable(cast_val):
                        if isinstance(cast_val, np.ndarray):
                            cast_val = np.array([_type(item) for item in cast_val])
                        else:
                            cast_val = type(cast_val)([_type(item) for item in cast_val])
                    # Don't cast things to bool, most types will convert and we
                    # don't actually want that
                    elif _type is not bool:
                        if _type is np.ndarray:
                            cast_val = np.array([cast_val])
                        else:
                            cast_val = _type(cast_val)
                except (ValueError, TypeError):
                    # try another value in expected_types
                    pass
                else:
                    break

    return cast_val
