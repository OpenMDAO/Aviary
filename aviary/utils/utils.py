"""
A home for helper functions that are used in multiple files in aviary/utils
Helps to avoid circular imports
"""
from enum import Enum

import numpy as np

from openmdao.utils.units import convert_units

_valid_iterables = (list, np.ndarray, tuple)


def isiterable(val):
    """
    Checks if provided value is an iterable, as defined by the _valid_iterables global
    variable

    Parameters
    ----------
    val : any object
        The object that will be checked if it is a supported iterable
    """
    return isinstance(val, _valid_iterables)


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
    value, units = val_unit_tuple

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
