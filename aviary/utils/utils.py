"""
A home for helper functions that are used in multiple files in aviary/utils
Helps to avoid circular imports
"""
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
