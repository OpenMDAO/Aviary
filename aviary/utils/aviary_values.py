"""
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
"""

from openmdao.utils.units import convert_units as _convert_units

from aviary.utils.named_values import NamedValues, get_items, get_keys, get_values
from aviary.utils.utils import cast_type, check_type
from aviary.variable_info.variable_meta_data import _MetaData

# TODO: workaround to avoid unused imports - a better solution is desired such as utils or making
#       get_*() methods of NamedValues
get_items = get_items
get_keys = get_keys
get_values = get_values


class AviaryValues(NamedValues):
    """Define a collection of aviary values with associated units and aviary tests."""

    def set_val(self, key, val, units='unitless', meta_data=_MetaData):
        """
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
        """
        if key in meta_data:
            val = cast_type(key, val, meta_data)
            check_type(key, val, meta_data)

            self._check_units_compatibility(key, val, units, meta_data=meta_data)

        super().set_val(key=key, val=val, units=units)

    def _check_units_compatibility(self, key, val, units, meta_data=_MetaData):
        """
        Check that the two provided units are compatible - we don't actually want to
        convert here, just verify that the provided units are allowed.
        """
        expected_units = meta_data[key]['units']

        try:
            # NOTE the value here is unimportant, we only care if OpenMDAO will
            # convert the units
            _convert_units(10, expected_units, units)
        except ValueError:
            raise ValueError(f'The units {units} which you have provided for {key} are invalid.')
        except TypeError:
            raise TypeError(
                f'The base units of {key} are {expected_units}, and you have tried to '
                f'set {key} with units of {units}, which are not compatible.'
            )
        except BaseException:
            raise KeyError('There is an unknown error with your units.')

    def items(self):
        """
        Return (name, value) for variables contained in this vector.

        Note that AviaryValues is not a dictionary, but this adds support for iterating over
        its contents.

        Yields
        ------
        str
            The name of an item.
        object
            The value of that item.
        """
        for key, val in self._mapping.items():
            yield key, val
