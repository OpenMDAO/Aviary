'''
Define utilities for using named values with associated units.

Utilities
---------
Units : type alias
    define a type hint for associated units

ValueAndUnits : type alias
    define a type hint for a single value paired with its associated units

OptionalValueAndUnits : type alias
    define a type hint for an optional single value paired with its associated units

class NamedValues
    define a collection of named values with associated units
'''
import copy
from collections.abc import Collection
from typing import Any, Tuple, Union

from openmdao.core.constants import _UNDEFINED
from openmdao.utils.units import convert_units as _convert_units

Units = str
ValueAndUnits = Tuple[Any, Units]
OptionalValueAndUnits = Union[ValueAndUnits, Any]


class NamedValues(Collection):
    '''
    Define a collection of named values with associated units.
    '''

    def __init__(self, other=None, **kwargs):
        '''
        Initialize this collection.

        Notes
        -----
        When intializing from another collection, the following types of
        collections are supported:

            * `NamedValues`
            * `Dict[str, ValueAndUnits]`
            * `Iterable[Tuple[str, ValueAndUnits]]`

        When initializing from keyword arguments, the mapped item must be of a
        type of `ValueAndUnits`.
        '''
        self._mapping = {}

        self.update(other, **kwargs)

    def get_item(self, key, default=(None, None)) -> OptionalValueAndUnits:
        '''
        Return the named value and its associated units.

        Note, this method never raises `KeyError` or `TypeError`.

        Parameters
        ----------
        key : str
            the name of the item

        default : OptionalValueAndUnits (None, None)
            if the item does not exist, return this object

        Returns
        -------
        OptionalValueAndUnits

        See Also
        --------
        get_val
        set_val
        '''
        item = self._mapping.get(key, _UNDEFINED)

        if item is _UNDEFINED:
            return default

        return item

    def copy(self):
        '''
        Return a copy of the instance of this class.

        Parameters
        ---------
        None

        Returns
        -------
        NamedValues()
        '''
        return copy.copy(self)

    def deepcopy(self):
        '''
        Return a deep copy of the instance of this class.

        Parameters
        ---------
        None

        Returns
        -------
        NamedValues()
        '''
        return copy.deepcopy(self)

    def get_val(self, key, units='unitless') -> Any:
        '''
        Return the named value in the specified units.

        Note, requesting a named value that does not exist will raise `KeyError`.

        Note, specifying units of `None` or units of any type other than `str` will raise
        `TypeError`.

        Parameters
        ----------
        key : str
            the name of the item

        units : str ('unitless')
            the units of the returned value

        Returns
        -------
        val

        Raises
        ------
        KeyError
            if the named value does not exist

        TypeError
            if units of `None` were specified or units of any type other than `str`
        '''
        self._check_units('get_val', key, units)

        item = self._mapping.get(key, _UNDEFINED)

        if item is _UNDEFINED:
            raise KeyError(f'KeyError: key not found: {key}')

        val, old_units = item

        if isinstance(val, tuple):
            val = tuple(_convert_units(v, old_units, units) for v in val)
        elif old_units != units:
            val = _convert_units(val, old_units, units)

        return val

    def set_val(self, key, val, units='unitless'):
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
        self._check_units('set_val', key, units)

        self._mapping[key] = (val, units)

    def __repr__(self):
        '''
        Return a string containing a printable representation of the collection.
        '''
        return repr(self._mapping)

    def clear(self):
        '''
        Remove all items from the collection.
        '''
        self._mapping.clear()

    def update(self, other=None, **kwargs):
        '''
        Assign named values and their associated units found in another
        collection to this collection, overwriting existing items.

        If keyword arguments are specified, the collection is then assigned those
        named values and their associated units, overwriting existing items.

        Parameters
        ----------
        other (None)
            a collection of named values and their associated units

        **kwargs (optional)
            individual named values and their associated units

        Notes
        -----
        The following types of collections are supported
            * `NamedValues`
            * `Dict[str, ValueAndUnits]`
            * `Iterable[Tuple[str, ValueAndUnits]]`

        When assigning from keyword arguments, the mapped item must be of a
        type of `ValueAndUnits`.
        '''
        if not (other or kwargs):
            return

        set_val = self.set_val

        if isinstance(other, type(self)):
            # NamedValues
            other = other._mapping

        if other is not None:
            # check for dictionary
            keys = getattr(other, 'keys', None)

            if keys is None:
                # iterable, but not dictionary
                for key, (val, units) in other:
                    set_val(key, val, units)

            else:
                # dictionary
                for key in keys():
                    val, units = other[key]
                    set_val(key, val, units)

        for key, (val, units) in kwargs.items():
            set_val(key, val, units)

    def delete(self, key):
        '''
        Remove the named value and its associated units.

        Raises
        ------
        KeyError
            if the named value does not exist
        '''
        try:
            del self._mapping[key]

        except KeyError:
            raise KeyError(f'KeyError: key not found: {key}')

    def __eq__(self, other):
        '''
        Return whether or not this collection is equivalent to another.
        '''
        collection = self._mapping

        if isinstance(other, type(self)):
            return collection == other._mapping

        return collection == other

    def __contains__(self, key):
        '''
        Return whether or not the named value exists.
        '''
        return key in self._mapping

    def __iter__(self):
        '''
        Return an iterator over the `(key, (val, units))` data stored in this collection.
        '''
        items = self._mapping.items()

        yield from items

    def __len__(self):
        '''
        Return the number of items in this collection.
        '''
        return len(self._mapping)

    def _check_units(self, funcname, key, units):
        '''
        If units of `None` were specified or units of any type other than `str`, raise
        `TypeError`. Otherwise, do nothing.

        Parameters
        ----------
        funcname : str
            the name of a method/function

        key : str
            the name of the item

        units : Any
            the units to check
        '''
        if ((units is None) or not isinstance(units, str)):
            raise TypeError(
                f'{self.__class__.__name__}: {funcname}({key}):'
                f' unsupported units: {units}'
            )

    __slots__ = ('_mapping',)


def get_keys(named_values: NamedValues):
    '''
    Return a new view of the collection's names.
    '''
    return named_values._mapping.keys()


def get_items(named_values: NamedValues):
    '''
    Return a new view of the collection's `(key, (val, units))`.
    '''
    return named_values._mapping.items()


def get_values(named_values: NamedValues):
    '''
    Return a new view of the collection's `(val, units)`.
    '''
    return named_values._mapping.values()
