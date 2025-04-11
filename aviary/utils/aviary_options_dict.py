from enum import Enum

import numpy as np

import openmdao.api as om
from openmdao.core.constants import _UNDEFINED
from openmdao.utils.units import convert_units


def units_setter(opt_meta, value):
    """
    Convert units for a tuple with form (val, "unitstring").

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
    new_val, new_units = value
    old_val, units = opt_meta['val']

    if new_val is not None:
        new_val = convert_units(new_val, new_units, units)

    return (new_val, units)


def bounds_units_setter(opt_meta, value):
    """
    Convert units for a tuple with form ((val1, val2, ...), "unitstring").

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
    val_tuple, new_units = value
    _, units = opt_meta['val']

    if units != new_units:
        val_list = []
        for val in val_tuple:
            if val is not None:
                val = convert_units(val, new_units, units)
                val_list.append(val)

        val_tuple = tuple(val_list)

    return (val_tuple, units)


def int_enum_setter(opt_meta, value):
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
    for type_ in types:
        if type_ not in (list, np.ndarray):
            enum_class = type_
            break

    if isinstance(value, Enum):
        return value

    elif isinstance(value, int):
        return enum_class(value)

    elif isinstance(value, str):
        return getattr(enum_class, value)

    elif isinstance(value, list):
        values = []
        for val in value:
            if isinstance(val, Enum):
                values.append(val)
            elif isinstance(val, int):
                values.append(enum_class(val))
            elif isinstance(val, str):
                values.append(getattr(enum_class, val))
            else:
                break
        else:
            return values

    msg = f"Value '{value}' not valid for option with types {enum_class}"
    raise TypeError(msg)


class AviaryOptionsDictionary(om.OptionsDictionary):
    """
    Modified OptionsDictionary that is used by Aviary to store the user_options
    for a phase.

    This subclass adds support for declaring options and getting options with units.

    Parameters
    ----------
    data : dict
        Dictionary of option name: value to set.
    parent_name : str
        Name or class name of System that owns this OptionsDictionary.
    """

    def __init__(self, data=None, parent_name=None):
        super().__init__(parent_name)

        self.declare_options()

        if data is None:
            return

        # Loop over all user_options and set them.
        for name, val in data.items():

            # Support for legacy format (unitless)
            if (isinstance(val, tuple) and
                self._dict[name]['set_function'] is None and
                    val[1] == "unitless"):
                val = val[0]

            self[name] = val

    def declare_options(self):
        """
        Hook for declaring options for a phase builder.
        """
        pass

    def declare(self, name, default=_UNDEFINED, values=None, types=None, desc='', units=None,
                upper=None, lower=None, check_valid=None, allow_none=False, deprecation=None):
        r"""
        Declare an option.

        The value of the option must satisfy the following:
        1. If values only was given when declaring, value must be in values.
        2. If types only was given when declaring, value must satisfy isinstance(value, types).
        3. It is an error if both values and types are given.

        Parameters
        ----------
        name : str
            Name of the option.
        default : object or Null
            Optional default value that must be valid under the above 3 conditions.
        values : set or list or tuple or None
            Optional list of acceptable option values.
        types : type or tuple of types or None
            Optional type or list of acceptable option types.
        desc : str
            Optional description of the option.
        units : str
            Units associated with the quantity in values.
        upper : float or None
            Maximum allowable value.
        lower : float or None
            Minimum allowable value.
        check_valid : function or None
            User-supplied function with arguments (name, value) that raises an exception
            if the value is not valid.
        allow_none : bool
            If True, allow None as a value regardless of values or types.
        deprecation : str or tuple or None
            If None, it is not deprecated. If a str, use as a DeprecationWarning
            during __setitem__ and __getitem__.  If a tuple of the form (msg, new_name),
            display msg as with str, and forward any __setitem__/__getitem__ to new_name.
        """

        if units is not None:

            if isinstance(default, tuple):
                set_function = bounds_units_setter
            else:
                set_function = units_setter

            default = (default, units)
            types = tuple

        else:
            set_function = None

        super().declare(
            name,
            default=default,
            values=values,
            types=types,
            desc=desc,
            upper=upper,
            lower=lower,
            check_valid=check_valid,
            allow_none=allow_none,
            set_function=set_function,
            deprecation=deprecation,
        )

    def get_val(self, key, units=None):
        """
        Return the current value for the requested default.

        Parameters
        ----------
        key : str
            the name of the item
        units : str ('unitless')
            the units of the returned value

        Returns
        -------
        val
        """

        if units is not None:

            if self._dict[key]['set_function'] is None:
                self._raise(f"Option '{key}' does not have declared units.",
                            exc_type=AttributeError)

            val, base_units = self[key]

            if units != base_units:

                if isinstance(val, tuple):
                    val_list = []
                    for single_val in val:
                        if single_val is not None:
                            single_val = convert_units(single_val, base_units, units)
                            val_list.append(single_val)

                    val = tuple(val_list)

                else:
                    val = convert_units(val, base_units, units)

        else:
            val = self[key]

        return val

    def to_phase_info(self):
        """
        Returns an equivalent phase_info dictionary for this options dict.

        Returns
        -------
        dict
            Equivalent phase_info.
        """
        phase_info = {}
        for name, val in self.items():
            phase_info[name] = val

        return phase_info
