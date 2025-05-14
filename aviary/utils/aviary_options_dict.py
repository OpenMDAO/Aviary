import openmdao.api as om
from openmdao.core.constants import _UNDEFINED

from aviary.utils.utils import wrapped_convert_units


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
    new_val, _ = value
    _, units = opt_meta['val']

    if new_val is not None:
        new_val = wrapped_convert_units(value, units)

    return (new_val, units)


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
            if (
                isinstance(val, tuple)
                and self._dict[name]['set_function'] is None
                and val[1] == 'unitless'
            ):
                val = val[0]

            self[name] = val

    def declare_options(self):
        """Hook for declaring options for a phase builder."""
        pass

    def declare(
        self,
        name,
        default=_UNDEFINED,
        values=None,
        types=None,
        desc='',
        units=None,
        upper=None,
        lower=None,
        check_valid=None,
        allow_none=False,
        deprecation=None,
    ):
        if units is not None:
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
                self._raise(
                    f"Option '{key}' does not have declared units.",
                    exc_type=AttributeError,
                )

            val, base_units = self[key]

            if units != base_units:
                val = wrapped_convert_units((val, base_units), units)

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
