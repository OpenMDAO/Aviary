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

    def add_state_options(self, state_name: str, units: str=None, defaults=None):
        """
        Adds all options needed for a state variable.

        For a state named mass, these are mass_initial, mass_final, mass_bounds, mass_ref,
        mass_ref0, mass_defect_ref, mass_solve_segments.

        Parameters
        ----------
        state_name : str
            Name of this state.
        units : str
            Units for this state if it has them.
        defaults : dict or None
            Optional dictionary of default values for any state option.
        """
        if defaults is None:
            defaults = {}

        name = f'{state_name}_initial'
        default = defaults.get(name, None)
        desc = f'Tuple of (value, units) containing value of {state_name} '
        desc += 'at the start of the phase.\n'
        desc += 'When unspecified, the value comes from upstream.\n'
        desc += f'When specified, a constraint is created on the initial {state_name}.'
        self.declare(
            name=name,
            default=default,
            types=tuple,
            allow_none=True,
            units=units,
            desc=desc,
        )
    
        name = f'{state_name}_final'
        default = defaults.get(name, None)
        desc = f'Tuple of (value, units) containing value of {state_name} '
        desc += 'at the end of the phase.\n'
        desc += 'If this phase is connected to a downstream phase, final values should be '
        desc += f'specified with {state_name}_initial in that phase instead of here.\n'
        desc += f'When specified, a constraint is created on the final {state_name}.'
        self.declare(
            name=name,
            default=default,
            types=tuple,
            allow_none=True,
            units=units,
            desc=desc,
        )

        name = f'{state_name}_bounds'
        default = defaults.get(name, (None, None))
        desc = 'Tuple of form ((lower, upper), units) containing the upper and lower bounds for '
        desc += f'all values of {state_name} in the phase.\n'
        desc += 'The default of None for upper or lower means that bound will not be declared.\n'
        self.declare(
            name=name,
            default=default,
            types=tuple,
            units=units,
            desc=desc,
        )

        name = f'{state_name}_ref'
        default = defaults.get(name, 1.0)
        desc = f'Multiplicative scale factor "ref" for {state_name}.\n'
        desc += 'Default is 1.0'
        self.declare(
            name=name,
            default=default,
            types=float,
            units=units,
            desc=desc,
        )

        name = f'{state_name}_ref0'
        default = defaults.get(name, 0.0)
        desc = f'Additive scale factor "ref0" for {state_name}.\n'
        desc += 'Default is 0.0'
        self.declare(
            name=name,
            default=default,
            types=float,
            units=units,
            desc=desc,
        )

        name = f'{state_name}_defect_ref'
        default = defaults.get(name, None)
        desc = f'Multiplicative scale factor "ref" for the {state_name} defect constraint.\n'
        desc += 'Default is None, which means the ref and defect_ref are the same.'
        self.declare(
            name=name,
            default=default,
            types=float,
            allow_none=True,
            units=units,
            desc=desc,
        )

        name = f'{state_name}_solve_segments'
        default = defaults.get(name, False)
        desc = 'When True, a solver will be used to converge the collocation defects within a '
        desc += 'segment. Note that the state continuity defects between segements will still be '
        desc += 'handled by the optimizer.'
        self.declare(
            name=name,
            default=default,
            types=bool,
            desc=desc,
        )

    def add_control_options(self, state_name: str, units: str=None, defaults=None):
        """
        Adds all options needed for a control variable.

        For a control named mach, these are mach_initial, mach_final, mach_bounds, mach_ref,
        mach_ref0, mach_defect_ref, mach_solve_segments.

        Parameters
        ----------
        state_name : str
            Name of this state.
        units : str
            Units for this state if it has them.
        defaults : dict or None
            Optional dictionary of default values for any state option.
        """
        if defaults is None:
            defaults = {}
