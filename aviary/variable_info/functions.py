import openmdao.api as om
from openmdao.core.component import Component
from openmdao.utils.units import convert_units
import dymos as dm
from dymos.utils.misc import _unspecified

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData

# ---------------------------
# Helper functions for setting up inputs/outputs in components
# ---------------------------


def add_aviary_input(comp, varname, val=None, units=None, desc=None, shape_by_conn=False,
                     meta_data=_MetaData, shape=None):
    """
    This function provides a clean way to add variables from the
    variable hierarchy into components as Aviary inputs. It takes
    the standard OpenMDAO inputs of the variable's name, initial
    value, units, and description, as well as the component which
    the variable is being added to.

    Parameters
    ----------
    comp: Component
        OpenMDAO component to add this variable.
    varname: str
        Name of variable.
    val: float or ndarray
        Default value for variable.
    units: str
        (Optional) when speficying val, units should also be specified.
    desc: str
        (Optional) description text for the variable.
    shape_by_conn: bool
        Set to True to infer the shape from the connected output.
    meta_data: dict
        (Optional) Aviary metadata dictionary. If unspecified, the built-in metadata will
        be used.
    shape: tuple
        (Optional) shape for this input.
    """
    meta = meta_data[varname]
    if units:
        input_units = units
    else:
        # units of None are overwritten with defaults. Overwriting units with None is
        # unecessary as it will cause errors down the line if the default is not already
        # None
        input_units = meta['units']
    if desc:
        input_desc = desc
    else:
        input_desc = meta['desc']
    if val is None:
        val = meta['default_value']
    comp.add_input(varname, val=val, units=input_units,
                   desc=input_desc, shape_by_conn=shape_by_conn, shape=shape)


def add_aviary_output(comp, varname, val, units=None, desc=None, shape_by_conn=False,
                      meta_data=_MetaData):
    """
    This function provides a clean way to add variables from the
    variable hierarchy into components as Aviary outputs. It takes
    the standard OpenMDAO inputs of the variable's name, initial
    value, units, and description, as well as the component which
    the variable is being added to.

    Parameters
    ----------
    comp: Component
        OpenMDAO component to add this variable.
    varname: str
        Name of variable.
    val: float or ndarray
        (Optional) Default value for variable. If not specified, the value from metadata
        is used.
    units: str
        (Optional) when speficying val, units should also be specified.
    desc: str
        (Optional) description text for the variable.
    shape_by_conn: bool
        Set to True to infer the shape from the connected output.
    meta_data: dict
        (Optional) Aviary metadata dictionary. If unspecified, the built-in metadata will
        be used.
    """
    meta = meta_data[varname]
    if units:
        output_units = units
    else:
        # units of None are overwritten with defaults. Overwriting units with None is
        # unecessary as it will cause errors down the line if the default is not already
        # None
        output_units = meta['units']
    if desc:
        output_desc = desc
    else:
        output_desc = meta['desc']
    comp.add_output(varname, val=val, units=output_units,
                    desc=output_desc, shape_by_conn=shape_by_conn)


def units_setter(opt_meta, value):
    """
    Check and convert new units tuple into

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

    converted_val = convert_units(new_val, new_units, units)
    return (converted_val, units)


def add_aviary_option(comp, name, val=_unspecified, units=None, desc=None, meta_data=_MetaData):
    """
    Adds an option to an Aviary component. Default values from the metadata are used
    unless a new value is specified.

    Parameters
    ----------
    comp: Component
        OpenMDAO component to add this option.
    name: str
        Name of variable.
    val: float or ndarray
        (Optional) Default value for option. If not specified, the value from metadata
        is used.
    desc: str
        (Optional) description text for the variable.
    units: str
        (Optional) OpenMDAO units string. This can be specified for variables with units.
    meta_data: dict
        (Optional) Aviary metadata dictionary. If unspecified, the built-in metadata will
        be used.
    """
    meta = meta_data[name]
    if not desc:
        desc = meta['desc']
    if val is _unspecified:
        val = meta['default_value']

    if units not in [None, 'unitless']:
        comp.options.declare(name, default=(val, units), types=meta['types'], desc=desc,
                             set_function=units_setter)
    else:
        comp.options.declare(name, default=val, types=meta['types'], desc=desc)


def override_aviary_vars(group, aviary_inputs: AviaryValues,
                         manual_overrides=None, external_overrides=None):
    '''
    This function provides the capability to override output variables
    with variables from the aviary_inputs input. The user can also
    optionally provide the names of variables that they would like to
    override manually. (Manual overriding is simply suppressing the
    promotion of the variable to make way for another output variable
    of the same name, or to create an unconnected input elsewhere.)
    '''
    def name_filter(name):
        return "aircraft:" in name or "mission:" in name

    if not manual_overrides:
        manual_overrides = []

    if not external_overrides:
        external_overrides = []

    # first need to make a list of all the inputs that anyone needs
    # so that we can keep track of any unclaimed inputs
    all_inputs = set()  # use a set to avoid duplicates
    for system in group.system_iter():
        meta = system.get_io_metadata(iotypes=("input",))
        in_var_names = meta.keys()
        for name in in_var_names:
            all_inputs.add(name)

    overridden_outputs = []
    external_overridden_outputs = []
    for comp in group.system_iter(typ=Component):
        # get a list of the variables to use
        out_var_names = list(filter(name_filter, comp.get_io_metadata(
            iotypes=("output",), return_rel_names=False)))
        # get a list of the metadata associated with each variable
        out_var_metadata = comp.get_io_metadata(
            iotypes=("output",), return_rel_names=False)
        in_var_names = filter(name_filter, comp.get_io_metadata(iotypes=("input", )))

        comp_promoted_outputs = []

        for abs_name in out_var_names:
            name = out_var_metadata[abs_name]['prom_name']

            if abs_name in manual_overrides:
                # These are handled outside of this function.
                continue

            elif name in external_overrides:

                # Overridden variables are given a new name
                comp_promoted_outputs.append((name, f"EXTERNAL_OVERRIDE:{name}"))
                external_overridden_outputs.append(name)

                continue  # don't promote it

            elif name in aviary_inputs:
                val, units = aviary_inputs.get_item(name)
                if name in all_inputs:
                    group.set_input_defaults(name, val=val, units=units)

                # Overridden variables are given a new name
                comp_promoted_outputs.append((name, f"AUTO_OVERRIDE:{name}"))
                overridden_outputs.append(name)

                continue  # don't promote it

            # This variable is not overriden, so the output is promoted.
            comp_promoted_outputs.append(name)

        # NOTE Always promoting all inputs into the "global" namespace
        # so its VERY important that we enforce all inputs names exist in the master
        # variable list
        rel_path = comp.pathname[len(group.pathname):].lstrip(".")
        if "." in rel_path:
            # comp is in a subgroup. We must find it.
            sub_path = ".".join(rel_path.split(".")[:-1])
            sub = group._get_subsystem(sub_path)
            sub.promotes(comp.name, inputs=in_var_names, outputs=comp_promoted_outputs)
        else:
            group.promotes(comp.name, inputs=in_var_names, outputs=comp_promoted_outputs)

    if overridden_outputs:
        print("\nThe following variables have been overridden:")
        for prom_name in sorted(overridden_outputs):
            print(f"  '{prom_name}")

    if external_overridden_outputs:
        print("\nThe following variables have been overridden by an external subsystem:")
        for prom_name in sorted(external_overridden_outputs):
            print(f"  '{prom_name}")

    return overridden_outputs


def setup_trajectory_params(
    model: om.Group, traj: dm.Trajectory, aviary_variables: AviaryValues, phases=['climb', 'cruise', 'descent'],
    variables_to_add=None, meta_data=_MetaData, external_parameters={},
):
    """
    This function smoothly sorts through the aviary variables which
    are being used in the trajectory, and for the variables which are
    not options it adds them as a parameter of the trajectory.
    """
    # TODO: variables_to_add is required, so should be an arg, not a kwarg.
    if variables_to_add is None:
        variables_to_add = []

    # Step 1: Initialize a dictionary to hold parameters and their associated phases
    parameters_with_phases = {}

    # Step 2: Loop through external_parameters to populate the dictionary
    for phase_name, parameter_dict in external_parameters.items():
        for key in parameter_dict.keys():
            if key not in parameters_with_phases:
                parameters_with_phases[key] = []
            parameters_with_phases[key].append(phase_name)

    # Step 3: Loop through the collected parameters and call traj.add_parameter
    already_added = []
    for key, phases in parameters_with_phases.items():
        # Assuming the kwargs are the same for shared parameters
        kwargs = external_parameters[phases[0]][key]
        targets = {phase: [key] for phase in phases}
        traj.add_parameter(
            key,
            **kwargs,
            targets=targets
        )

        model.promotes('traj', inputs=[(f'parameters:{key}', key)])
        already_added.append(key)

    # Process the core mission inputs last, because some of them might have already
    # been covered by the phase builders.
    # TODO: As we use more builders, we may reach the point where we don't need
    # to do these anymore.
    for key in sorted(variables_to_add):

        if key in already_added:
            continue

        meta = meta_data[key]

        if not meta['option']:
            val = meta['default_value']
            if val is None:
                val = _unspecified
            units = meta['units']

            if key in aviary_variables:
                try:
                    val = aviary_variables.get_val(key, units)
                except TypeError:
                    val = aviary_variables.get_val(key)

            # TODO temp line to ignore dynamic mission variables, will not work
            #      if names change to 'dynamic:mission:*'
            if ':' not in key:
                continue

            traj.add_parameter(
                key,
                opt=False,
                units=units,
                val=val,
                static_target=True,
                targets={phase_name: [key] for phase_name in phases})

            model.promotes('traj', inputs=[(f'parameters:{key}', key)])

    return traj


def get_units(key, meta_data=None) -> str:
    """
    Returns the units for the specified variable as defined in the MetaData.

    Parameters
    ----------
    key: str
        Name of the variable
    meta_data : dict
        Dictionary containing metadata for the variable. If None, Aviary's built-in
        metadata will be used.
    """
    if meta_data is None:
        meta_data = _MetaData

    return meta_data[key]['units']


def extract_options(aviary_inputs: AviaryValues, metadata=_MetaData) -> dict:
    """
    Extract a dictionary of options from the given aviary_inputs.

    Parameters
    ----------
    aviary_inputs : AviaryValues
        Instance of AviaryValues containing all initial values.
    meta_data : dict
        (Optional) Dictionary of aircraft metadata. Uses Aviary's built-in
        metadata by default.

    Returns
    -------
    dict
        Dictionary of option names and values.
    """
    options = {}
    for key, meta in metadata.items():

        if key not in aviary_inputs:
            continue

        if not meta['option']:
            continue

        val, units = aviary_inputs.get_item(key)
        meta_units = meta['units']

        if meta_units is 'unitless' or meta_units is None:
            options[key] = val

        else:
            # Implement as (quanitity, unit)
            options[key] = (val, units)

    return options
