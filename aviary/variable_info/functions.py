from enum import Enum

import dymos as dm
import numpy as np
import openmdao.api as om
from dymos.utils.misc import _unspecified
from openmdao.core.component import Component

from aviary.utils.aviary_options_dict import units_setter
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.utils import cast_type, check_type, enum_setter, wrapped_convert_units
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Settings

# ---------------------------
# Helper functions for setting up inputs/outputs in components
# ---------------------------


def add_aviary_input(
    comp,
    varname,
    val=None,
    units=None,
    desc=None,
    shape_by_conn=False,
    meta_data=_MetaData,
    shape=None,
):
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
        (Optional) when specifying val, units should also be specified.
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
    # units of None are overwritten with defaults. Overwriting units with None is
    # unnecessary as it will cause errors down the line if the default is not already
    # None
    default_units = meta['units']

    if units is None:
        units = default_units

    if desc is None:
        desc = meta['desc']

    if val is None:
        if shape is None:
            val = meta['default_value']
            if val is None:
                val = 0.0
        else:
            val = meta['default_value']
            if val is None:
                val = np.zeros(shape)
            else:
                val = np.ones(shape) * val

        # val was not provided but different units were
        if units != default_units:
            try:
                # convert the default units to requested units
                val = wrapped_convert_units((val, default_units), units)
            except ValueError:
                raise ValueError(
                    f'The requested units {units} for input {varname} in component '
                    f'{comp.name} are invalid.'
                )

    # check types
    val = cast_type(varname, val, meta_data)
    check_type(varname, val, meta_data)

    primal_name = varname.replace(':', '__')

    comp.add_input(
        varname,
        val=val,
        units=units,
        desc=desc,
        shape_by_conn=shape_by_conn,
        shape=shape,
        primal_name=primal_name,
    )


def add_aviary_output(
    comp,
    varname,
    val=None,
    units=None,
    desc=None,
    shape_by_conn=False,
    meta_data=_MetaData,
    shape=None,
):
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
        (Optional) when specifying val, units should also be specified.
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
    # units of None are overwritten with defaults. Overwriting units with None is
    # unnecessary as it will cause errors down the line if the default is not already
    # None
    default_units = meta['units']

    if units is None:
        units = default_units

    if desc is None:
        desc = meta['desc']

    if val is None:
        if shape is None:
            val = meta['default_value']
            if val is None:
                val = 0.0
        else:
            val = meta['default_value']
            if val is None:
                val = np.zeros(shape)
            else:
                val = np.ones(shape) * val

        # val was not provided but different units were
        if units != default_units:
            try:
                # convert the default units to requested units
                val = wrapped_convert_units((val, default_units), units)
            except ValueError:
                raise ValueError(
                    f'The requested units {units} for output {varname} in component '
                    f'{comp.name} are invalid.'
                )

    # check types
    val = cast_type(varname, val, meta_data)
    check_type(varname, val, meta_data)

    primal_name = varname.replace(':', '__')

    comp.add_output(
        varname,
        val=val,
        units=units,
        desc=desc,
        shape_by_conn=shape_by_conn,
        primal_name=primal_name,
    )


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
    # units of None are overwritten with defaults. Overwriting units with None is
    # unnecessary as it will cause errors down the line if the default is not already
    # None
    default_units = meta['units']

    if units is None:
        units = default_units

    if desc is None:
        desc = meta['desc']

    # check units
    if val is _unspecified:
        val = meta['default_value']

        # val was not provided but different units were
        if units != default_units:
            try:
                # convert the default units to requested units
                val = wrapped_convert_units((val, default_units), units)
            except ValueError:
                raise ValueError(
                    f'The requested units {units} for output {name} in component '
                    f'{comp.name} are invalid.'
                )

    # value of None needs to be skipped due to unusual setup of
    # subsystems/mass/flops_based/wing_group.py
    if val is not None:
        # check types
        val = cast_type(name, val, meta_data)
        check_type(name, val, meta_data)

    # Need to gather expected types since sometimes we need to use these to declare the
    # option. Most of the time, types is overwritten to tuple to support (val, units)
    types = meta['types']
    if meta_data[name]['multivalue']:
        if isinstance(types, tuple):
            types = (list, np.ndarray, tuple, *types)
        else:
            types = (list, np.ndarray, tuple, types)

    if units not in [None, 'unitless']:
        types = tuple
        comp.options.declare(
            name,
            default=(val, units),
            types=types,
            desc=desc,
            set_function=units_setter,
        )

    elif isinstance(val, Enum):
        comp.options.declare(name, default=val, types=types, desc=desc, set_function=enum_setter)

    else:
        comp.options.declare(name, default=val, types=types, desc=desc)


def override_aviary_vars(
    group: om.Group,
    aviary_inputs: AviaryValues,
    manual_overrides=None,
    external_overrides=None,
):
    """
    This function provides the capability to override output variables
    with variables from the aviary_inputs input. The user can also
    optionally provide the names of variables that they would like to
    override manually. (Manual overriding is simply suppressing the
    promotion of the variable to make way for another output variable
    of the same name, or to create an unconnected input elsewhere.).
    """

    def name_filter(name):
        return 'aircraft:' in name or 'mission:' in name

    if not manual_overrides:
        manual_overrides = []

    if not external_overrides:
        external_overrides = []

    # first need to make a list of all the inputs that anyone needs
    # so that we can keep track of any unclaimed inputs
    all_inputs = set()  # use a set to avoid duplicates
    for system in group.system_iter():
        meta = system.get_io_metadata(iotypes=('input',))
        in_var_names = meta.keys()
        for name in in_var_names:
            all_inputs.add(name)

    overridden_outputs = []
    external_overridden_outputs = []
    for comp in group.system_iter(typ=Component):
        # get a list of the variables to use
        out_var_names = list(
            filter(
                name_filter,
                comp.get_io_metadata(iotypes=('output',), return_rel_names=False),
            )
        )
        # get a list of the metadata associated with each variable
        out_var_metadata = comp.get_io_metadata(iotypes=('output',), return_rel_names=False)
        in_var_names = filter(name_filter, comp.get_io_metadata(iotypes=('input',)))

        comp_promoted_outputs = []

        for abs_name in out_var_names:
            name = out_var_metadata[abs_name]['prom_name']

            if abs_name in manual_overrides:
                # These are handled outside of this function.
                continue

            elif name in external_overrides:
                # Overridden variables are given a new name
                comp_promoted_outputs.append((name, f'EXTERNAL_OVERRIDE:{name}'))
                external_overridden_outputs.append(name)

                continue  # don't promote it

            elif name in aviary_inputs:
                val, units = aviary_inputs.get_item(name)
                if name in all_inputs:
                    group.set_input_defaults(name, val=val, units=units)

                # Overridden variables are given a new name
                comp_promoted_outputs.append((name, f'AUTO_OVERRIDE:{name}'))
                overridden_outputs.append(name)

                continue  # don't promote it

            # This variable is not overridden, so the output is promoted.
            comp_promoted_outputs.append(name)

        # NOTE Always promoting all inputs into the "global" namespace
        # so its VERY important that we enforce all inputs names exist in the master
        # variable list
        rel_path = comp.pathname[len(group.pathname) :].lstrip('.')
        if '.' in rel_path:
            # comp is in a subgroup. We must find it.
            sub_path = '.'.join(rel_path.split('.')[:-1])
            sub = group._get_subsystem(sub_path)
            sub.promotes(comp.name, inputs=in_var_names, outputs=comp_promoted_outputs)
        else:
            group.promotes(comp.name, inputs=in_var_names, outputs=comp_promoted_outputs)

    if overridden_outputs:
        if aviary_inputs.get_val(Settings.VERBOSITY).value >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
            print('\nThe following variables have been overridden:')
            for prom_name in sorted(overridden_outputs):
                val, units = aviary_inputs.get_item(prom_name)
                print(f"  '{prom_name}  {val}  {units}")

    if external_overridden_outputs:
        if aviary_inputs.get_val(Settings.VERBOSITY).value >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
            print('\nThe following variables have been overridden by an external subsystem:')
            for prom_name in sorted(external_overridden_outputs):
                # do not print values because they will be updated by an external subsystem later.
                print(f"  '{prom_name}")

    return overridden_outputs


def setup_trajectory_params(
    model: om.Group,
    traj: dm.Trajectory,
    aviary_variables: AviaryValues,
    phases=['climb', 'cruise', 'descent'],
    variables_to_add=None,
    meta_data=_MetaData,
    external_parameters={},
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
        traj.add_parameter(key, **kwargs, targets=targets)

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
                targets={phase_name: [key] for phase_name in phases},
            )

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

        if meta_units == 'unitless' or meta_units is None:
            options[key] = val

        else:
            # Implement as (quantity, unit)
            options[key] = (val, units)

    return options


def setup_model_options(
    prob: om.Problem,
    aviary_inputs: AviaryValues,
    meta_data=_MetaData,
    engine_models=None,
    prefix='',
):
    """
    Setup the correct model options for an aviary problem.

    Parameters
    ----------
    prob: Problem
        OpenMDAO problem prior to setup.
    aviary_inputs : AviaryValues
        Instance of AviaryValues containing all initial values.
    meta_data : dict
        (Optional) Dictionary of aircraft metadata. Uses Aviary's built-in
        metadata by default.
    engine_models : List of EngineModels or None
        (Optional) Engine models
    prefix : str
        Prefix for model options. Used for multi-mission.
    """
    # Use OpenMDAO's model options to pass all options through the system hierarchy.
    prob.model_options[f'{prefix}*'] = extract_options(aviary_inputs, meta_data)

    # Multi-engines need to index into their options.
    try:
        num_engine_models = len(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES))
    except KeyError:
        # No engine data.
        return

    if num_engine_models > 1:
        if engine_models is None:
            engine_models = prob.engine_builders

        for idx in range(num_engine_models):
            eng_name = engine_models[idx].name

            # TODO: For future flexibility, need to tag the required engine options.
            opt_names = [
                Aircraft.Engine.SCALE_PERFORMANCE,
                Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER,
                Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER,
                Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM,
                Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM,
            ]
            opt_names_units = [
                Aircraft.Engine.REFERENCE_SLS_THRUST,
                Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION,
            ]
            opts = {}
            for key in opt_names:
                opts[key] = aviary_inputs.get_item(key)[0][idx]
            for key in opt_names_units:
                val, units = aviary_inputs.get_item(key)
                opts[key] = (val[idx], units)

            path = f'{prefix}*core_propulsion.{eng_name}*'
            prob.model_options[path] = opts
