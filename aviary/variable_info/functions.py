import dymos as dm
import openmdao.api as om
from dymos.utils.misc import _unspecified
from openmdao.core.component import Component

from aviary.utils.aviary_values import AviaryValues

import dymos as dm

from dymos.utils.misc import _unspecified

import openmdao.api as om

from aviary.variable_info.core_promotes import core_mission_inputs
from aviary.variable_info.variable_meta_data import _MetaData

# ---------------------------
# Helper functions for setting up inputs/outputs in components
# ---------------------------


def add_aviary_input(comp, varname, val=None, units=None, desc=None, shape_by_conn=False, meta_data=_MetaData):
    '''
    This function provides a clean way to add variables from the
    variable hierarchy into components as Aviary inputs. It takes
    the standard OpenMDAO inputs of the variable's name, initial
    value, units, and description, as well as the component which
    the variable is being added to.
    '''
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
                   desc=input_desc, shape_by_conn=shape_by_conn)


def add_aviary_output(comp, varname, val, units=None, desc=None, shape_by_conn=False, meta_data=_MetaData):
    '''
    This function provides a clean way to add variables from the
    variable hierarchy into components as Aviary outputs. It takes
    the standard OpenMDAO inputs of the variable's name, initial
    value, units, and description, as well as the component which
    the variable is being added to.
    '''
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
                if name in all_inputs:
                    val, units = aviary_inputs.get_item(name)
                    group.set_input_defaults(name, val=val, units=units)

                    # Overridden variables are given a new name
                    comp_promoted_outputs.append((name, f"AUTO_OVERRIDE:{name}"))
                    overridden_outputs.append(name)

                continue  # don't promote it

            # This variable is not overriden, so the output is promoted.
            comp_promoted_outputs.append(name)

        # note: Always promoting all inputs into the "global" namespace
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
    '''
    This function smoothly sorts through the aviary variables which
    are being used in the trajectory, and for the variables which are
    not options it adds them as a parameter of the trajectory.
    '''
    # TODO: make this automated as part of the trajectory subclass
    if variables_to_add is None:
        variables_to_add = sorted(core_mission_inputs)

    for key in sorted(variables_to_add):
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

            # TODO: make a trajectory subclass that does this
            traj.add_parameter(
                key,
                opt=False,
                units=units,
                val=val,
                static_target=True,
                targets={phase_name: [key] for phase_name in phases})

            model.promotes('traj', inputs=[(f'parameters:{key}', key)])

    # Step 1: Initialize a dictionary to hold parameters and their associated phases
    parameters_with_phases = {}

    # Step 2: Loop through external_parameters to populate the dictionary
    for phase_name, parameter_dict in external_parameters.items():
        for key in parameter_dict.keys():
            if key not in parameters_with_phases:
                parameters_with_phases[key] = []
            parameters_with_phases[key].append(phase_name)

    # Step 3: Loop through the collected parameters and call traj.add_parameter
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
