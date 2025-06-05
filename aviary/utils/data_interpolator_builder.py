from pathlib import Path

import numpy as np
import openmdao.api as om

from aviary.utils.csv_data_file import read_data_file
from aviary.utils.functions import get_path
from aviary.utils.named_values import NamedValues, get_items, get_keys


def build_data_interpolator(
    num_nodes,
    interpolator_data=None,
    interpolator_outputs=None,
    method='slinear',
    extrapolate=True,
    structured=None,
    connect_training_data=False,
):
    """
    Builder for openMDAO metamodel components using data provided via data file, directly
    provided as an argument, or training data passed through openMDAO connections.
    If using a structured grid, data can either be converted from a semistructured
    grid format, or directly provided in structured grid format.

    Parameters
    ----------
    num_nodes : int
        Number of points that will be simultaneously interpolated during model execution.

    interpolator_data : (str, Path, NamedValues)
        Path to the Aviary csv file containing all data required for interpolation, or
        the data directly given as a NamedValues object.

    interpolator_outputs : dict
        Dictionary describing the names of dependent variables (keys) and their
        units (values). If connect_training_data is False, these variable names must reference
        variables in data_file or interpolator_data. If connect_training_data is True, then
        this dictionary describes the names and units for training data that will be
        provided via openMDAO connections during model execution.

    method : str, optional
        Interpolation method for metamodel. See openMDAO documentation for valid
        options.

    extrapolate : bool, optional
        Flag that sets if metamodel should allow extrapolation

    structured : bool, optional
        Flag to set if interpolation data is a structure grid. If True, the
        structured metamodel component is used, if False, the semistructured metamodel is
        used. If None, the builder chooses based on provided data structure.

    connect_training_data : bool, optional
        Flag that sets if dependent data for interpolation will be passed via openMDAO
        connections. If True, any provided values for dependent variables will
        be ignored.

    Returns
    -------
    interp_comp : om.MetaModelSemiStructuredComp, om.MetaModelStructuredComp
        OpenMDAO metamodel component using the provided data and flags
    """
    # Argument checking #
    if interpolator_outputs is None:
        raise UserWarning('Independent variables for interpolation were not provided.')
    # if interpolator data is a filepath, get data from file
    if isinstance(interpolator_data, str):
        interpolator_data = get_path(interpolator_data)
    if isinstance(interpolator_data, Path):
        interpolator_data = read_data_file(interpolator_data)

    # Pre-format data: Independent variables placed before dependent variables - position
    #                  of these variables relative to others of their type is preserved
    #                  All data converted to numpy arrays
    indep_vars = NamedValues()
    dep_vars = NamedValues()
    for key, (val, units) in get_items(interpolator_data):
        if not isinstance(val, np.ndarray):
            val = np.array(val)
        if key in interpolator_outputs:
            dep_vars.set_val(key, val, units)
        else:
            indep_vars.set_val(key, val, units)
    # update interpolator_data with correctly ordered indep/dep vars in numpy arrays
    interpolator_data.update(indep_vars)
    for key, (val, units) in get_items(dep_vars):
        interpolator_data.set_val(key, val, units)

    # TODO investigate creating structured grid from semistructured grid via extrapolation

    # is data already in structured format?
    # assume data is structured until proven otherwise
    data_pre_structured = True
    shape = []
    # check inputs, should be vector of unique values only
    for key, (val, units) in get_items(interpolator_data):
        if len(val.shape) == 1:
            if key not in interpolator_outputs:
                # try:
                if np.array_equal(np.unique(val), val):
                    # if vector is only unique values, could be structured!
                    # Store shape and keep going
                    shape.append(len(np.unique(val)))
                else:
                    # Data is not structured. Stop looping through inputs
                    data_pre_structured = False
                    break

    # check outputs, should be array matching shape of input vector lengths
    # if we already know data needs formatting, don't bother checking outputs
    if data_pre_structured:
        for key in interpolator_outputs:
            (val, units) = interpolator_data.get_item(key)
            if np.shape(val) != tuple(shape):
                if len(np.shape(val)) > 1:
                    # we assume user was *trying* to set up a structured grid
                    # if output is multi-dimensional array. If output is 1d it could
                    # be a structured grid with one input, or a semistructured grid
                    raise ValueError(
                        f'shape of output <{key}>, {np.shape(val)}, does '
                        f'not match expected shape {tuple(shape)}'
                    )
                else:
                    # We don't know if data is structured or not if 1d. No harm
                    # in sorting and "reformatting", so assume it needs to be converted
                    data_pre_structured = False
                    break

    if structured is None and data_pre_structured:
        # If the data is already structured, just use a structured grid - it's faster
        # with no downsides
        structured = True
    elif structured is None:
        # In case structured is still None, set it to False - we know data is unstructured
        structured = False

    if not connect_training_data:
        # Sort and format data. Only if not using training data - since we have control
        # over both input and output data they are guaranteed to match after reformatting

        # sort data in semistructured grid format
        # always sort unless data is in structured format
        if not data_pre_structured:
            # first check that data are all vectors of the same length
            for idx, item in enumerate(get_items(interpolator_data)):
                key = item[0]
                units = item[1][1]
                if idx != 0:
                    prev_model_length = model_length
                else:
                    prev_model_length = len(interpolator_data.get_val(key, units))
                model_length = len(interpolator_data.get_val(key, units))
                if model_length != prev_model_length:
                    raise IndexError('Lengths of data provided for interpolation do not match.')

            # get data into column array format
            sorted_values = np.array(
                [val for (key, (val, units)) in get_items(interpolator_data)]
            ).transpose()

            # get all the independent values in format needed for sorting
            independent_vals = np.array([val for (key, (val, units)) in get_items(indep_vars)])

            # Sort by dependent variables in priority order of their appearance
            sorted_values = sorted_values[np.lexsort(np.flip(independent_vals, 0))]

            # reset interpolator_data with sorted values
            for idx, (var, (val, units)) in enumerate(get_items(interpolator_data)):
                interpolator_data.set_val(var, sorted_values[:, idx], units)

        # If user wants structured data, but provided data is not formatted correctly,
        # convert it!
        if structured and not data_pre_structured:
            # Use assumptions for structured grid to format data
            # Only need to reformat data when not using training data, user is responsible
            # for formatting in that case
            # Assumes independent variables are first columns

            (length, var_count) = np.shape(sorted_values)
            indep_var_count = np.shape(independent_vals)[0]

            structured_data = []
            # only need unique independent variables
            unique_data = []
            for i in range(indep_var_count):
                unique_data.append(np.unique(sorted_values[:, i]))
                structured_data.append(unique_data[i])

            shape = tuple([np.size(unique_data[i]) for i in range(indep_var_count)])

            # output data needs to be in nd array format
            for i in range(indep_var_count, var_count):
                structured_data.append(np.reshape(sorted_values[:, i], shape))

            # reset interpolator_data with structured grid formatted values
            for idx, (var, (val, units)) in enumerate(get_items(interpolator_data)):
                interpolator_data.set_val(var, structured_data[idx], units)

    if connect_training_data and structured and not data_pre_structured:
        # User has asked for structured data but not provided it. Use of training data
        # means we can't do any processing on the data including ensuring sorted order,
        # since that might misalign inputs with future connections we can't control here
        # Just convert inputs to structure grid format
        for key in get_keys(indep_vars):
            (val, units) = interpolator_data.get_item(key)
            # take unique values only, put back into interpolator_data
            val = np.unique(val)
            interpolator_data.set_val(key, val, units)

    # create interpolation component
    if structured:
        interp_comp = om.MetaModelStructuredComp(
            method=method,
            extrapolate=extrapolate,
            vec_size=num_nodes,
            training_data_gradients=connect_training_data,
        )
    else:
        interp_comp = om.MetaModelSemiStructuredComp(
            method=method,
            extrapolate=extrapolate,
            vec_size=num_nodes,
            training_data_gradients=connect_training_data,
        )

    # add interpolator inputs
    for key in get_keys(indep_vars):
        values, units = interpolator_data.get_item(key)
        interp_comp.add_input(key, training_data=values, units=units)
    # add interpolator outputs
    for key in interpolator_outputs:
        if key in interpolator_data:
            values, units = interpolator_data.get_item(key)
        if connect_training_data:
            units = interpolator_outputs[key]
            interp_comp.add_output(key, units=units)
        else:
            interp_comp.add_output(key, training_data=values, units=units)

    return interp_comp
