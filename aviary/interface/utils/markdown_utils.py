from math import floor, log10

import numpy as np

# TODO openMDAO has generate_table() that might be able to replace this

# TODO rounding might have other use cases, move to utils if so
# It is confirmed that both functions are used else where.


def round_it(x, sig=None):
    """
    Round a float to a specified significance.
    If the number is equal to zero, "0" will be returned, regardless of the number of significant digits specified
    If the number is NaN, directly returns it (stays NaN).

    Parameters
    ----------
    x : str or float
        the float that needs to be rounded.
    sig : int
        the number of significant digits to include (If this is unspecified, the number will be rounded to two decimal places).

    Returns
    -------
        The rounded number, or provided string if not convertible to float, or original
        number if it is NaN
    """
    # default sig figs to 2 decimal places out
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x

    if np.isnan(x):
        # return NaNs directly back to markdown report
        return x

    if not sig:
        sig = len(str(round(x))) + 2

    if x != 0:
        return round(x, sig - int(floor(log10(abs(x)))) - 1)
    else:
        return 0


def write_markdown_variable_table(open_file, problem, outputs, metadata):
    """
    Writes a table of the provided variable names in outputs. Converts units to defaults
    from metadata if available.

    Parameters
    ----------
    open_file : Path
        The output file to be written to. This file should have been opened for writing.
    problem : dict
        The dictionary that contains the data.
    outputs : List
        The list of keywords that will go to outputs file.
    metadata : dict
        The dictionary that contains the metadata of the data with desired units.
    """
    open_file.write('\n| Variable Name | Value | Units |\n')
    open_file.write('| :- | :- | :- |\n')
    for var_name in outputs:
        # get default units from metadata
        try:
            units = metadata[var_name]['units']
        except KeyError:
            units = None
        # get value from problem
        try:
            if units:
                val = problem.get_val(var_name, units)
            else:
                # TODO find units for variable in problem?
                val = problem.get_val(var_name)
                units = 'unknown'
        # variable not in problem, get from aviary_inputs instead
        except KeyError:
            try:
                if units:
                    val = problem.aviary_inputs.get_val(var_name, units)
                else:
                    val, units = problem.aviary_inputs.get_item(var_name)
                    if (val, units) == (None, None):
                        raise KeyError
            except KeyError:
                val = 'Not Found in Model'
                units = None
        # handle rounding + formatting
        if isinstance(val, (np.ndarray, list, tuple)):
            val = [round_it(item) for item in val]
            # if an interable with a length of 1, remove bracket/parentheses, etc.
            if len(val) == 1:
                val = val[0]
        else:
            val = round_it(val)
        if not units:
            units = 'unknown'
        if units == 'unitless':
            units = '-'
        summary_line = f'| {var_name} | {val} | {units} |\n'
        open_file.write(summary_line)
