import numpy as np
from math import floor, log10

# TODO openMDAO has generate_table() that can eventually replace this

# TODO this might have other use cases, move to utils if so


def round_it(x, sig=None):
    # default sig figs to 2 decimal places out
    if not sig:
        sig = len(str(round(x)))+2
    if x != 0:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    else:
        return 0


def write_markdown_variable_table(open_file, problem, outputs, metadata):
    """
    Writes a table of the provided variable names in outputs. Converts units to defaults
    from metadata if avaliable.
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
            except KeyError:
                val = 'Not Found in Model'
                units = None
        # handle rounding + formatting
        if isinstance(val, (np.ndarray, list, tuple)):
            val = [round_it(item) for item in val]
            # if an interable with a length of 1, remove bracket/paretheses, etc.
            if len(val) == 1:
                val = val[0]
        else:
            round_it(val)
        if not units:
            units = 'unknown'
        summary_line = f'| {var_name} | {val} |'
        if units != 'unitless':
            summary_line = summary_line + f' {units}'
        summary_line = summary_line + ' |\n'
        open_file.write(summary_line)
