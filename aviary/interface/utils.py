import warnings

import numpy as np
import openmdao.api as om

from aviary.variable_info.enums import Verbosity
from aviary.utils.utils import round_it

# TODO openMDAO has generate_table() that might be able to replace this

# It is confirmed that all three functions are used else where.


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
    outputs : list
        The list of keywords that will go to outputs file.
    metadata : dict
        The dictionary that contains the metadata of the data with desired units.
    """
    open_file.write('\n| Variable Name | Value | Units |\n')
    open_file.write('| :- | :- | :- |\n')
    for var_name in outputs:
        val, units = find_variable_in_problem(var_name, problem, metadata)
        summary_line = f'| {var_name} | {val} | {units} |\n'
        open_file.write(summary_line)


def find_variable_in_problem(var_name, problem, metadata):
    """
    Find the value and units of a variable in an AviaryProblem. Priority is to directly find the
    value via problem.get_val(), followed by searching in problem.aviary_inputs.

    Parameters
    ----------
    var_name : str
        The name of the variable to find.
    problem : AviaryProblem
        The AviaryProblem to search for var_name in.
    metadata : dict
        The dictionary that contains the metadata of the data with desired units. Used to get
        default units if unit information cannot be found.
    """
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
    return val, units


def set_warning_format(verbosity):
    # if verbosity not set / not known yet, default to most simple warning format rather than no
    # warnings at all
    if verbosity is None:
        verbosity = Verbosity.BRIEF

    # Reset all warning filters
    warnings.resetwarnings()

    # NOTE identity comparison is preferred for Enum but here verbosity is often an int, so we need
    # an equality comparison
    if verbosity == Verbosity.QUIET:
        # Suppress all warnings
        warnings.filterwarnings('ignore')

    elif verbosity == Verbosity.BRIEF:

        def simplified_warning(message, category, filename, lineno, line=None):
            return f'Warning: {message}\n\n'

        warnings.formatwarning = simplified_warning
        warnings.simplefilter('ignore', DeprecationWarning)

        # Suppress the specific rhs_checking warning from Dymos internal components
        warnings.filterwarnings(
            action='ignore',
            category=om.SolverWarning,
            message=".*'rhs_checking' is active but no redundant adjoint dependencies were found.*",
        )

        # Suppress the OpenMDAO atomic group warning
        warnings.filterwarnings(
            action='ignore',
            category=om.OpenMDAOWarning,
            message='.*will be treated as atomic for the purposes of determining.*',
        )

    elif verbosity == Verbosity.VERBOSE:

        def simplified_warning(message, category, filename, lineno, line=None):
            return f'{category.__name__}: {message}\n\n'

        warnings.formatwarning = simplified_warning

    else:  # DEBUG
        # use the default warning formatting
        warnings.filterwarnings('default')
