import getpass
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from openmdao.utils.units import is_compatible, valid_units

from aviary.utils.functions import get_path
from aviary.utils.named_values import NamedValues, get_items, get_keys
from aviary.variable_info.enums import Verbosity


# multiple type annotation uses "typeA | typeB" syntax, but requires Python 3.10+
# filename: (str, Path)
def read_data_file(
    filename,
    metadata=None,
    aliases=None,
    save_comments=False,
    verbosity=Verbosity.BRIEF,
):
    """
    Read data file in Aviary format, which is data delimited by commas with any amount of
    whitespace allowed between data entries. Spaces are not allowed in openMDAO
    variables, so any spaces in header entries are replaced with underscores.

    Parameters
    ----------
    filename : (str, Path)
        filename or filepath of data file to be read
    metadata : dict, optional
        metadata to check validity of variable names provided in data file. Columns with
        variable names that can't be found in metadata will be skipped. If not provided,
        all validly formatted columns are always read.
    aliases : dict, optional
        optional dictionary to define a mapping of variables to allowable aliases in the
        data file header. Keys are variable names, to be used in openMDAO, values
        are a list of headers that correspond to that variable. Alias matching is not
        case-sensitive, and underscores and spaces are treated as equivalent.
    save_comments : bool, optional
        flag if comments in data file should be returned along with data. Defaults to
        False.
    verbosity : (int, Verbosity), optional
        controls level of printouts when running this method. Default is BRIEF (1).

    Returns
    -------
    data : NamedValues
        data read from file in NamedValues format, including variable name, units, and
        values (stored in a numpy array)
    comments : list of str
        any comments from file, with comment characters ('#') stripped out (only if
        save_comments=True)
    """
    verbosity = Verbosity(verbosity)

    filepath = get_path(filename)

    data = NamedValues()
    comments = []

    # prep aliases for case-insensitive matching, with spaces == underscores
    if aliases:
        for key in aliases:
            if isinstance(aliases[key], str):
                aliases[key] = [aliases[key]]
            aliases[key] = [re.sub('\\s', '_', item).lower() for item in aliases[key]]

    with open(filepath, newline=None, encoding='utf-8-sig') as file:
        # csv.reader() and other available packages that can read csv files are not used
        # Manual control of file reading ensures that comments are kept intact and other
        # checks can be performed
        check_for_header = True
        for line_count, line_data in enumerate(file):
            # if comments are present in line, strip them out
            if '#' in line_data:
                index = line_data.index('#')
                comments.append(line_data[index + 1 :].strip())
                line_data = line_data[:index]

            # split by delimiters, remove whitespace and newline characters
            line_data = re.split(r'[;,]\s*', line_data.strip())

            # ignore empty lines
            if not line_data or line_data == ['']:
                continue

            # try to convert line_data to float, skip any blank strings
            try:
                line_data = [float(var) for var in line_data if var != '']
            # data contains things other than floats
            except ValueError:
                # skip checking for header data if not required
                if check_for_header:
                    # dictionary of header name: units
                    header = {}
                    # list of which column goes with each valid header entry
                    valid_indices = []
                    for index in range(len(line_data)):
                        item = re.split('[(]', line_data[index])
                        item = [item[i].strip(') ') for i in range(len(item))]
                        # OpenMDAO vars can't have spaces, convert to underscores
                        name = re.sub('\\s', '_', item[0])
                        if aliases:
                            # "reverse" lookup name in alias dict
                            for key in aliases:
                                if name.lower() in aliases[key]:
                                    name = key
                                    break
                        # 'default' default_units
                        default_units = 'unitless'
                        # if metadata is provided, ensure variable exists and update
                        # default_units
                        if metadata is not None:
                            if name not in metadata.keys():
                                if verbosity > Verbosity.QUIET:  # BRIEF, VERBOSE, DEBUG
                                    warnings.warn(
                                        f'<{filename}: Header <{name}> was not '
                                        'recognized, and will be skipped'
                                    )
                                continue
                            else:
                                default_units = metadata[name]['units']

                        # if units are provided, check that they are valid
                        if len(item) > 1:
                            units = item[-1]
                            if valid_units(item[1]):
                                # check that units are compatible with expected units
                                if metadata is not None:
                                    if not is_compatible(units, default_units):
                                        # Raising error here, as trying to use default
                                        # units could mean accidental conversion which
                                        # would significantly impact analysis
                                        raise ValueError(
                                            f'Provided units of <{units}> '
                                            f'for column <{name}>, which '
                                            'are not compatible with default '
                                            f'units of {default_units}'
                                        )
                            else:
                                # Units were not recognized. Raise error
                                raise ValueError(
                                    f'Invalid units <{units}> provided for '
                                    f'column <{name}> while reading '
                                    f'<{filepath}>.'
                                )
                        else:
                            if metadata is not None and default_units != 'unitless':
                                # units were not provided, but variable should have them
                                # assume default units for that variable
                                if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                                    warning = (
                                        f'Units were not provided for column <{name}> '
                                        f'while reading <{filepath}>. Using default '
                                        f'units of {default_units}.'
                                    )
                                    warnings.warn(warning)
                            units = default_units

                        header[name] = units
                        valid_indices.append(index)

                    if len(header) > 0:
                        check_for_header = False
                        raw_data = {key: [] for key in header.keys()}
                        continue

                # only raise error if not checking for header, or invalid header found
                raise ValueError(
                    f'Non-numerical value found in data file <{filepath}> on line {str(line_count)}'
                )

            # This point is reached when the first valid numerical entry in data file
            # is found. Stop looking for header data from now on
            check_for_header = False

            # pull out data for each valid header, ignore other columns
            for idx, variable in enumerate(header.keys()):
                # valid_indices matches dictionary order, pull data from correct column
                raw_data[variable].append(line_data[valid_indices[idx]])

    # store data in NamedValues object
    for variable in header.keys():
        data.set_val(variable, val=np.array(raw_data[variable]), units=header[variable])

    if save_comments:
        return data, comments
    else:
        return data


# multiple type annotation uses "typeA | typeB" syntax, but requires Python 3.10+
# filename: (str, Path)
# comments: (str, list)
def write_data_file(
    filename,
    data: NamedValues = None,
    comments=[],
    include_timestamp: bool = False,
):
    """
    Write data to a comma-separated values (csv) format file using the Aviary data table
    format.

    Parameters
    ----------
    filename : (str, Path)
        filename or filepath for data file to be written
    data : NamedValues
        NamedValues object containing data that will be written to file, which includes
        variable name, units, and values
    comments : (str, list of str), optional
        optional comments that will be included in the top of the output file, before
        data begins
    include_timestamp : bool, optional
        optional flag to set if timestamp and user should be include in file comments
    """
    if isinstance(filename, str):
        filepath = Path(filename)
    else:
        filepath = filename

    if data is None:
        raise UserWarning(f'No data provided to write to {filepath.name}')

    if type(comments) is str:
        comments = [comments]

    # strip '#' from comments - np.savetxt() will automatically add them
    for idx, line in enumerate(comments):
        if len(line) > 0:
            if line[0] != '#':
                comments[idx] = '# ' + line.strip()

    # if there are comments, add some spacing afterwards - otherwise it should be empty
    if comments:
        comments.append('\n')

    if include_timestamp:
        timestamp = datetime.now().strftime('%m/%d/%y at %H:%M')
        try:
            user = ' by ' + getpass.getuser()
        except Exception:
            user = ''
        stamp = [f'# created {timestamp}{user}\n']
        comments = stamp + comments

    # assemble separate variable name and units information into single list for header
    header = []
    data_dict = {}
    for var, val_and_units in get_items(data):
        units = val_and_units[1]
        formatted_units = ''
        # only explicitly include units if there are any
        if units is not None and units != 'unitless':
            formatted_units = ' (' + units + ')'
        header.append(var + formatted_units)
        data_dict[var] = np.array([str(i) for i in data.get_val(var, units)])

    # set column widths, for more human-readable format
    col_format = []
    for i, key in enumerate(get_keys(data)):
        header_len = len(header[i])
        data_len = len(max(data_dict[key], key=len))
        # min column width is 10 - spaced out columns are visually easier to follow
        # don't pad first column
        if i > 0:
            min_width = 10
        else:
            min_width = 0
        col_len = max(header_len, data_len, min_width)

        # if headers are smaller than column, pad with leading whitespace
        if header_len < col_len:
            header[i] = ' ' * (col_len - header_len) + header[i]

        # special string to define column formatting with specific width
        format = f'%{col_len}s'
        # don't include commas for last column
        if i < len(header) - 1:
            format = format + ', '
        col_format.append(format)

    # convert engine_data from dict to array so it can be written using savetxt
    formatted_data = np.array([data_dict[key] for key in data_dict]).transpose()

    # write to output file w/ header and comments
    np.savetxt(
        filepath,
        formatted_data,
        fmt=''.join(col_format),
        delimiter=',',
        header=', '.join(header),
        comments='\n'.join(comments),
    )
