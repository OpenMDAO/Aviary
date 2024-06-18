#!/usr/bin/python

import argparse
import getpass
import itertools
import numpy as np
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

from aviary.api import NamedValues
from aviary.utils.csv_data_file import write_data_file
from aviary.utils.functions import get_path
from aviary.subsystems.propulsion.utils import default_units


class PropMapType(Enum):
    GASP = 'GASP'

    def __str__(self):
        return self.value


class PropModelVariables(Enum):
    '''
    Define constants that map to supported variable names in an propeller model.
    '''
    MACH = auto()
    CP = auto()  # power coefficient
    CT = auto()  # thrust coefficient
    J = auto()  # advanced ratio


default_units = {
    PropModelVariables.MACH: 'unitless',
    PropModelVariables.CP: 'unitless',
    PropModelVariables.CT: 'unitless',
    PropModelVariables.J: 'unitless',
}

MACH = PropModelVariables.MACH
CP = PropModelVariables.CP
CT = PropModelVariables.CT
J = PropModelVariables.J

gasp_keys = [MACH, CP, CT, J]

header_names = {
    MACH: 'Mach_Number',
    CP: 'CP',
    CT: 'CT',
    J: 'J',
}


def PropDataConverter(input_file, output_file, data_format: PropMapType):
    timestamp = datetime.now().strftime('%m/%d/%y at %H:%M')
    user = getpass.getuser()
    comments = []
    header = {}
    data = {}

    data_file = get_path(input_file)

    comments.append(f'# created {timestamp} by {user}')

    if data_format is PropMapType.GASP:
        scalars, tables, fields = _read_gasp_propeller(data_file, comments)

        data[J] = tables['thrust_coefficient'][:, 2]
        data[MACH] = tables['thrust_coefficient'][:, 0]
        data[CP] = tables['thrust_coefficient'][:, 1]
        data[CT] = tables['thrust_coefficient'][:, 3]

        # define header now that we know what is in the propeller map
        header = {key: default_units[key] for key in gasp_keys}

        # data needs to be string so column length can be easily found later
        for var in data:
            data[var] = np.array([str(item) for item in data[var]])

    else:
        quit("Invalid propeller map format provided")

    # store formatted data into NamedValues object
    write_data = NamedValues()
    for idx, key in enumerate(data):
        write_data.set_val(header_names[key], data[key], default_units[key])

    write_data_file(output_file, write_data, comments, include_timestamp=False)


def _read_gasp_propeller(fp, cmts):
    """Read a GASP propeller map file and parse its scalars and tabular data.
    Data table is returned as a dictionary.
    The table consists of both the independent variables and the dependent variable for
    the corresponding field. The table is a "tidy format" 2D array where the first three
    columns are the independent varaiables (Advance ratio, Mach number, and power coefficient)
    and the final column is the dependent variable thrust coefficient.
    """
    with open(fp, "r") as f:
        table_types = ["thrust_coefficient",]
        scalars = _read_pm_header(f)
        if scalars['iread'] == 1:
            cmts.append('# CT = f(Helical Mach at 75% Radius, Adv ratio & CP)')
        elif scalars['iread'] == 2:
            cmts.append('Propfan format - CT = f(Mach, Adv Ratio & CP)')
        else:
            raise RuntimeError(f"IREAD = 1 or 2 expected, got {scalars['iread']}")

        tables = {k: _read_pm_table(f, cmts) for k in table_types}

    return scalars, tables, table_types


def _read_pm_header(f):
    """Read GASP propeller map header, returning the propeller scalars in a dict"""
    # parameter 2 is IPRINT in GASP and is ignored
    iread, _ = _parse(
        f, [*_rep(2, (int, 5))]
    )

    return {
        "iread": iread,
    }


def _read_pm_table(f, cmts):
    """Read an entire table from a GASP propeller map file.
    The table data is returned as a "tidy format" array with three columns for the
    independent variables (altitude, T4/T2, and Mach number) and the final column for
    the table field (one of thrust, fuelflow, or airflow for TurboFans or 
    shaft_power_corrected, fuelflow, or tailpipe_thrust for TurboProps).
    """
    tab_data = None

    # table title
    title = f.readline().strip()
    cmts.append(f'# {title}')
    # number of maps in the table
    (nmaps,) = _parse(f, [(int, 5)])
    # blank line
    f.readline()

    for i in range(nmaps):
        map_data = _read_map(f)

        # blank line following all but the last map in the table
        if i < nmaps - 1:
            f.readline()

        if tab_data is None:
            tab_data = map_data
        else:
            tab_data = np.r_[tab_data, map_data]

    return tab_data


def _rep(n, t):
    """Shorthand for ``itertools.repeat`` with the multiplier first."""
    return itertools.repeat(t, n)


def _parse(f, fmt):
    """Read a line from file ``f`` and parse it according to the given ``fmt``"""
    return _strparse(f.readline(), fmt)


def _strparse(s, fmt):
    """Parse a string into fixed-width numeric fields.
    ``fmt`` should be a list of tuples specifying (type, length) for each field in
    string ``s``. Use None for the type to skip (i.e. not yield) that field.
    """
    p = 0
    for typ, length in fmt:
        sub = s[p: p + length]
        if typ is not None:
            yield typ(sub)
        p += length


def _read_map(f):
    """Read a single map of a table from the propeller map file.
    The map data is returned in the same format as in ``read_table``.
    """
    # map dimensions: FORMAT(/2I5,F10.1,10X))
    npts, nline, amap = _parse(f, [*_rep(2, (int, 5)), (float, 10)])

    map_data = np.empty((npts * nline, 4))
    map_data[:, 0] = amap

    # number of points on a single line - wrapped if more than 6
    max_columns = 6

    # point vals: FORMAT(10X,6F10.4,10X)
    x = []
    npts_remaining = npts
    while npts_remaining > 0:
        npts_to_read = min(max_columns, npts_remaining)
        # remaining vals on wrapped line
        x.extend(list(_parse(f, [(None, 10), *_rep(npts_to_read, (float, 10))])))
        npts_remaining -= npts_to_read

    map_data[:, 2] = np.tile(x, nline)

    for j in range(nline):
        npts_remaining = npts
        npts_to_read = min(max_columns, npts_remaining)
        # line (y) val then z vals: FORMAT(F10.4,6F10.1,10X,/(6F10.1,10X))
        vals = list(_parse(f, [(float, 10), *_rep(npts_to_read, (float, 10))]))
        y = vals[0]
        z = vals[1:]
        npts_remaining -= npts_to_read
        while npts_remaining > 0:
            npts_to_read = min(max_columns, npts_remaining)
            # add remaining vals on warapped line
            line_format = [*_rep(npts_to_read, (float, 10))]
            line_format = [(None, 10), *line_format]
            z.extend(list(_parse(f, line_format)))
            npts_remaining -= npts_to_read

        sl = slice(j * npts, (j + 1) * npts)
        map_data[sl, 1] = y
        map_data[sl, 3] = z

    return map_data


def _setup_PMC_parser(parser):
    parser.add_argument('input_file', type=str,
                        help='path to propeller map file to be converted')
    parser.add_argument('output_file', type=str,
                        help='path to file where new converted data will be written')
    parser.add_argument('-f', '--data_format', type=PropMapType, choices=list(PropMapType),
                        help='data format used by input_file')


def _exec_PMC(args, user_args):
    PropDataConverter(
        input_file=args.input_file,
        output_file=args.output_file,
        data_format=args.data_format
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts GASP-formatted '
                                     'propeller map files into Aviary csv format.\n')
    _setup_PMC_parser(parser)
    args = parser.parse_args()
    _exec_PMC(args, None)
