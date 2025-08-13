#!/usr/bin/python
import argparse
import getpass

from datetime import datetime
from enum import Enum

import numpy as np

from aviary.api import NamedValues
from aviary.utils.conversion_utils import _parse, _read_map, _rep
from aviary.utils.csv_data_file import write_data_file
from aviary.utils.functions import get_path
from aviary.interface.utils import round_it


class PropMapType(Enum):
    GASP = 'GASP'

    def __str__(self):
        return self.value


sig_figs = {
    'Helical Mach': 6,
    'Mach': 5,
    'Power Coefficient': 5,
    'Advance Ratio': 5,
    'Thrust Coefficient': 6,
}

outputs = ['Thrust Coefficient']


def convert_propeller_map(
    input_file, output_file, data_format: PropMapType = PropMapType.GASP, round_data=False
):
    """
    This is a utility class to convert a propeller map file to Aviary format.
    Currently, there is only one option: from GASP format to Aviary format.
    As an Aviary command, the usage is:
    aviary convert_prop_table -f GASP input_file output_file.
    """
    timestamp = datetime.now().strftime('%m/%d/%y at %H:%M')
    user = getpass.getuser()
    comments = []
    data = {}

    data_file = get_path(input_file)

    comments.append(f'# created {timestamp} by {user}')
    comments.append(f'# {data_format}-derived propeller map converted from {data_file.name}')

    if data_format is PropMapType.GASP:
        scalars, tables, fields = _read_gasp_propeller(data_file, comments)

        if scalars['iread'] == 1:
            data['Helical Mach'] = tables['thrust_coefficient'][:, 0]
        else:
            data['Mach'] = tables['thrust_coefficient'][:, 0]
        data['Power Coefficient'] = tables['thrust_coefficient'][:, 1]
        data['Advance Ratio'] = tables['thrust_coefficient'][:, 2]
        data['Thrust Coefficient'] = tables['thrust_coefficient'][:, 3]

        # data needs to be string so column length can be easily found later
        for var in data:
            if round_data:
                data[var] = np.array([round_it(val, sig_figs[var]) for val in data[var]])
            data[var] = np.array([str(item) for item in data[var]])

    else:
        raise UserWarning('Invalid propeller map format provided')

    # store formatted data into NamedValues object
    write_data = NamedValues()
    for key in data:
        write_data.set_val(key, data[key], 'unitless')

    if output_file is None:
        output_file = data_file.stem + '.csv'
    write_data_file(output_file, write_data, outputs, comments, include_timestamp=False)


def _read_gasp_propeller(fp, cmts):
    """Read a GASP propeller map file and parse its scalars and tabular data.
    Data table is returned as a dictionary.
    The table consists of both the independent variables and the dependent variable for
    the corresponding field. The table is a "tidy format" 2D array where the first three
    columns are the independent variables (Advance ratio, Mach number, and power coefficient)
    and the final column is the dependent variable thrust coefficient.
    """
    with open(fp, 'r') as f:
        table_types = [
            'thrust_coefficient',
        ]
        scalars = _read_pm_header(f)
        if scalars['iread'] == 1:
            cmts.append('# CT = f(Helical Mach at 75% Radius, Adv ratio & CP)')
        elif scalars['iread'] == 2:
            cmts.append('# Propfan format - CT = f(Mach, Adv Ratio & CP)')
        else:
            raise RuntimeError(f'IREAD = 1 or 2 expected, got {scalars["iread"]}')

        tables = {k: _read_pm_table(f, cmts) for k in table_types}

    return scalars, tables, table_types


def _read_pm_header(f):
    """Read GASP propeller map header (first line), returning the propeller scalars in a dict
    parameter 1 is Mach type. It is either 1 or 2.
    parameter 2 is IPRINT in GASP and is ignored in Aviary.
    """
    iread, _ = _parse(f, [*_rep(2, (int, 5))])

    return {
        'iread': iread,
    }


def _read_pm_table(f, cmts):
    """Read an entire table from a GASP propeller map file.
    The table data is returned as a "tidy format" array with three columns for the
    independent variables (advanced ratio (J), Mach number and power coefficient)
    and the final column for thrust coefficient.
    """
    tab_data = None
    is_turbo_prop = True

    # table title
    title = f.readline().strip()
    cmts.append(f'# {title}')
    # number of maps in the table
    (nmaps,) = _parse(f, [(int, 5)])
    # blank line
    f.readline()

    for i in range(nmaps):
        map_data = _read_map(f, is_turbo_prop)

        # blank line following all but the last map in the table
        if i < nmaps - 1:
            f.readline()

        if tab_data is None:
            tab_data = map_data
        else:
            tab_data = np.r_[tab_data, map_data]

    return tab_data


def _setup_PMC_parser(parser):
    parser.add_argument('input_file', type=str, help='path to propeller map file to be converted')
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        help='path to file where new converted data will be written (optional)',
    )
    # currently removing as there is only one allowed map type at the moment
    # parser.add_argument(
    #     '-f',
    #     '--data_format',
    #     type=PropMapType,
    #     choices=list(PropMapType),
    #     nargs='?',
    #     default='GASP',
    #     help='data format used by input_file',
    # )
    parser.add_argument('--round', action='store_true', help='round data to improve readability')


def _exec_PMC(args, user_args):
    convert_propeller_map(
        input_file=args.input_file,
        output_file=args.output_file,
        # data_format=args.data_format,
        round_data=args.round,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts GASP-formatted propeller map files into Aviary csv format.\n'
    )
    _setup_PMC_parser(parser)
    args = parser.parse_args()
    _exec_PMC(args, None)
