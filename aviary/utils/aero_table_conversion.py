#!/usr/bin/python

import argparse
import re
from enum import Enum
from pathlib import Path
from scipy.interpolate import interp1d

import numpy as np
from openmdao.components.interp_util.interp import InterpND

from aviary.api import NamedValues
from aviary.interface.utils import round_it
from aviary.utils.conversion_utils import _parse, _read_map, _rep
from aviary.utils.csv_data_file import write_data_file
from aviary.utils.functions import get_path


class CodeOrigin(Enum):
    FLOPS = 'FLOPS'
    GASP = 'GASP'
    GASP_ALT = 'GASP_ALT'


_gasp_keys = ['Altitude', 'Mach', 'Angle of Attack']
default_units = {
    'Altitude': 'ft',
    'Mach': 'unitless',
    'Angle of Attack': 'deg',
    'CL': 'unitless',
    'CD': 'unitless',
}


allowed_headers = {
    'altitude': 'Altitude',
    'alt': 'Altitude',
    'alpha': 'Angle of Attack',
    'mach': 'Mach',
    'delflp': 'Flap Deflection',
    'cltot': 'CL',
    'cl': 'CL',
    'cd': 'CD',
    'hob': 'Hob',
    'del_cl': 'Delta CL',
    'del_cd': 'Delta CD',
}

outputs = ['CL', 'CL', 'CD', 'Hob', 'Delta CL', 'Delta CD']

# number of sig figs to round each header to, if requested
sig_figs = {
    'Altitude': 7,
    'Mach': 4,
    'Angle of Attack': 4,
    'CL': 4,
    'CD': 4,
}


def convert_aero_table(input_file=None, output_file=None, data_format=None):
    """This is a utility class to convert a legacy aero data file to Aviary format.
    There are two options for the legacy aero data file format: FLOPS and GASP.
    As an Aviary command, the usage is:
    aviary convert_aero_table -F {FLOPS|GASP|GASP_ALT} input_file output_file.
    Note: In case of GASP, reading of a possible cd0 table is not implemented yet.
    """
    data_format = CodeOrigin(data_format)
    data_file = get_path(input_file)
    if isinstance(output_file, str):
        output_file = Path(output_file)
    elif isinstance(output_file, list):
        for ii, file in enumerate(output_file):
            output_file[ii] = Path(file)
    if not output_file:
        if data_format in (CodeOrigin.GASP, CodeOrigin.GASP_ALT):
            # Default output file name is same location and name as input file, with
            # '_aviary' appended to filename
            path = data_file.parents[0]
            name = data_file.stem
            output_file = path / (name + '_aviary.csv')
        elif data_format is CodeOrigin.FLOPS:
            # Default output file name is same location and name as input file, with
            # '_aviary' appended to filename
            path = data_file.parents[0]
            name = data_file.stem
            suffix = data_file.suffix
            file1 = path / name + '_aviary_CDI' + suffix
            file2 = path / name + '_aviary_CD0' + suffix
            output_file = [file1, file2]

    stamp = f'# {data_format.value}-derived aerodynamics data converted from {data_file.name}'

    if data_format is CodeOrigin.GASP_ALT:
        data, comments = _load_gasp_aero_table(data_file)
        comments = [stamp] + comments
        write_data_file(output_file, data, outputs, comments, include_timestamp=True)
    elif data_format is CodeOrigin.GASP:
        data = {key: [] for key in _gasp_keys}
        scalars, tables, fields = _load_gasp_alt_aero_table(data_file)
        # save scalars as comments
        comments = []
        comments.extend(['# ' + key + ': ' + str(scalars[key]) for key in scalars.keys()])
        structured_data = _make_structured_grid(tables, method='lagrange3', fields=fields)
        data['Altitude'] = structured_data['CL']['alts']
        data['Mach'] = structured_data['CL']['machs']
        data['Angle of Attack'] = structured_data['CL']['aoas']
        data['CL'] = structured_data['CL']['vals']
        data['CD'] = structured_data['CD']['vals']

        # round data if requested, using sig_figs as guide
        round_data = True
        if round_data:
            for key in data:
                data[key] = np.array([round_it(val, sig_figs[key]) for val in data[key]])

        # data needs to be string so column length can be easily found later
        for var in data:
            data[var] = np.array([str(item) for item in data[var]])

        # sort data
        # create parallel dict to data that stores floats
        formatted_data = {}
        for key in data:
            formatted_data[key] = data[key].astype(float)

        # convert engine_data from dict to list so it can be sorted
        sorted_values = np.array(list(formatted_data.values())).transpose()

        # Sort by altitude, then mach, then angle of attack
        sorted_values = sorted_values[
            np.lexsort(
                [
                    formatted_data['Angle of Attack'],
                    formatted_data['Mach'],
                    formatted_data['Altitude'],
                ]
            )
        ]
        for idx, key in enumerate(formatted_data):
            formatted_data[key] = sorted_values[:, idx]

        # store formatted data into NamedValues object
        write_data = NamedValues()

        header_names = {
            'Altitude': 'Altitude',
            'Mach': 'Mach',
            'Angle of Attack': 'Angle of Attack',
            'CL': 'CL',
            'CD': 'CD',
        }
        for key in data:
            write_data.set_val(header_names[key], formatted_data[key], default_units[key])

        comments = [stamp] + comments
        write_data_file(output_file, write_data, outputs, comments, include_timestamp=True)
    elif data_format is CodeOrigin.FLOPS:
        if type(output_file) is not list:
            # if only one filename is given, split into two
            path = output_file.parents[0]
            name = output_file.stem
            suffix = output_file.suffix
            file1 = path / (name + '_CDi' + suffix)
            file2 = path / (name + '_CD0' + suffix)
            output_file = [file1, file2]

        lift_drag_data, lift_drag_comments, zero_lift_drag_data, zero_lift_drag_comments = (
            _load_flops_aero_table(data_file)
        )

        # write lift-dependent drag file
        lift_drag_comments = [stamp] + lift_drag_comments
        write_data_file(output_file[0], lift_drag_data, lift_drag_comments, include_timestamp=True)

        # write zero-lift drag file
        zero_lift_drag_comments = [stamp] + zero_lift_drag_comments
        write_data_file(
            output_file[1], zero_lift_drag_data, zero_lift_drag_comments, include_timestamp=True
        )


def _load_flops_aero_table(filepath: Path):
    """Load an aero table in FLOPS format."""

    def _read_line(line_count, comments):
        line = file_contents[line_count].strip()
        items = re.split(r'[\s]*\s', line)
        if items[0] == '#':
            comments.append(line)
            nonlocal offset
            offset += 1
            try:
                items = _read_line(line_count + 1, comments)
            except IndexError:
                return
        else:
            # try to convert line to float
            try:
                items = [float(var) for var in items]
            # data contains things other than floats
            except ValueError:
                raise ValueError(
                    f'Non-numerical value found in data file <{filepath.name}> on '
                    f'line {str(line_count)}'
                )

        return items

    lift_drag = []
    lift_drag_data = NamedValues()
    lift_drag_comments = []
    zero_lift_drag = []
    zero_lift_drag_data = NamedValues()
    zero_lift_drag_comments = []

    file_contents = []
    with open(filepath, 'r') as reader:
        for line in reader:
            file_contents.append(line)

    offset = 0
    # these are not needed, we can determine the length of data vectors directly
    lift_drag_mach_count, cl_count = _read_line(0 + offset, lift_drag_comments)
    lift_drag_machs = _read_line(1 + offset, lift_drag_comments)
    cls = _read_line(2 + offset, lift_drag_comments)
    lift_drag = []
    for i in range(len(lift_drag_machs)):
        drag = _read_line(3 + i + offset, lift_drag_comments)
        if len(drag) == len(cls):
            lift_drag.append(drag)
        else:
            raise ValueError(
                'Number of data points provided for '
                f'lift-dependent drag at Mach {lift_drag_machs[i]} '
                'does not match number of CLs provided '
                f'(FLOPS aero data file {filepath.name})'
            )
    if len(lift_drag) != len(lift_drag_machs):
        raise ValueError(
            'Number of data rows provided for lift-dependent drag does '
            'not match number of Mach numbers provided (FLOPS aero data '
            f'file {filepath.name})'
        )
    offset = offset + i
    # these are not needed, we can determine the length of data vectors directly
    altitude_count, zero_lift_mach_count = _read_line(4 + offset, zero_lift_drag_comments)
    zero_lift_machs = _read_line(5 + offset, zero_lift_drag_comments)
    altitudes = _read_line(6 + offset, zero_lift_drag_comments)
    for i in range(len(zero_lift_machs)):
        drag = _read_line(7 + i + offset, zero_lift_drag_comments)
        if len(drag) == len(altitudes):
            zero_lift_drag.append(drag)
        else:
            raise ValueError(
                'Number of data points provided for '
                f'zero-lift drag at Mach {zero_lift_machs[i]} '
                'does not match number of Altitudes provided '
                f'(FLOPS aero data file {filepath.name})'
            )
    if len(zero_lift_drag) != len(zero_lift_machs):
        raise ValueError(
            'Number of data rows provided for zero-lift drag does '
            'not match number of Mach numbers provided (FLOPS aero data '
            f'file {filepath.name})'
        )

    cl, mach = np.meshgrid(cls, lift_drag_machs)
    lift_drag_data.set_val('Mach', mach.flatten(), 'unitless')
    lift_drag_data.set_val('Lift Coefficient', cl.flatten(), 'unitless')
    lift_drag_data.set_val(
        'Lift-Dependent Drag Coefficient', np.array(lift_drag).flatten(), 'unitless'
    )

    altitude, mach = np.meshgrid(altitudes, zero_lift_machs)
    zero_lift_drag_data.set_val('Altitude', altitude.flatten(), 'ft')
    zero_lift_drag_data.set_val('Mach', mach.flatten(), 'unitless')
    zero_lift_drag_data.set_val(
        'Zero-Lift Drag Coefficient', np.array(zero_lift_drag).flatten(), 'unitless'
    )

    return lift_drag_data, lift_drag_comments, zero_lift_drag_data, zero_lift_drag_comments


def _load_gasp_aero_table(filepath: Path):
    """Load an aero table in GASP format."""
    data = NamedValues()
    raw_data = []
    comments = []
    variables = []
    units = []
    read_header = True
    read_units = False
    with open(filepath, 'r') as reader:
        for line_count, line in enumerate(reader):
            # ignore empty lines
            if not line or line.strip() == ['']:
                continue
            if line[0] == '#':
                line = line.strip('# ').strip()
                if read_header:
                    items = re.split(r'[\s]*\s', line)
                    if all(name.lower() in allowed_headers for name in items):
                        variables = [name for name in items]
                        read_header = False
                        read_units = True
                    else:
                        if line:
                            comments.append(line)
                else:
                    if read_units:
                        items = re.split(r'[\s]*\s', line)
                        for item in items:
                            item = item.strip('()')
                            if item == '-':
                                item = 'unitless'
                            units.append(item)
                continue
            else:
                # this is data
                items = re.split(r'[\s]*\s', line.strip())
                # try to convert line to float
                try:
                    line_data = [float(var) for var in items]
                # data contains things other than floats
                except ValueError:
                    raise ValueError(
                        f'Non-numerical value found in data file <{filepath.name}> on '
                        f'line {str(line_count)}'
                    )
                else:
                    raw_data.append(line_data)

    raw_data = np.array(raw_data)

    # translate raw data into NamedValues object
    for idx, var in enumerate(variables):
        if var.lower() in allowed_headers:
            data.set_val(allowed_headers[var.lower()], raw_data[:, idx], units[idx])

    return data, comments


def _load_gasp_alt_aero_table(filepath: Path):
    with open(filepath, 'r') as f:
        fields = ['CL', 'CD']
        scalars = _read_header(f)
        tables = {k: _read_table(f) for k in fields}

    # Now, tables['CD'] is a function of altitude, Mach, and CL.
    # For Aviary table, we need to convert it to a function of Altitude, Mach and alpha.
    cl_table = tables['CL']
    cd_table = tables['CD']
    n = len(cl_table)
    cd_table_new = np.zeros((n, 4))
    i = 0
    for elem in cl_table:
        alt = elem[0]
        alpha = elem[1]
        mach = elem[2]
        cl = elem[3]
        selected_cd_table = cd_table[(cd_table[:, 0] == alt) & (cd_table[:, 2] == mach)]
        # print(selected_cd_table)
        x = selected_cd_table[:, 1]
        y = selected_cd_table[:, 3]
        f_interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
        z = float(f_interp(cl))
        new_elem = [alt, alpha, mach, z]
        cd_table_new[i] = new_elem
        i = i + 1
    tables['CD'] = cd_table_new

    return scalars, tables, fields


def _read_header(f):
    """Read GASP aero header, returning the area scalars in a dict."""
    # file header: FORMAT(4I5,10X,2F10.4)
    iread, iprint, icd0, icompss, sref_at, cbar_at = _parse(
        f, [*_rep(4, (int, 5)), *_rep(2, (float, 10))]
    )
    if iread == 1:
        raise Exception('Reading of Reynolds number is not implemented yet.')
    if icd0 == 1:
        raise Exception('Reading of CD0 table is not implemented yet.')

    return {
        'sref_at': sref_at,
        'cbar_at': cbar_at,
    }


def _read_table(f, is_turbo_prop=False):
    """Read an entire table from a GASP area file.
    The table data is returned as a "tidy format" array with three columns for the
    independent variables (altitude, alpha, and Mach number) and the final columns for
    the table field (CL and CD).
    """
    tab_data = None

    # strip out table title, not used
    f.readline().strip()
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


def _make_structured_grid(data, method='lagrange3', fields=['CL', 'CD']):
    """Generate a structured grid of unique mach/aoa/alt values in the deck."""
    aoa_step = 0.05
    # step size in Mach number used in generating the structured grid
    # mach_step = 0.02 # original value
    mach_step = 0.05

    structured_data = {}

    # find min/max from CL table
    aoa = data['CL'][:, 1]
    tma = data['CL'][:, 2]
    min_aoa = min(aoa)
    max_aoa = max(aoa) + aoa_step
    min_tma = min(tma)
    max_tma = max(tma) + mach_step

    aoas = np.arange(min_aoa, max_aoa + aoa_step, aoa_step)
    machs = np.arange(min_tma, max_tma + mach_step, mach_step)

    # need altitude in first column, mach varies on each row
    pts = np.dstack(np.meshgrid(aoas, machs, indexing='ij')).reshape(-1, 2)
    npts = pts.shape[0]

    for field in fields:
        map_data = data[field]
        all_alts = map_data[:, 0]
        alts = np.unique(all_alts)

        sizes = (alts.size, aoas.size, machs.size)
        vals = np.zeros(np.prod(sizes), dtype=float)
        alt_vec = np.zeros(np.prod(sizes), dtype=float)
        mach_vec = np.zeros(np.prod(sizes), dtype=float)
        aoa_vec = np.zeros(np.prod(sizes), dtype=float)

        for i, alt in enumerate(alts):
            d = map_data[all_alts == alt]
            aoa = np.unique(d[:, 1])
            mach = np.unique(d[:, 2])
            f = d[:, 3].reshape(aoa.size, mach.size)

            # would explicitly use lagrange3 here to mimic GASP, but some aero
            # tables may not have enough points per dimension
            # For GASP aero table, try to provide at least 4 Mach numbers.
            # avoid devide-by-zero RuntimeWarning
            if len(mach) == 3 and method == 'lagrange3':
                method = 'lagrange2'
            elif len(mach) == 2:
                method = 'slinear'
            interp = InterpND(method='2D-' + method, points=(aoa, mach), values=f, extrapolate=True)
            sl = slice(i * npts, (i + 1) * npts)
            vals[sl] = interp.interpolate(pts)
            alt_vec[sl] = [alt] * len(pts)
            mach_vec[sl] = pts[:, 1]
            aoa_vec[sl] = pts[:, 0]

        structured_data[field] = {
            'alts': alt_vec,
            'machs': mach_vec,
            'aoas': aoa_vec,
            'vals': vals,
        }

    return structured_data


def _setup_ATC_parser(parser):
    parser.add_argument('input_file', type=str, help='path to aero data file to be converted')
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        help='path to file where new converted data will be written',
    )
    parser.add_argument(
        '-f',
        '--data_format',
        type=str,
        choices=[origin.value for origin in CodeOrigin],
        help='data format used by input_file',
    )


def _exec_ATC(args, user_args):
    convert_aero_table(
        input_file=args.input_file, output_file=args.output_file, data_format=args.data_format
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts FLOPS- or GASP-formatted aero data files into Aviary csv format.\n'
    )
    _setup_ATC_parser(parser)
    args = parser.parse_args()
    _exec_ATC(args, None)
