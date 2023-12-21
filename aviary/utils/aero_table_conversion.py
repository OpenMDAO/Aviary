#!/usr/bin/python

import argparse
import re

import numpy as np

from enum import Enum
from pathlib import Path

from aviary.api import NamedValues
from aviary.utils.csv_data_file import write_data_file
from aviary.utils.functions import get_path


class CodeOrigin(Enum):
    FLOPS = 'FLOPS'
    GASP = 'GASP'


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
    'del_cd': 'Delta CD'
}


def AeroDataConverter(input_file=None, output_file=None, data_format=None):
    data_format = CodeOrigin(data_format)
    data_file = get_path(input_file)
    if not output_file:
        if data_format is CodeOrigin.GASP:
            # Default output file name is same location and name as input file, with
            # '_aviary' appended to filename
            path = input_file.parents[0]
            name = input_file.name
            suffix = input_file.suffix
            output_file = path / (name + '_aviary' + suffix)
        elif data_format is CodeOrigin.FLOPS:
            # Default output file name is same location and name as input file, with
            # '_aviary' appended to filename
            path = input_file.parents[0]
            name = input_file.stem
            suffix = input_file.suffix
            file1 = path / name + '_aviary_CDI' + suffix
            file2 = path / name + '_aviary_CD0' + suffix
            output_file = [file1, file2]

    stamp = f'# {data_format.value}-derived aerodynamics data converted from {data_file.name}'

    if data_format is CodeOrigin.GASP:
        data, comments = _load_gasp_aero_table(data_file)
        comments = [stamp] + comments

        write_data_file(output_file, data, comments, include_timestamp=True)
    elif data_format is CodeOrigin.FLOPS:
        if type(output_file) is not list:
            # if only one filename is given, split into two
            path = output_file.parents[0]
            name = output_file.stem
            suffix = output_file.suffix
            file1 = path / (name + '_CDi' + suffix)
            file2 = path / (name + '_CD0' + suffix)
            output_file = [file1, file2]

        lift_drag_data, lift_drag_comments, \
            zero_lift_drag_data, zero_lift_drag_comments = _load_flops_aero_table(
                data_file)

        # write lift-dependent drag file
        lift_drag_comments = [stamp] + lift_drag_comments
        write_data_file(output_file[0], lift_drag_data,
                        lift_drag_comments, include_timestamp=True)

        # write zero-lift drag file
        zero_lift_drag_comments = [stamp] + zero_lift_drag_comments
        write_data_file(output_file[1], zero_lift_drag_data,
                        zero_lift_drag_comments, include_timestamp=True)


def _load_flops_aero_table(filepath: Path):
    """Load an aero table in FLOPS format"""

    def _read_line(line_count, comments):
        line = file_contents[line_count].strip()
        items = re.split(r'[\s]*\s', line)
        if items[0] == '#':
            comments.append(line)
            nonlocal offset
            offset += 1
            try:
                items = _read_line(line_count+1, comments)
            except IndexError:
                return
        else:
            # try to convert line to float
            try:
                items = [float(var) for var in items]
            # data contains things other than floats
            except (ValueError):
                raise ValueError(
                    f'Non-numerical value found in data file <{filepath.name}> on '
                    f'line {str(line_count)}')

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
            raise ValueError('Number of data points provided for '
                             f'lift-dependent drag at Mach {lift_drag_machs[i]} '
                             'does not match number of CLs provided '
                             f'(FLOPS aero data file {filepath.name})')
    if len(lift_drag) != len(lift_drag_machs):
        raise ValueError('Number of data rows provided for lift-dependent drag does '
                         'not match number of Mach numbers provided (FLOPS aero data '
                         f'file {filepath.name})')
    offset = offset + i
    # these are not needed, we can determine the length of data vectors directly
    altitude_count, zero_lift_mach_count = _read_line(
        4 + offset, zero_lift_drag_comments)
    altitudes = _read_line(5 + offset, zero_lift_drag_comments)
    zero_lift_machs = _read_line(6 + offset, zero_lift_drag_comments)
    for i in range(len(altitudes)):
        drag = _read_line(7 + i + offset, zero_lift_drag_comments)
        if len(drag) == len(zero_lift_machs):
            zero_lift_drag.append(drag)
        else:
            raise ValueError('Number of data points provided for '
                             f'zero-lift drag at altitude {altitudes[i]} '
                             'does not match number of Machs provided '
                             f'(FLOPS aero data file {filepath.name})')
    if len(zero_lift_drag) != len(altitudes):
        raise ValueError('Number of data rows provided for zero-lift drag does '
                         'not match number of altitudes provided (FLOPS aero data '
                         f'file {filepath.name})')

    cl, mach = np.meshgrid(cls, lift_drag_machs)
    lift_drag_data.set_val('Mach', mach.flatten(), 'unitless')
    lift_drag_data.set_val('Lift Coefficient', cl.flatten(), 'unitless')
    lift_drag_data.set_val('Lift-Dependent Drag Coefficient',
                           np.array(lift_drag).flatten(), 'unitless')

    mach, altitude = np.meshgrid(zero_lift_machs, altitudes)
    zero_lift_drag_data.set_val('Altitude', altitude.flatten(), 'ft')
    zero_lift_drag_data.set_val('Mach', mach.flatten(), 'unitless')
    zero_lift_drag_data.set_val('Zero-Lift Drag Coefficient',
                                np.array(zero_lift_drag).flatten(), 'unitless')

    return lift_drag_data, lift_drag_comments, zero_lift_drag_data, zero_lift_drag_comments


def _load_gasp_aero_table(filepath: Path):
    """Load an aero table in GASP format"""
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
                except (ValueError):
                    raise ValueError(
                        f'Non-numerical value found in data file <{filepath.name}> on '
                        f'line {str(line_count)}')
                else:
                    raw_data.append(line_data)

    raw_data = np.array(raw_data)

    # translate raw data into NamedValues object
    for idx, var in enumerate(variables):
        if var.lower() in allowed_headers:
            data.set_val(allowed_headers[var.lower()], raw_data[:, idx], units[idx])

    return data, comments


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts FLOPS- or GASP-formatted '
                                     'aero data files into Aviary csv format.\n')
    parser.add_argument('input_file', type=str,
                        help='path to engine deck file to be converted')
    parser.add_argument('output_file', type=str,
                        help='path to file where new converted data will be written')
    parser.add_argument('data_format', type=str, choices=[origin.value for origin in CodeOrigin],
                        help='data format used by input_file')

    args = parser.parse_args()

    AeroDataConverter(**vars(args))
