"""
Command line api for the aero table converter.
Kept in a separate file to speed up the command line imports.
"""

from aviary.variable_info.enums import CodeOrigin


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
    from aviary.utils.aero_table_conversion import convert_aero_table

    convert_aero_table(
        input_file=args.input_file, output_file=args.output_file, data_format=args.data_format
    )
