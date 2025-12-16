"""
Command line api for the engine deck converter.
Kept in a separate file to speed up the command line imports.
"""
from aviary.variable_info.enums import EngineDeckType


def _setup_EDC_parser(parser):
    parser.add_argument('input_file', type=str, help='path to engine deck file to be converted')
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        help='path to file where new converted data will be written',
    )
    parser.add_argument(
        '-f',
        '--data_format',
        type=EngineDeckType,
        choices=list(EngineDeckType),
        help='data format used by input_file',
    )
    parser.add_argument('--round', action='store_true', help='round data to improve readability')


def _exec_EDC(args, user_args):

    from aviary.utils.engine_deck_conversion import convert_engine_deck

    convert_engine_deck(
        input_file=args.input_file,
        output_file=args.output_file,
        data_format=args.data_format,
        round_data=args.round,
    )
