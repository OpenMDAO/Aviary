import argparse
import os
import sys

import aviary
from aviary.interface.download_models import _exec_hangar, _setup_hangar_parser
from aviary.interface.graphical_input import _exec_flight_profile, _setup_flight_profile_parser
from aviary.interface.methods_for_level1 import _exec_level1, _setup_level1_parser
from aviary.interface.plot_drag_polar import _exec_plot_drag_polar, _setup_plot_drag_polar_parser
from aviary.interface.test_installation import _exec_installation_test, _setup_installation_test
from aviary.utils.aero_table_conversion import _exec_ATC, _setup_ATC_parser
from aviary.utils.engine_deck_conversion import EDC_description, _exec_EDC, _setup_EDC_parser
from aviary.utils.fortran_to_aviary import _exec_F2A, _setup_F2A_parser
from aviary.utils.propeller_map_conversion import _exec_PMC, _setup_PMC_parser
from aviary.visualization.dashboard import _dashboard_cmd, _dashboard_setup_parser


def _load_and_exec(script_name, user_args):
    """
    Load and exec the given script as __main__.

    Parameters
    ----------
    script_name : str
        The name of the script to load and exec.
    user_args : list of str
        Args to be passed to the user script.
    """
    sys.path.insert(0, os.path.dirname(script_name))

    sys.argv[:] = [script_name] + user_args

    with open(script_name, 'rb') as fp:
        code = compile(fp.read(), script_name, 'exec')

    globals_dict = {
        '__file__': script_name,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    exec(code, globals_dict)  # nosec: private, internal use only


_command_map = {
    'check': (
        _setup_installation_test,
        _exec_installation_test,
        'Verifies Aviary installation',
    ),
    'fortran_to_aviary': (
        _setup_F2A_parser,
        _exec_F2A,
        'Converts legacy Fortran input decks to Aviary csv based decks',
    ),
    'run_mission': (_setup_level1_parser, _exec_level1, 'Runs Aviary using a provided input deck'),
    'draw_mission': (
        _setup_flight_profile_parser,
        _exec_flight_profile,
        'Allows users to draw a mission profile for use in Aviary.',
    ),
    'dashboard': (_dashboard_setup_parser, _dashboard_cmd, 'Run the Dashboard tool'),
    'hangar': (
        _setup_hangar_parser,
        _exec_hangar,
        'Allows users that pip installed Aviary to download models from the Aviary hangar',
    ),
    'convert_engine': (_setup_EDC_parser, _exec_EDC, EDC_description),
    'convert_aero_table': (
        _setup_ATC_parser,
        _exec_ATC,
        'Converts FLOPS- or GASP-formatted aero data files into Aviary csv format.',
    ),
    'convert_prop_table': (
        _setup_PMC_parser,
        _exec_PMC,
        'Converts GASP-formatted propeller map file into Aviary csv format.',
    ),
    'plot_drag_polar': (
        _setup_plot_drag_polar_parser,
        _exec_plot_drag_polar,
        'Plot a Drag Polar Graph using a provided polar data csv input',
    ),
}


def aviary_cmd():
    """Run an 'aviary' sub-command or list help info for 'aviary' command or sub-commands."""
    # pre-parse sys.argv to split between before and after '--'
    if '--' in sys.argv:
        idx = sys.argv.index('--')
        sys_args = sys.argv[:idx]
        user_args = sys.argv[idx + 1 :]
        sys.argv[:] = sys_args
    else:
        user_args = []

    parser = argparse.ArgumentParser(
        description='aviary Command Line Tools',
        #  epilog='Use -h after any sub-command for sub-command help, '
        #  'for example, "openmdao tree -h" for help on the "tree" '
        #  'command. If using a tool on a script that takes its own '
        #  'command line arguments, place those arguments after a "--". '
        #  'For example: '
        #  '"openmdao n2 -o foo.html myscript.py -- -x --myarg=bar"'
    )

    # Adding the --version argument
    parser.add_argument('--version', action='store_true', help='show version and exit')

    subs = parser.add_subparsers(title='Tools', metavar='', dest='subparser_name')
    for p, (parser_setup_func, executor, help_str) in sorted(_command_map.items()):
        subp = subs.add_parser(p, help=help_str)
        parser_setup_func(subp)
        subp.set_defaults(executor=executor)

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    # '--version', '--dependency_versions')]
    cmdargs = [a for a in sys.argv[1:] if a not in ('-h',)]

    if len(args) == 1 and len(user_args) == 0:
        # if command requires arguments but is run without any, return help for that command
        if args[0] not in ('check', 'draw_mission', 'run_mission', 'plot_drag_polar'):
            parser.parse_args([args[0], '-h'])

    if not set(args).intersection(subs.choices) and len(args) == 1 and os.path.isfile(cmdargs[0]):
        _load_and_exec(args[0], user_args)
    else:
        options, unknown = parser.parse_known_args()

        # Check if --version was passed
        if options.version:
            print(f'Aviary version: {aviary.__version__}')
            return

        if unknown:
            msg = 'unrecognized arguments: ' + ', '.join(unknown)
            try:
                sub = subs.choices[options.subparser_name]
            except KeyError:
                parser.error(msg)
            else:
                print(sub.format_usage(), file=sys.stderr)
                print(msg, file=sys.stderr)
            parser.exit(2)

        if hasattr(options, 'executor'):
            options.executor(options, user_args)
        else:
            os.system('aviary -h')


if __name__ == '__main__':
    aviary_cmd()
