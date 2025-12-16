"""
This file contains the command-line hooks for the dashboard. It is in a separate file so that we don't
import Bokeh unless we need it. This greatly speeds up the command line.
"""
import argparse
from pathlib import Path
import shutil
import zipfile


def _dashboard_setup_parser(parser):
    """
    Set up the aviary subparser for the 'aviary dashboard' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        'script_name',
        type=str,
        nargs='*',
        help='Name of aviary script that was run (not including .py).',
    )
    parser.add_argument(
        '--port',
        dest='port',
        type=int,
        default=0,
        help='dashboard server port ID (default is 0, which indicates get any free port)',
    )
    parser.add_argument(
        '-b',
        '--background',
        action='store_true',
        dest='run_in_background',
        help="Run the server in the background (don't automatically open the browser)",
    )

    # For future use
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        dest='debug_output',
        help='show debugging output',
    )

    parser.add_argument(
        '--save',
        nargs='?',
        const=True,
        default=False,
        help='Name of zip file in which dashboard files are saved. If no argument given, use the script name to name the zip file',
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='When displaying data from a shared zip file, if the directory in the reports directory exists, overrite if this is True',
    )


def _dashboard_cmd(options, user_args):
    """
    Run the dashboard command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if options.save and not options.script_name:
        if options.save is not True:
            options.script_name = options.save
            options.save = True

    if not options.script_name:
        raise argparse.ArgumentError('script_name argument missing')

    if isinstance(options.script_name, list):
        options.script_name = options.script_name[0]

    from aviary.visualization.dashboard import dashboard

    # check to see if options.script_name is a zip file
    # if yes, then unzip into reports directory and run dashboard on it
    if zipfile.is_zipfile(options.script_name):
        report_dir_name = Path(options.script_name).stem
        report_dir_path = Path(f'{report_dir_name}_out')
        # need to check to see if that directory already exists
        if not options.force and report_dir_path.is_dir():
            raise RuntimeError(
                f'The reports directory {report_dir_path} already exists. If you wish '
                'to overwrite the existing directory, use the --force option'
            )
        if (
            report_dir_path.is_dir()
        ):  # need to delete it. The unpacking will just add to what is there, not do a clean unpack
            shutil.rmtree(report_dir_path)

        shutil.unpack_archive(options.script_name, report_dir_path)
        dashboard(
            report_dir_name,
            # options.problem_recorder,
            # options.driver_recorder,
            options.port,
            options.run_in_background,
        )
        return

    # Save the dashboard files to a zip file that can be shared with others
    if options.save is not False:
        if options.save is True:
            save_filename_stem = options.script_name
        else:
            save_filename_stem = Path(options.save).stem
        print(f'Saving to {save_filename_stem}.zip')
        shutil.make_archive(save_filename_stem, 'zip', f'{options.script_name}_out')
        return

    dashboard(
        options.script_name,
        # options.problem_recorder,
        # options.driver_recorder,
        options.port,
        options.run_in_background,
    )


