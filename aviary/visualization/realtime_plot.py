import mimetypes
import os
import pathlib
import subprocess
from openmdao.utils.file_utils import _load_and_exec
import openmdao.utils.hooks as hooks


def is_python_file(file_path):
    """
    Check if file is a Python source file using multiple methods.

    Parameters
    ----------
    file_path : str
        The path to a file.

    Returns
    -------
    bool
        True if file is a python file. False, if not.
    """
    # Method 1: Check file extension
    if pathlib.Path(file_path).suffix.lower() in ['.py', '.pyw', '.pyi']:
        return True

    # Method 2: Check MIME type
    try:
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type in ['text/x-python', 'application/x-python-code']:
            return True
    except (TypeError, ValueError, OSError):
        pass

    return False


def _rtplot_setup_parser(parser):
    """
    Set up the aviary subparser for the 'aviary rtplot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')


def _rtplot_cmd(options, user_args):
    """
    Return the post_setup hook function for 'aviary rtplot'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    script_path = options.file[0]
    if not is_python_file(script_path):
        raise RuntimeError(
            'The argument to the aviary rtplot command must be an OpenMDAO python script.'
        )

    def _view_realtime_plot_hook(problem):
        driver = problem.driver
        if not driver:
            raise RuntimeError(
                'Unable to run realtime optimization progress plot because no Driver'
            )
        if len(problem.driver._rec_mgr._recorders) == 0:
            raise RuntimeError(
                'Unable to run realtime optimization progress plot '
                'because no case recorder attached to Driver'
            )

        case_recorder_file = str(problem.driver._rec_mgr._recorders[0]._filepath)

        cmd = ['openmdao', 'realtime_plot', '--pid', str(os.getpid()), case_recorder_file]
        cmd.insert(-1, '--script')
        cmd.insert(-1, script_path)
        cp = subprocess.Popen(cmd)  # nosec: trusted input

        # Do a quick non-blocking check to see if it immediately failed
        # This will catch immediate failures but won't wait for the process to finish
        quick_check = cp.poll()
        if quick_check is not None and quick_check != 0:
            # Process already terminated with an error
            stderr = cp.stderr.read().decode()
            raise RuntimeError(
                f'Failed to start up the realtime plot server with code {quick_check}: {stderr}.'
            )

    # register the hook
    hooks._register_hook('_setup_recording', 'Problem', post=_view_realtime_plot_hook, ncalls=1)
    # run the script
    _load_and_exec(script_path, user_args)
