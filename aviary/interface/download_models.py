import os
from pathlib import Path
import argparse
import pkg_resources
import shutil


def aviary_resource(resource_name: str) -> str:
    return pkg_resources.resource_filename("aviary", resource_name)


def get_model(file_name: str, verbose=False) -> Path:
    '''
    This function attempts to find the path to a file or folder in aviary/models
    If the path cannot be found in any of the locations, a FileNotFoundError is raised.

    Parameters
    ----------
    path : str or Path
        The input path, either as a string or a Path object.

    Returns
    -------
    aviary_path
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the path is not found.
    '''

    # Get the path to Aviary's models
    path = Path('models', file_name)
    aviary_path = Path(aviary_resource(str(path)))

    # If the file name was provided without a path, check in the subfolders
    if not aviary_path.exists():
        sub_dirs = [x[0] for x in os.walk(aviary_resource('models'))]
        for sub_dir in sub_dirs:
            temp_path = Path(sub_dir, file_name)
            if temp_path.exists():
                # only return the first matching file
                aviary_path = temp_path
                continue

    # If the path still doesn't exist, raise an error.
    if not aviary_path.exists():
        raise FileNotFoundError(
            f"File or Folder not found in Aviary's hangar"
        )
    if verbose:
        print('found', aviary_path, '\n')
    return aviary_path


def save_file(aviary_path: Path, outdir: Path, verbose=False) -> Path:
    '''
    Saves the file or folder specified into the output directory, creating directories as needed.
    '''
    outdir.mkdir(parents=True, exist_ok=True)
    if aviary_path.is_dir():
        if verbose:
            print(aviary_path, 'is a directory, getting all files')
        outdir = outdir.joinpath(aviary_path.stem)
        outdir.mkdir(exist_ok=True)
        for file in next(os.walk(aviary_path))[-1]:
            if verbose:
                print('copying', str(aviary_path / file), 'to', str(outdir / file))
            shutil.copy2(aviary_path / file, outdir)
    else:
        if verbose:
            print('copying', str(aviary_path), 'to', str(outdir / aviary_path.name))
        shutil.copy2(aviary_path, outdir)
    return outdir


def _setup_hangar_parser(parser: argparse.ArgumentParser):
    def_outdir = os.path.join(os.getcwd(), "aviary_models")
    parser.add_argument(
        'input_decks', metavar='indecks', type=str, nargs='+', help='Name of file or folder to download from Aviary/models'
    )
    parser.add_argument(
        "-o", "--outdir", default=def_outdir, help="Directory to write outputs. Defaults to aviary_models in the current directory."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose outputs",
    )


def _exec_hangar(args, user_args):
    input_decks = []
    for input_deck in args.input_decks:
        input_decks.append(get_model(input_deck, args.verbose))

    for input_deck in input_decks:
        save_file(input_deck, Path(args.outdir), args.verbose)
