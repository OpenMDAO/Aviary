import atexit
from contextlib import ExitStack
from pathlib import Path
from typing import Union

import importlib_resources

from aviary.utils.aviary_values import AviaryValues, get_items
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variable_meta_data import _MetaData


def get_aviary_resource_path(resource_name: str) -> Path:
    """
    Get the file path of a resource in the Aviary package.

    Parameters
    ----------
        resource_name : str
            The name of the resource.

    Returns
    -------
        path : Path
            The file path of the resource.

    """
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    if resource_name:
        ref = importlib_resources.files('aviary') / resource_name
    else:
        ref = importlib_resources.files('aviary')
    path = file_manager.enter_context(importlib_resources.as_file(ref))
    return path


top_dir = Path(get_aviary_resource_path(''))


def set_aviary_initial_values(prob, aviary_inputs: AviaryValues):
    """
    Sets initial values for all inputs in the aviary inputs.

    This method is mostly used in tests and level 3 scripts.

    Parameters
    ----------
    prob : Problem
        OpenMDAO problem after setup.
    aviary_inputs : AviaryValues
        Instance of AviaryValues containing all initial values.
    """
    for key, (val, units) in get_items(aviary_inputs):
        try:
            prob.set_val(key, val, units)

        except BaseException:
            # Should be an option or an overridden output.
            continue


def set_aviary_input_defaults(model, inputs, aviary_inputs: AviaryValues, meta_data=_MetaData):
    """
    This function sets the default values and units for any inputs prior to
    setup. This is needed to resolve ambiguities when inputs are promoted
    with the same name, but different units or values.

    This method is mostly used in tests and level 3 scripts.

    Parameters
    ----------
    model : System
        Top level aviary model.
    inputs : list
        List of variables that are causing promotion problems. This needs to
        be crafted based on the openmdao exception messages.
    aviary_inputs : AviaryValues
        Instance of AviaryValues containing all initial values.
    meta_data : dict
        (Optional) Dictionary of aircraft metadata. Uses Aviary's built-in
        metadata by default.
    """
    for key in inputs:
        if key in aviary_inputs:
            val, units = aviary_inputs.get_item(key)
        else:
            val = meta_data[key]['default_value']
            units = meta_data[key]['units']

        model.set_input_defaults(key, val=val, units=units)


def convert_strings_to_data(input_string):
    """
    convert_strings_to_data will convert a string or list of strings to usable data.
    Strings that can't be converted to numbers will attempt to store as a boolean,
    otherwise they are passed as is.
    """
    # pack input_string into a list if it is not
    # setup output list size
    if isinstance(input_string, list):
        islist = True
        value_list = [0] * len(input_string)
    else:
        islist = False
        input_string = [input_string]
        value_list = input_string

    for ii, dat in enumerate(input_string):
        dat = dat.strip('[]')
        try:
            # if the value is a number store it as a float or an int as appropriate
            # BUG this returns floats that can be converted to int (e.g. 1.0) as an
            # int (1), even if the variable requires floats
            value_list[ii] = int(dat) if '.' not in dat else float(dat)
        except ValueError:
            # store value as a boolean if it is a string that represents True or False
            if dat.lower() == 'true':
                value_list[ii] = True
            elif dat.lower() == 'false':
                value_list[ii] = False
            else:
                # if the value isn't a number or a boolean, store it as a string
                value_list[ii] = dat
    # unpack output value from list if it isn't supposed to be one
    if not islist:
        value_list = value_list[0]
    return value_list


def promote_aircraft_and_mission_vars(group):
    """Promotes inputs and outputs in Aircraft and Mission hierarchy categories for provided group."""
    external_outputs = []
    for comp in group.system_iter(recurse=False):
        # Skip all aviary systems.
        if comp.name == 'core_subsystems':
            continue

        try:
            resolver = comp._resolver
            out_names = [item for item in resolver.prom_iter(iotype='output')]
            in_names = [item for item in resolver.prom_iter(iotype='input')]

        except AttributeError:
            # This is an older version of OpenMDAO
            out_names = [item for item in comp._var_allprocs_prom2abs_list['output']]
            in_names = [item for item in comp._var_allprocs_prom2abs_list['input']]

        external_outputs.extend(out_names)

        # Locally promote aircraft:* and mission:* only.
        promote_in = []
        promote_out = []

        for stem in ['mission:', 'aircraft:', 'dynamic:']:
            for name in out_names:
                if name.startswith(stem):
                    promote_out.append(f'{stem}*')
                    break

            for name in in_names:
                if name.startswith(stem):
                    promote_in.append(f'{stem}*')
                    break

        group.promotes(comp.name, outputs=promote_out, inputs=promote_in)

    return external_outputs


# Python 3.10 adds the ability to specify multiple types using type hints like so:
# "str | Path" which is cleaner but Aviary still supports older versions


def get_path(path: Union[str, Path], verbosity=Verbosity.BRIEF) -> Path:
    """
    Convert a string or Path object to an absolute Path object, prioritizing different locations.

    This function attempts to find the existence of a path in the following order:
    1. As an absolute path.
    2. Relative to the current working directory.
    3. Relative to the Aviary package.

    If the path cannot be found in any of the locations, a FileNotFoundError is raised.

    Parameters
    ----------
    path : str or Path
        The input path, either as a string or a Path object.
    verbosity : Verbosity, optional
        Sets level of printouts for this function.

    Returns
    -------
    Path
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the path is not found in any of the prioritized locations.
    """
    # Store the original path for reference in error messages.
    original_path = path

    # If the input is a string, convert it to a Path object.
    if isinstance(path, str):
        path = Path(path)

    # Check if the path exists as an absolute path.
    if not path.exists():
        # If not, try finding the path relative to the current working directory.
        relative_path = Path.cwd() / path
        path = relative_path

    # If the path still doesn't exist, attempt to find it relative to the Aviary package.
    if not path.exists():
        if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
            print(
                f"Unable to locate '{original_path}' as an absolute or relative path. "
                'Trying Aviary package path.'
            )
        # Determine the path relative to the Aviary package.
        aviary_based_path = Path(get_aviary_resource_path(original_path))

        path = aviary_based_path

    # If the path still doesn't exist, attempt to find it in the models directory.
    if not path.exists():
        if verbosity > Verbosity.BRIEF:
            print(
                f"Unable to locate '{aviary_based_path}' as an Aviary package path, "
                'checking built-in models'
            )
        try:
            hangar_based_path = get_model(original_path)
            path = hangar_based_path
        except FileNotFoundError:
            pass

    # If the path still doesn't exist in any of the prioritized locations, raise an error.
    if not path.exists():
        raise FileNotFoundError(
            f'File not found in absolute path: {original_path}, relative path: '
            f'{relative_path}, or Aviary-based path: '
            f'{Path(get_aviary_resource_path(original_path))}'
        )

    # Print the path being used.
    if verbosity > Verbosity.BRIEF:
        print(f'Found {path}')

    return path


def get_model(file_name: str, verbosity=Verbosity.BRIEF) -> Path:
    """
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
    """
    # Get the path to Aviary's models
    aviary_path = Path(get_aviary_resource_path(str(Path('models', file_name))))
    # Check if provided path is valid
    if aviary_path.exists():
        return aviary_path
    # otherwise check models folder contents
    else:
        from glob import glob

        contents = glob(str(get_aviary_resource_path('models') / '**'), recursive=True)
        close_match = None
        for item in contents:
            item = Path(item)
            # check if full filepath, file name with extension, or just file (or folder) name
            # matches target
            if aviary_path == item or aviary_path.name == item.name:
                return item
            elif aviary_path.stem == item.stem:
                close_match = item

    if close_match is not None:
        # Probably requested the wrong file extension.
        return close_match

    # If the path doesn't exist, raise an error.
    raise FileNotFoundError("File or Folder not found in Aviary's hangar")
