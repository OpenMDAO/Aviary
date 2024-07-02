import os

import numpy as np
import openmdao.api as om

from aviary.variable_info.variable_meta_data import _MetaData

base_path = os.path.abspath(os.path.join(__file__, "..", "..", ".."))


def assert_no_duplicates(
    check_list,
):
    full_list = []
    for entry in check_list:
        if np.shape(entry) == ():
            full_list.append(entry)
        else:
            entry = set(entry)
            for name in entry:
                full_list.append(name)
    full_list = list(filter(filter_empty, full_list))

    sorted_list = sorted(full_list)
    set_list = set(full_list)
    duplicates = []
    if len(sorted_list) != len(set_list):
        for idx, check_name in enumerate(sorted_list[1:]):
            if check_name == sorted_list[idx]:
                duplicates.append(check_name)
        duplicates = list(set(duplicates))

        raise ValueError(
            f"The variables {duplicates} are duplicates in the provided list."
        )

    return duplicates


def assert_structure_alphabetization(file_loc):
    """
    Assert that an aviary variable hierarchy is properly sorted.

    Parameters:
    -----------
    file_loc : str
        Location of the hierarchy file relative to the aviary top directory.
    """
    full_file_loc = os.path.abspath(os.path.join(base_path, file_loc))

    with open(full_file_loc) as variable_file:
        lines = variable_file.readlines()

    previous_line = ""
    previous_stem = ""
    nested_stems = {0: ''}
    bad_sort = []
    in_block_comment = False
    for line in lines:
        line_ls = line.lstrip()

        if line_ls.startswith('#'):
            # Skip any comments.
            continue

        leading_spaces = len(line) - len(line_ls)

        line = line_ls.rstrip()

        if len(line) == 0:
            # Skip whitespace-only lines
            continue

        if line.startswith("'''") or line.startswith('"""'):
            in_block_comment = not in_block_comment
            continue
        elif in_block_comment:
            continue

        if line.startswith('class'):
            # Class lines
            class_name = line.split("class ")[-1]
            current_stem = f"{nested_stems[leading_spaces]}{class_name}"
            nested_stems[leading_spaces + 4] = current_stem
            current_stem_fix = current_stem.casefold()

            if current_stem_fix < previous_stem:
                bad_sort.append(current_stem)

            previous_stem = current_stem_fix
            previous_line = ""

        elif "=" in line:
            # Variable lines
            var_name = line.split("=")[0].strip()
            current_line = f"{nested_stems[leading_spaces]}{var_name}"
            current_line_fix = current_line.casefold()

            if current_line_fix < previous_line:
                bad_sort.append(current_line)

            previous_line = current_line_fix

        if leading_spaces % 4 > 0:
            msg = "The variable file is not using proper Python spacing."
            raise ValueError(msg)

    if len(bad_sort) > 0:
        txt = '\n'.join(bad_sort)
        msg = f'The following variables are out of order:\n{txt}'
        raise ValueError(msg)


def assert_metadata_alphabetization(metadata_variables_list):
    previous_var = metadata_variables_list[0].split(":")
    out_of_order_vars = []

    for var_name in metadata_variables_list[1:]:
        current_var = var_name.split(":")
        max_size = min(len(current_var), len(previous_var))
        same = [previous_var[ct] == current_var[ct] for ct in np.arange(max_size)]
        try:
            diff_idx = same.index(False)
        except ValueError:
            raise ValueError(
                "There are two variables in the metadata file that have the same string name."
            )
        # only compare up to class level, avoid comparing variables to classes
        if len(current_var) > len(previous_var):
            diff_idx = max(0, diff_idx - 1)

        old_to_new = [previous_var[diff_idx], current_var[diff_idx]]
        old_to_new_alphabetical = old_to_new.copy()
        old_to_new_alphabetical.sort(key=str.casefold)
        is_alphabetical = old_to_new == old_to_new_alphabetical

        if not is_alphabetical:
            out_of_order_vars.append(var_name)

        previous_var = current_var

    if out_of_order_vars:
        raise ValueError(
            f"The variable(s) {out_of_order_vars} in the metadata are out of alphabetical order with their previous value."
        )


def assert_match_varnames(system, MetaData=None):

    prob = om.Problem()
    prob.model = system
    prob.setup()
    prob.final_setup()
    sys_inputs = prob.model.list_inputs(out_stream=None, prom_name=True)
    sys_inputs = set([val[1]["prom_name"] for val in sys_inputs])
    sys_outputs = prob.model.list_outputs(out_stream=None, prom_name=True)
    sys_outputs = set([val[1]["prom_name"] for val in sys_outputs])

    proper_var_names = set(
        [key for key in (_MetaData if MetaData is None else MetaData)]
    )

    input_overlap = sys_inputs.intersection(proper_var_names)
    output_overlap = sys_outputs.intersection(proper_var_names)

    if input_overlap != sys_inputs:
        diff = sys_inputs - input_overlap
        raise ValueError(
            f"The inputs {diff} in the provided subsystem are not found in the provided variable structure."
        )

    if output_overlap != sys_outputs:
        diff = sys_outputs - output_overlap
        raise ValueError(
            f"The outputs {diff} in the provided subsystem are not found in the provided variable structure."
        )


def get_names_from_hierarchy(hierarchy):
    """
    Return a list of all openmdao variable names in the variable hierarchy.

    This is used for finding duplicates names.

    Parameters:
    -----------
    hierarchy: object
        Instance of a class hierarchy such as Aircraft.

    Returns
    -------
    list
        List of all names in the hierarchy, including duplicates.
    """
    names = []

    # Keys
    keys = vars(hierarchy).keys()
    keys = list(filter(filter_underscore, list(keys)))

    for key in keys:
        leaf = getattr(hierarchy, key)
        if isinstance(leaf, str):
            # Variable String.
            names.append(leaf)

        else:
            # Subclass.
            sub_names = get_names_from_hierarchy(leaf)
            names.extend(sub_names)

    return names


def filter_empty(entry):
    if (entry != '') and (entry != None):
        return True

    else:
        return False


def filter_underscore(entry):
    if '__' not in entry:
        return True

    else:
        return False
