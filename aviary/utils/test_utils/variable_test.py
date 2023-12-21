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


def assert_structure_alphabetization(variables_file_loc):
    #################################################################################################################
    # Extract Class and Variable Names from File #
    #################################################################################################################

    full_file_loc = os.path.abspath(os.path.join(base_path, variables_file_loc))
    with open(full_file_loc) as variable_file:
        # variable_file = open(full_file_loc, "r")
        lines = variable_file.readlines()
        variable_file.close()
        ct = 0
        superclass_names = []  # comprehensive list of the names of the superclasses
        subclass_names_tot = []  # comprehensive list of the names of the subclasses
        subclass_names = []  # list of names of subclasses in each individual superclass, intermediate variable that is overwritten for each superclass
        variable_names_tot = []  # comprehensive list of the names of the variables
        variable_names = []  # list of names of variable in each individual subclass, intermediate variable that is overwritten for each subclass
        # list of names of variables in each individual superclass, intermediate variable that is overwritten for each superclass
        variable_names_tot_per_super = []

        first_super = True
        first_sub = True

        for ct, line in enumerate(lines):
            line_ls = line.lstrip()
            if line_ls.startswith('#'):
                # Skip any comments.
                continue

            leading_spaces = len(line) - len(line_ls)

            # Get Class Names #
            is_class = line.find("class") != -1 and (line[line.find("class") + 5]) == " "

            if is_class:
                split_line = line.split("class ")
                class_name = split_line[1].split(":")
                class_name = class_name[0]  # name of class on this line
                if split_line[0] == "":  # this means it is a top level class
                    superclass_names.append(class_name)
                    if subclass_names:  # checks to see if any subclasses have been encountered yet
                        # appends the subclass names in the previous superclass to the total subclass list
                        subclass_names_tot.append(subclass_names)
                    subclass_names = []  # resets the list of subclasses in one individual superclass

                    if variable_names_tot_per_super:  # checks to see if any arrays of variables need to be added
                        if variable_names:  # checks to see if any lists of variable names need to be added
                            # appends the variable names in the previous subclass to the total variable names list per superclass
                            variable_names_tot_per_super.append(variable_names)
                        variable_names = []  # resets the list of variable names in one individual subclass
                        # appends the variable names of the entire previous superclass to the total variable names list
                        variable_names_tot.append(variable_names_tot_per_super)
                    variable_names_tot_per_super = []

                elif split_line[0] == "    ":  # this means it is a subclass
                    subclass_names.append(class_name)
                    if variable_names:  # checkts to see if any variables have been encountered yet in this superclass
                        # appends the variable names in the previous subclass to the total variable names list per superclass
                        variable_names_tot_per_super.append(variable_names)
                    variable_names = []  # resets the list of variable names in one individual subclass

            # Get Variable Names #
            if leading_spaces == 8:
                var_name = line.split()[0]
                # append variable to variable name list for variables in a single subclass
                variable_names.append(var_name)

        # Append final variable names #
        if subclass_names:
            subclass_names_tot.append(subclass_names)

        if variable_names:
            variable_names_tot_per_super.append(variable_names)

        if variable_names_tot_per_super:
            variable_names_tot.append(variable_names_tot_per_super)

        #################################################################################################################
        # Finish Extracting Class and Variable Names from File #
        #################################################################################################################

        # Check For Alphabetization in Superclasses #
        sorted_superclass = superclass_names.copy()
        sorted_superclass.sort(key=str.casefold)
        superclass_is_sorted = sorted_superclass == superclass_names
        if not superclass_is_sorted:
            raise ValueError(
                f'The list of classes in the "{variables_file_loc}" file is not alphabetical. The class order should be: {sorted_superclass} but instead it is: {superclass_names}.'
            )

        # Check for Alphabetization in Subclasses #
        for idx, family in enumerate(subclass_names_tot):
            sorted_family = family.copy()
            sorted_family.sort(key=str.casefold)
            family_is_sorted = sorted_family == family
            if not family_is_sorted:
                raise ValueError(
                    f'The list of classes in "{superclass_names[idx]}" is not alphabetical. The class order should be: {sorted_family} but instead it is: {family}.'
                )

        # Check for Alphabetization in Variables #
        for super_cnt, var_per_super in enumerate(variable_names_tot):
            for sub_cnt, var_per_sub in enumerate(var_per_super):
                sorted_var_list = var_per_sub.copy()
                sorted_var_list.sort(key=str.casefold)
                var_list_is_sorted = sorted_var_list == var_per_sub
                if not var_list_is_sorted:
                    raise ValueError(
                        f'The list of variables in class "{subclass_names_tot[super_cnt][sub_cnt]}" within class "{superclass_names[super_cnt]}" is not alphabetical. The variable order should be: {sorted_var_list} but instead it is: {var_per_sub}.'
                    )


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
        old_to_new = [previous_var[diff_idx], current_var[diff_idx]]
        old_to_new_alphabetical = old_to_new.copy()
        old_to_new_alphabetical.sort(key=str.casefold)
        is_alphabetical = old_to_new == old_to_new_alphabetical
        previous_var = current_var

        if not is_alphabetical:
            out_of_order_vars.append(var_name)

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
        [key for key in (_MetaData if MetaData is None else MetaData)])

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

    names = []
    keys = vars(hierarchy).keys()  # get initial class keys
    # filter out keys that aren't for relevant variables
    keys = list(filter(filter_underscore, list(keys)))

    for key in keys:
        subclass_vars = vars(
            getattr(hierarchy, key)
        )  # grab dictionary of variables for the subclass
        # filter out keys that aren't for relevant variables
        subclass_keys = list(filter(filter_underscore, list(subclass_vars.keys())))

        for var_name in subclass_keys:
            names.append(
                subclass_vars[var_name]
            )  # add relevant variables to dictionary

    return names


class DuplicateHierarchy:

    class Design:
        CRUISE_MACH = 'mission:design:cruise_mach'
        RANGE = 'mission:design:range'

    class OperatingLimits:
        MAX_MACH = 'mission:design:cruise_mach'  # this is a duplicate


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
