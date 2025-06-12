import copy

from aviary.utils.compare_hierarchies import compare_hierarchies_to_merge, compare_inner_classes


def merge_attributes(base_class, merge_class, base_class_attributes, merge_class_attributes):
    """
    Adds unique attributes of merge_class to base_class.

    For all the attributes of merge_class that are not present in base_class we add them to base_class.
    Attributes present in both classes are ignored because if they are variables they have already
    been checked and found identical and if they are inner classes they will be addressed on a
    recursive call to this function. Attributes present only in base_class are ignored because we are
    adding to base_class and thus all of its features are automatically preserved.

    Parameters
    ----------
    base_class : class
        A class that is all or part of a variable hierarchy. This can be the top=level
        class in the hierarchy, or any inner class nested at any depth within that top-level class.

    merge_class : class
        A class that is all or part of a variable hierarchy. This can be the top=level
        class in the hierarchy, or any inner class nested at any depth within that top-level class.

    base_class_attributes : set of strings
        A set of the name of all attributes (either variables, inner classes, or both) of base_class.

    merge_class_attributes : set of strings
        A set of the name of all attributes (either variables, inner classes, or both) of merge_class.

    Returns
    -------
    base_class : class
        A class that contains the merged together attributes of base_class and merge_class, with
        the exception that inner class base_class and merge_class share which have diverging attributes inside
        of them are not necessarily included.

    Raises
    ------
    None
        No exceptions raised by this method, although other methods called within may raise exceptions.
    """
    merge_class_unique = (
        merge_class_attributes - base_class_attributes
    )  # attributes present only in merge_class

    for attr in merge_class_unique:
        setattr(base_class, attr, getattr(merge_class, attr))

    return base_class


def recursive_merge(overlapping_inners, base_class, merge_class):
    """
    Recursively compares all inner classes to an infinite depth and identifies mismatched string-named values.

    For all of the inner class names provided in overlapping_inner_classes this function calls compare_inner_classes
    and compares those inner classes recursively until it reaches the full depth to which outer_class_a and
    outer_class_b have any inner classes in common.

    Parameters
    ----------
    overlapping_inner_classes : set of strings
        This is a set of strings where each string is the name of an inner class that outer_class_a
        and outer_class_b have in common.

    outer_class_a : class
        A class that is all or part of a variable hierarchy. This can be the top-level class in the
        hierarchy, or any inner class nested at any depth within that top-level class.

    outer_class_b : class
        A class that is all or part of a variable hierarchy. This can be the top-level class in the
        hierarchy, or any inner class nested at any depth within that top-level class.

    Returns
    -------
    None
        No variables returned by this method.

    Raises
    ------
    None
        No exceptions raised by this method, although other methods called within may raise exceptions.
    """
    for overlapping_class_name in overlapping_inners:
        overlapping_inner_class_base = getattr(base_class, overlapping_class_name)
        overlapping_inner_class_merge = getattr(merge_class, overlapping_class_name)
        [
            overlapping_second_inners,
            vars_base_inner,
            vars_merge_inner,
            inners_base_inner,
            inners_merge_inner,
        ] = compare_inner_classes(
            overlapping_inner_class_base, overlapping_inner_class_merge, show_all=True
        )
        merge_attributes(
            overlapping_inner_class_base,
            overlapping_inner_class_merge,
            vars_base_inner,
            vars_merge_inner,
        )
        merge_attributes(
            overlapping_inner_class_base,
            overlapping_inner_class_merge,
            inners_base_inner,
            inners_merge_inner,
        )
        recursive_merge(
            overlapping_second_inners, overlapping_inner_class_base, overlapping_inner_class_merge
        )


def merge_two_hierarchies(base_hierarchy, hierarchy_b):
    """
    Merge two variable hierarchies together by adding the second into the first.

    Add the attributes (variables and inner classes) of two variable hierarchies together, so that
    the attributes that are unique to each hierarchy are combined in the resultant hierarchy. This
    is accomplished by adding on to the first of the two provided hierarchies, and returning it.

    Parameters
    ----------
    base_hierarchy : class
        An Aviary variable hierarchy. This hierarchy will function as the 'base' hierarchy,
        which is the hierarchy that has attributes added onto it from another hierarchy in order to combine
        the attributes of the base and the other hierarchy.

    hierarchy_b : class
        An Aviary variable hierarchy. This hierarchy will function as the 'auxiliary' hierarchy,
        which is the hierarchy whose unique attributes are added onto a base in order to combine the
        attributes of the base and the auxiliary.

    Returns
    -------
    base_hierarchy : class
        An Aviary variable hierarchy which includes both the attributes of the inputted base_hierarchy
        and hierarchy_b. This is the same object as the inputted base_hierarchy and has been updated through
        mutability.

    Raises
    ------
    None
        No exceptions raised by this method, although other methods called within may raise exceptions.
    """
    [overlapping_inners, merged_vars, b_vars, merged_inners, b_inners] = compare_inner_classes(
        base_hierarchy, hierarchy_b, show_all=True
    )
    # this adds the variables of hierarchy_b which are not in base_hierarchy to the overall merged hierarchy
    base_hierarchy = merge_attributes(base_hierarchy, hierarchy_b, merged_vars, b_vars)
    # this adds the inner classes of hierarchy_b which are not in base_hierarchy to the overall merged hierarchy
    base_hierarchy = merge_attributes(base_hierarchy, hierarchy_b, merged_inners, b_inners)
    recursive_merge(overlapping_inners, base_hierarchy, hierarchy_b)

    return base_hierarchy


def merge_hierarchies(hierarchies_to_merge):
    """
    Combines all provided variable hierarchies into one unified variable hierarchy.

    Performs checks on the user-provided list of variable hierarchies in order to ensure that they are
    compatible for merge. Ensures that they are not derived from different superclasses, and that they do
    not have conflicting values. Assuming all checks pass, the provided hierarchies are combined into a
    single hierarchy.

    Parameters
    ----------
    hierarchies_to_merge : list of classes
        This is a list of all the variable hierarchies which should be merged into a single hierarchy. This list should not include hierarchies of multiple types (i.e. an av.Mission hierarchy extension should not be mixed with an av.Aircraft hierarchy extension)

    Returns
    -------
    merged_hierarchy : class
        Aviary variable hierarchy that includes all the variables present in the inputted hierarchy list. Duplicates have been removed, and conflicting duplicates are not possible.

    Raises
    ------
    ValueError
        Raises an exception if the list of inputted variable hierarchies includes hierarchies that have been subclassed from different superclasses.
    """
    compare_hierarchies_to_merge(hierarchies_to_merge)
    subclass_type = None
    subclass_hierarchy = None
    for hierarchy in hierarchies_to_merge:  # check that there are not hierarchies subclassing from different classes that we are attempting to merge
        # checks if the given hierarchy is the first subclass of the hierarchies
        if (len(hierarchy.__mro__) > 2) and (subclass_type is None):
            # gets highest level superclass of the class before "class" itself
            subclass_type = copy.deepcopy(hierarchy.__mro__[-2])
            subclass_hierarchy = copy.deepcopy(hierarchy)
        elif len(hierarchy.__mro__) > 2:  # checks if the given hierarchy is a subclass
            if hierarchy.__mro__[-2] != subclass_type:
                raise ValueError(
                    f"You have attempted to merge together variable hierarchies that subclass from different superclasses. '{subclass_hierarchy.__qualname__}' is a subclass of '{subclass_type}' and '{hierarchy.__qualname__}' is a subclass of '{hierarchy.__mro__[-2]}'."
                )

    if (
        subclass_hierarchy is not None
    ):  # ensure that we build on the subclassed hierarchy if we have one, that way the super class information is not lost
        merged_hierarchy = subclass_hierarchy
    else:
        merged_hierarchy = copy.deepcopy(hierarchies_to_merge[0])

    for hierarchy in hierarchies_to_merge:
        merged_hierarchy = merge_two_hierarchies(merged_hierarchy, hierarchy)

    return merged_hierarchy
