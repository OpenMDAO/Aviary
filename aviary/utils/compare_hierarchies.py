from aviary.variable_info.variables import Aircraft as _Aircraft
from aviary.variable_info.variables import Mission as _Mission


def compare_inner_classes(class1, class2, show_all=False):
    """
    Compare two nested class hierarchies and return a set of shared inner-classes.

    Summary:
    This function takes in two classes that both are part of variable hierarchies and may contain
    inner-classes or variables with string-named values. This function compares those two classes 
    and returns a set of inner classes that have the same name in both inputted classes. It will throw 
    an error if the two classes have the same variable with a different string-named value.

    Parameters
    ----------
    class1 : class
        A class that is all or part of a variable hierarchy. This can be the top-level 
        class in the hierarchy, or any inner class nested at any depth within that top-level class.
    class2 : class
        A class that is all or part of a variable hierarchy. This can be the top-level
        class in the hierarchy, or any inner class nested at any depth within that top-level class.
    show_all : bool, optional
        Flag to tell the function to return the sets of variables and inner classes for
        each provided class.

    Returns
    -------
    overlapping_inner_classes : set of strings
        A set of string names of all the inner classes which are common between the two
        input classes.
    class1_vars_set : set of strings, optional
        Set of string names of the variables belonging to class1, optional return based 
        on show_all flag.
    class2_vars_set : set of strings, optional
        Set of string names of the variables belonging to class2, optional return based
        on show_all flag.
    class1_inner_classes_set : set of strings, optional
        Set of the string names of inner classes belonging to class1, optional return based
        on show_all flag.
    class2_inner_classes_set : set of strings, optional
        Set of the string names of inner classes belonging to class2, optional return based
        on show_all flag.

    Raises
    ------
    ValueError
        If the two input classes both have a variable with the same variable name but 
        different string-named value.
    """
    class1_vars_inner_classes = vars(class1)
    class2_vars_inner_classes = vars(class2)

    class1_vars = []
    class1_inner_classes = []
    class2_vars = []
    class2_inner_classes = []

    # separate out a list of string names of the variables belonging to class1, and the inner classes belonging to class1
    for key in class1_vars_inner_classes.keys():
        # just checks if it is a class
        if type(class1_vars_inner_classes[key]) == type(class1):
            class1_inner_classes.append(key)
        elif (type(class1_vars_inner_classes[key]) == str) and not (key == '__module__') and not (key == '__doc__'):
            class1_vars.append(key)

    # separate out a list of string names of the variables belonging to class2, and the inner classes belonging to class2
    for key in class2_vars_inner_classes.keys():
        if type(class2_vars_inner_classes[key]) == type(class2):
            class2_inner_classes.append(key)
        elif (type(class2_vars_inner_classes[key]) == str) and not (key == '__module__') and not (key == '__doc__'):
            class2_vars.append(key)

    class1_vars_set = set(class1_vars)
    class2_vars_set = set(class2_vars)
    # get set of the variables in the provided classes that have the same name
    overlapping_vars = class1_vars_set & class2_vars_set
    for var in overlapping_vars:  # go through overlapping variables and check that they have the same value associated with them
        value1 = getattr(class1, var)  # value of the variable in the first class
        value2 = getattr(class2, var)  # value of the variable in the second class

        if value1 != value2:
            raise ValueError(
                f"You have attempted to merge two variable hierarchies together that have the same variable with a different string name associated to it. The offending variable is '{var}'. In '{class1.__qualname__}' it has a value of '{value1}' and in '{class2.__qualname__}' it has a value of '{value2}'.")

    class1_inner_classes_set = set(class1_inner_classes)
    class2_inner_classes_set = set(class2_inner_classes)
    overlapping_inner_classes = class1_inner_classes_set & class2_inner_classes_set

    if show_all:
        return (overlapping_inner_classes, class1_vars_set, class2_vars_set, class1_inner_classes_set, class2_inner_classes_set)

    else:
        return (overlapping_inner_classes)


def recursive_comparison(overlapping_inner_classes, outer_class_a, outer_class_b):
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

    Exceptions
    ----------
    No exceptions explicitly raised by this function, although called functions may raise exceptions.
    """
    for overlapping_class_name in overlapping_inner_classes:
        overlapping_inner_class_a = getattr(outer_class_a, overlapping_class_name)
        overlapping_inner_class_b = getattr(outer_class_b, overlapping_class_name)
        overlapping_second_inner_classes = compare_inner_classes(
            overlapping_inner_class_a, overlapping_inner_class_b)
        recursive_comparison(overlapping_second_inner_classes,
                             overlapping_inner_class_a, overlapping_inner_class_b)


def compare_hierarchies_to_merge(hierarchies_to_merge):
    """
    Compares variable hierarchies to ensure there are no string-valued variable conflicts.

    For all the variable hierarchies provided in hierarchies_to_merge this function compares each
    hierarchy with every other hierarchy as well as the Aviary core aircraft and mission hierarchies
    to ensure there are no string-valued variable conflicts within the same class or inner class to
    and infinite depth.

    Parameters
    ----------
    hierarchies_to_merge : list of classes
        This is a list of variable hierarchy classes which will be compared for merge compatibility
        with one another.

    Returns
    -------
    None

    Raises
    ------
    No explicit exceptions are raised by this function, although called functions may raise exceptions.
    """

    for hierarchy in hierarchies_to_merge:
        # check if hierarchy has developed conflicts with original hierarchy
        if issubclass(hierarchy, _Aircraft):
            overlap = compare_inner_classes(hierarchy, _Aircraft)
            recursive_comparison(overlap, hierarchy, _Aircraft)

        # check if hierarchy has developed conflicts with original hierarchy
        if issubclass(hierarchy, _Mission):
            overlap = compare_inner_classes(hierarchy, _Mission)
            recursive_comparison(overlap, hierarchy, _Mission)

        for hierarchy2 in hierarchies_to_merge:
            if hierarchy != hierarchy2:
                overlap = compare_inner_classes(hierarchy, hierarchy2)
                recursive_comparison(overlap, hierarchy, hierarchy2)
