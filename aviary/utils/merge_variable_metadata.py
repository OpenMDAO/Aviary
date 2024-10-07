import numpy as np


def almost_equal(a, b, rel_tol=1e-8, abs_tol=0.0):
    """check if two floats, or two ndarray, or two dictionary are equal. Return True if they are equal."""
    if isinstance(a, (float, np.float32, np.float64)) and isinstance(b, (float, np.float32, np.float64)):
        return np.isclose(a, b, rtol=rel_tol, atol=abs_tol)

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.allclose(a, b, rtol=rel_tol, atol=abs_tol)

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        for key in a.keys():
            if not almost_equal(a[key], b[key]):
                return False
        return True

    return a == b


def merge_2_meta_data_dicts(dict1, dict2):
    '''
    Combines metadata from two dictionaries into a single dictionary containing metadata of both.

    Performs a check to ensure that the two dictionaries don't have the same variable with conflicting
    metadata. Assuming that check passes, the variables and their associated metadata from both
    dictionaries are combined into one dictionary.

    Parameters
    ----------
    dict1, dict2 : dict
        Dictionaries with keys of Aviary variables and values of their associated metadata.

    Returns
    -------
    new_dict : dict
        Single dictionary with all the combined entries from the two input Aviary metadata dictionaries.

    Raises
    ------
    ValueError
        Raises error if dict1 and dict2 have differing metadata for the same variable.
    '''

    dict1_keys = set(dict1.keys())  # get keys of provided dictionaries
    dict2_keys = set(dict2.keys())  # get keys of provided dictionaries
    # get keys that are unique to each dictionary
    all_unique_keys = (dict1_keys | dict2_keys) - (dict1_keys & dict2_keys)
    # get all keys that are repeated in both dictionaries
    all_duplicate_keys = dict1_keys & dict2_keys
    # merge the two dictionaries, and allow the repeated keys to be overwritten
    merged = {**dict1, **dict2}

    dict_of_unique_variables = dict(
        # get a dictionary of all the unique keys (and their values) from the input dictionaries
        map(lambda key: (key, merged.get(key, None)), all_unique_keys))

    # initialize a dictionary of the values with duplicated keys in both input dictionaries
    dict_of_duplicate_variables = {}

    for key in sorted(all_duplicate_keys):
        # check that the metadata in both dictionaries associated with the same key is the same
        value1, value2 = dict1[key], dict2[key]
        # throw an error if the dicts have the same key with different metadata
        if not almost_equal(value1, value2):
            raise ValueError(
                f'You have attempted to merge metadata dictionaries that contain the same variable with different metadata. The offending variable present in multiple dictionaries is "{key}".')

        # add the key and metadata to the dictionary of duplicate variables
        dict_of_duplicate_variables[key] = dict1[key]

    # merge together the final dictionary
    new_dict = {**dict_of_duplicate_variables, **dict_of_unique_variables}

    return new_dict


def merge_meta_data(dicts_to_merge):
    '''
    Merges the metadata of multiple Aviary metadata dictionaries into a single metadata dictionary.

    Checks the metadata of all the provided Aviary metadata dictionaries to see if there are identical
    variables with conflicting metadata. Assuming this check passes, the input dictionaries are merged
    together into a single dictionary containing all their information.

    Parameters
    ----------
    dicts_to_merge : list of dicts
        List of Aviary metadata dictionaries to be merged.

    Returns
    -------
    merged_dict : dict
        Single Aviary metadata dictionary with all the information of the inputted metadata dictionaries.

    Raises
    ------
    None
        No exceptions raised by this method, although other methods called within may raise exceptions.
    '''
    merged_dict = dicts_to_merge[0]

    # iterate through all the dictionaries and merge them together into one, two by two
    for dict_item in dicts_to_merge:
        merged_dict = merge_2_meta_data_dicts(merged_dict, dict_item)

    return merged_dict
