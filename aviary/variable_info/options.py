from aviary.variable_info.variable_meta_data import CoreMetaData


def is_option(key, meta_data=CoreMetaData) -> bool:
    """
    Returns True if the variable is defined as an option in the MetaData.

    Parameters
    ----------
    key: str
        Name of the variable to be checked
    meta_data : dict
        Dictionary containing metadata for the variable. If None, Aviary's built-in
        metadata will be used.
    """
    return meta_data[key]['option']
