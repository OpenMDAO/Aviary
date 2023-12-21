from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft
from aviary.subsystems.propulsion.engine_deck import EngineDeck


def get_option_defaults(engine=True, meta_data=_MetaData) -> AviaryValues:
    """
    Returns a deep copy of the collection of all options for which default values exist.

    Parameters
    ----------
    engine : bool
        If true, the collection includes the default engine model
    meta_data : dict
        Dictionary containing metadata for the options. If None, Aviary's built-in
        metadata will be used.
    """

    option_defaults = AviaryValues()

    # Load all variables marked as options in the MetaData
    for key in meta_data:
        var = meta_data[key]
        if var['option'] and var['default_value'] is not None:
            option_defaults.set_val(key, var['default_value'], var['units'])

    if engine:
        engine_options = option_defaults.deepcopy()
        engine_options.set_val(Aircraft.Engine.DATA_FILE,
                               get_path('models/engines/turbofan_23k_1.deck'))
        engine_options.set_val(Aircraft.Engine.SCALE_FACTOR,
                               meta_data[Aircraft.Engine.SCALE_FACTOR]['default_value'])
        engine_deck = EngineDeck(options=engine_options)
        preprocess_propulsion(option_defaults, [engine_deck])

    return option_defaults


def is_option(key, meta_data=_MetaData) -> bool:
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
