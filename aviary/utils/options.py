from openmdao.core.system import System

from aviary.utils.aviary_values import AviaryValues, get_keys


def list_options(model: System, aviary_keys: list = None):
    """
    Lists all option values in the provided model. All top-level option values will
    be listed, and items in model.options['aviary_options'] will also be listed.
    A list of keys may be provided to limit the list of items in aviary_options.

    Parameters
    ----------
    model : System
        A model.
    aviary_keys: iter of str
        List of aviary_options keys whose values will be looked up and
        listed in the options printout. If None, all items in
        model.options['aviary_options'] will be listed.
    """
    print('\nOptions:\n')
    for subsystem in model.system_iter():
        if subsystem.name == '_auto_ivc':
            continue
        print(subsystem.name)
        for (key, obj) in subsystem.options.items():
            if isinstance(obj, AviaryValues):
                aviary_options = obj
                print('  aviary_options:')
                if isinstance(aviary_keys, list):
                    keys = aviary_keys
                else:
                    keys = get_keys(aviary_options)
                for key in keys:
                    (val, units) = aviary_options.get_item(key)
                    if units == 'unitless':
                        print(f'    {key} = {val}')
                    else:
                        print(f'    {key} = {val} {units}')
            else:
                print(f'  {key} = {str(obj)[0:80]}')
    print()
