def add_meta_data(
    key: str,
    meta_data: dict,
    units='unitless',
    default_value=0.0,
    types=None,
    multivalue: bool = False,
    option: bool = False,
    desc: str = None,
    historical_name=None,
    _check_unique=True,
):
    """
    Add new meta data associated with variables in the Aviary data hierarchy.

    Parameters
    ----------
    key : str
        Aviary variable name

    meta_data : dict
        dictionary of meta data to add the variable to

    units : str or None
        units of measure

    desc : str
        brief description of the variable

    default_value : any
        in context, the Aviary value assumed if the variable is missing from
        options and/or inputs

        Note, a default value of `None` indicates that the variable is
        optional, but that there is no default.

    types : type, optional
        gives the allowable type(s) of the variable in the aviary API. if `multivalue` is
        True, this specifies which types individual elements in the list can be. If
        `types` is not provided, the type of `default_value` is used. Numerical data
        for `default_value` is assumed to be a float in this case.

    option : bool
        indicates that this variable is an option, rather than a normal input

    multivalue : bool
        when True, the variable can become an iterable (list, tuple, numpy array) of
        elements whose type is in `types`.

    historical_name : dict or None
        dictionary of names that the variable held in prior codes

        Example: {"FLOPS":"WTIN.WNGWT", "LEAPS1": "aircraft.inputs.wing_weight", "GASP":
        "INGASP.WWGHT"}

        NAMELIST naming convention
        &<function_name>.<namelist_name>.<var_name>

        Example: &DEFINE.CONFIN.GW
            represents the GW variable of the CONFIN namelist as defined in
            the DEFINE subroutine

        COMMON block naming convention, including aliases:
        <block_name>.<var_name>

        Example: CONFIG.GW
            represents the GW variable of the CONFIG common block

        Local variable naming convention, including equivalence statements, parameters,
        and other local declarations:
        ~<function_name>.<var_name>

        Example: ~ANALYS.GWTOL
            represents the GWTOL variable of the ANALYS subroutine

    _check_unique : bool
        private use only flag that tells whether to check the meta_data for the
        pre-existing presence of the provided key. This should only be set to false when
        update_meta_data is the calling function.

    Returns
    -------
    None
        No variables returned by this method.

    Raises
    ------
    None
        No exceptions raised by this method, although other methods called within may
        raise exceptions.
    """
    if key in meta_data and _check_unique:
        raise ValueError(
            f'You added the variable {key} to a variable metadata dictionary via the '
            f'add_meta_data function, but {key} already was present in the dictionary. '
            'If you are sure you want to overwrite this variable, call the '
            'update_meta_data function instead.'
        )

    if units is None:
        units = 'unitless'

    if types is None and default_value is not None:
        try:
            # Try casting to float, as it is common for users to intend use of floats but
            # not include decimal points in default_value
            default_value = float(default_value)
        except (ValueError, TypeError):
            pass  # default_value is not a numerical value
        types = type(default_value)

    meta_data[key] = {
        'historical_name': historical_name,
        'units': units,
        'desc': desc,
        'option': option,
        'default_value': default_value,
        'types': types,
        'multivalue': multivalue,
    }


def update_meta_data(
    key: str,
    meta_data: dict,
    units='unitless',
    desc: str = None,
    default_value=0.0,
    option: bool = False,
    types=None,
    multivalue: bool = False,
    historical_name=None,
):
    """
    Update existing meta data associated with variables in the Aviary data hierarchy.

    Parameters
    ----------
    key : str
        Aviary variable name

    meta_data : dict
        dictionary of meta data to add the variable to

    units : str or None
        units of measure

    desc : str
        brief description of the variable

    default_value : Any
        in context, the Aviary value assumed if the variable is missing from
        options and/or inputs

        Note, a default value of `None` indicates that the variable is
        optional, but that there is no default.

    option : bool
        indicates that this variable is an option, rather than a normal input

    types : type
        gives the allowable type(s) of the variable

    multivalue : bool
        when True, the variable can become a list of elements whose type is in types.
        This is mainly used when there are multiple engine types.

    historical_name : dict or None
        dictionary of names that the variable held in prior codes

        Example: {"FLOPS":"WTIN.WNGWT", "LEAPS1": "aircraft.inputs.wing_weight", "GASP":
        "INGASP.WWGHT"}

        NAMELIST naming convention
        &<function_name>.<namelist_name>.<var_name>

        Example: &DEFINE.CONFIN.GW
            represents the GW variable of the CONFIN namelist as defined in
            the DEFINE subroutine

        COMMON block naming convention, including aliases:
        <block_name>.<var_name>

        Example: CONFIG.GW
            represents the GW variable of the CONFIG common block

        Local variable naming convention, including equivalence statements, parameters,
        and other local declarations:
        ~<function_name>.<var_name>

        Example: ~ANALYS.GWTOL
            represents the GWTOL variable of the ANALYS subroutine

    Returns
    -------
    None
        No variables returned by this method.

    Raises
    ------
    None
        No exceptions raised by this method, although other methods called within may
        raise exceptions.
    """
    if key not in meta_data:
        raise ValueError(
            f'You provided the variable {key} to a variable metadata dictionary via the '
            f'update_meta_data function, but {key} does not exist in the dictionary. If '
            'you are sure you want to add this variable to the dictionary, call the '
            'add_meta_data function instead.'
        )

    add_meta_data(
        key=key,
        meta_data=meta_data,
        units=units,
        desc=desc,
        default_value=default_value,
        option=option,
        types=types,
        multivalue=multivalue,
        historical_name=historical_name,
        _check_unique=False,
    )
