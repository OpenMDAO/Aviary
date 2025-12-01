import warnings
from enum import Enum

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import param

from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.options import list_options as list_options_func
from aviary.utils.preprocessors import preprocess_options
from aviary.validation_cases.validation_data.flops_data.FLOPS_Test_Data import (
    FLOPS_Lacking_Test_Data,
    FLOPS_Test_Data,
)
from aviary.variable_info.functions import extract_options
from aviary.variable_info.variables import Aircraft

Version = Enum('Version', ['ALL', 'TRANSPORT', 'ALTERNATE', 'BWB'])


def do_validation_test(
    prob: om.Problem,
    case_name: str,
    input_validation_data: AviaryValues,
    output_validation_data: AviaryValues,
    input_keys: list,
    output_keys: list,
    aviary_option_keys: list = None,
    tol: float = 1.0e-4,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    method: str = 'cs',
    step: float = None,
    check_values: bool = True,
    check_partials: bool = True,
    excludes: list = None,
    list_inputs: bool = False,
    list_outputs: bool = False,
    list_options: bool = False,
):
    """
    Runs a validation test with user-supplied validation data.

    Parameters
    ----------
    prob : om.Problem
        An instantiated and set-up Problem.
    case_name : str
        Name of the case being run.
    input_validation_data : AviaryValues
        Input variables and their values to use when running the problem.
        The object must contain at a minimum all the input variables listed
        in input_keys.
    output_validation_data : AviaryValues
        Output variables and their values to which output from the problem
        will be compared. The object must contain at a minimum all the
        output variables listed in output_keys. Note that input_validation_data
        and output_validation_data may be the same object.
    input_keys : str, or iter of str
        List of input variables whose values will be transferred from
        the input validation data in order to run the problem.
    output_keys : str, or iter of str
        List of output variables whose values will be looked up in
        the output validation data and compared against the outputs from the
        problem.
    aviary_option_keys: str, or iter of str
        List of aviary_options keys whose values will be looked up and
        listed in the options printout. If None, all items in aviary_options
        will be listed.
    tol : float
        Relative tolerance for comparing problem outputs against
        validation data. The default is 1.0e-4.
    atol : float
        Absolute tolerance for checking partial derivative calculations. The
        default is 1.0e-12.
    rtol : float
        Relative tolerance for checking partial derivative calculations. The
        default is 1.0e-12.
    method : str
        Type of differencing to use. The default is "cs".
    step : float
        Step size for approximation. Default is None, which means 1e-6 for 'fd' and 1e-40 for
        'cs'.
    check_values : bool
        If true, check output values against validation data. The default is
        true.
    check_partials : bool
        If true, check partial derivative calculations. The default is true.
    excludes : None or list_like
        List of glob patterns for pathnames to exclude from the partial derivative check.
        Default is None, which excludes nothing.
    list_options: bool
        If True, values of options for all components in the model will be listed
        on standard output. Default is False.
    list_inputs: bool
        If True, prob.model.list_inputs() will be called after the model is run.
        Default is False.
    list_outputs: bool
        If True, prob.model.list_outputs() will be called after the model is run.
        Default is False.
    """
    input_key_list = _assure_is_list(input_keys)
    output_key_list = _assure_is_list(output_keys)
    aviary_option_key_list = _assure_is_list(aviary_option_keys)

    for key in input_key_list:
        if key in input_validation_data:
            data = input_validation_data
        else:
            data = output_validation_data
        desired, units = data.get_item(key)
        prob.set_val(key, desired, units)

    prob.run_model()

    if list_options:
        list_options_func(prob.model, aviary_keys=aviary_option_key_list)

    if list_inputs:
        prob.model.list_inputs()

    if list_outputs:
        prob.model.list_outputs()

    if check_values:
        for key in output_key_list:
            desired, units = output_validation_data.get_item(key)
            actual = prob.get_val(key, units)
            try:
                assert_near_equal(actual, desired, tol)
            except ValueError as err:
                raise ValueError(f'ValueError for key = {key}') from err

    if check_partials:
        partial_data = prob.check_partials(
            out_stream=None, method=method, step=step, excludes=excludes
        )
        assert_check_partials(partial_data, atol=atol, rtol=rtol)


def flops_validation_test(
    prob: om.Problem,
    case_name: str,
    input_keys: list,
    output_keys: list,
    aviary_option_keys: list = None,
    version: Version = Version.ALL,
    tol: float = 1.0e-4,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    method: str = 'cs',
    step: float = None,
    check_values: bool = True,
    check_partials: bool = True,
    excludes: list = None,
    list_inputs: bool = False,
    list_outputs: bool = False,
    list_options: bool = False,
    flops_inputs=None,
    flops_outputs=None,
):
    """
    Set a model, runs the model and runs a validation test using FLOPS validation data.

    Parameters
    ----------
    prob : om.Problem
        An instantiated and set-up Problem.
    case_name : str
        Name of the case being run. Validation data will be looked up from
        the corresponding case in the FLOPS validation data collection.
    input_keys : str, or iter of str
        List of input variables whose values will be transferred from
        the input validation data in order to run the problem.
    output_keys : str, or iter of str
        List of output variables whose values will be looked up in
        the output validation data and compared against the outputs from the
        problem.
    aviary_option_keys: str, or iter of str
        List of aviary_options keys whose values will be looked up and
        listed in the options printout. If None, all items in aviary_options
        will be listed.
    version: Version
        If this is a FLOPS-based mass analysis test, version specifies which
        version of the mass equations being tested. Currently, there is no
        BWB validation data so output values will not be checked.
        Default is ALL.
    tol : float
        Relative tolerance for comparing problem outputs against
        validation data. The default is 1.0e-4.
    atol : float
        Absolute tolerance for checking partial derivative calculations. The
        default is 1.0e-12.
    rtol : float
        Relative tolerance for checking partial derivative calculations. The
        default is 1.0e-12.
    method : str
        Type of differencing to use. The default is "cs".
    check_values : bool
        If true, check output values against validation data. The default is
        true. This option may be overridden in cases where validation data are
        not available; currently this applies to BWB aircraft.
    check_partials : bool
        If true, check partial derivative calculations. The default is true.
    excludes : None or list_like
        List of glob patterns for pathnames to exclude from the partial derivative check.
        Default is None, which excludes nothing.
    list_options: bool
        If True, values of options for all components in the model will be listed
        on standard output. Default is False.
    list_inputs: bool
        If True, prob.model.list_inputs() will be called after the model is run.
        Default is False.
    list_outputs: bool
        If True, prob.model.list_outputs() will be called after the model is run.
        Default is False.
    flops_inputs: None or AviaryValues
        Allows a custom set of inputs to be tested. Default is None, which reads
        data from FLOPS_Test_Data with key case_name.
    flops_outputs: None or AviaryValues
        Allows a custom set of outputs to be tested. Default is None, which reads
        data from FLOPS_Test_Data with key case_name.
    """
    if not isinstance(version, Version):
        raise TypeError('parameter "version" must be of enumeration type "Version"')

    if flops_inputs is None and flops_outputs is None:
        flops_data = FLOPS_Test_Data[case_name]
        flops_inputs = flops_data['inputs'].deepcopy()
        flops_outputs = flops_data['outputs'].deepcopy()

    if (
        version is Version.TRANSPORT
        and flops_inputs.get_val(Aircraft.Design.USE_ALT_MASS)
        or version is Version.ALTERNATE
        and not flops_inputs.get_val(Aircraft.Design.USE_ALT_MASS)
    ):
        return

    # TODO: Currently no BWB validation data.
    # For BWBs, skip the validation test, but do check the partials.
    check_values_in = check_values
    check_values = check_values and version is not Version.BWB
    if not check_values and check_values_in:
        warnings.warn('Not checking values because validation data not available.')

    do_validation_test(
        prob=prob,
        case_name=case_name,
        input_validation_data=flops_inputs,
        output_validation_data=flops_outputs,
        input_keys=input_keys,
        output_keys=output_keys,
        aviary_option_keys=aviary_option_keys,
        tol=tol,
        atol=atol,
        rtol=rtol,
        method=method,
        step=step,
        check_values=check_values,
        check_partials=check_partials,
        excludes=excludes,
        list_options=list_options,
        list_inputs=list_inputs,
        list_outputs=list_outputs,
    )


def get_flops_data(case_name: str, keys: str = None, preprocess: bool = False) -> AviaryValues:
    """
    Returns an AviaryValues object containing input and output data for
    the named FLOPS validation case.

    Parameters
    ----------
    case_name : str
        Name of the case being run. Validation data will be looked up from
        the corresponding case in the FLOPS validation data collection.
    keys : str, or iter of str
        List of variables whose values will be transferred from the validation data.
        The default is all variables.
    preprocess: bool
        If true, the input data will be passed through preprocess_options() to
        fill in any missing options before being returned. The default is False.
    """
    flops_data_copy: AviaryValues = get_flops_inputs(case_name, preprocess=preprocess)
    flops_data_copy.update(get_flops_outputs(case_name))
    if keys is None:
        return flops_data_copy
    keys_list = _assure_is_list(keys)

    return AviaryValues({key: flops_data_copy.get_item(key) for key in keys_list})


def get_flops_inputs(case_name: str, keys: str = None, preprocess: bool = False) -> AviaryValues:
    """
    Returns an AviaryValues object containing input data for the named FLOPS validation case.

    Parameters
    ----------
    case_name : str
        Name of the case being run. Input data will be looked up from
        the corresponding case in the FLOPS validation data collection.
    keys : str, or iter of str
        List of variables whose values will be transferred from the input data.
        The default is all variables.
    preprocess: bool
        If true, the input data will be passed through preprocess_options() to
        fill in any missing options before being returned. The default is False.
    """
    try:
        flops_data: dict = FLOPS_Test_Data[case_name]
    except KeyError:
        flops_data: dict = FLOPS_Lacking_Test_Data[case_name]

    flops_inputs_copy: AviaryValues = flops_data['inputs'].deepcopy()
    if preprocess:
        preprocess_options(
            flops_inputs_copy,
            engine_models=[build_engine_deck(flops_inputs_copy)],
            verbosity=0,
        )
    if keys is None:
        return flops_inputs_copy
    keys_list = _assure_is_list(keys)

    return AviaryValues({key: flops_inputs_copy.get_item(key) for key in keys_list})


def get_flops_options(case_name: str, keys: str = None, preprocess: bool = False) -> AviaryValues:
    """
    Returns a dictionary containing options for the named FLOPS validation case.

    Parameters
    ----------
    case_name : str
        Name of the case being run. Input data will be looked up from
        the corresponding case in the FLOPS validation data collection.
    keys : str, or iter of str
        List of variables whose values will be transferred from the input data.
        The default is all variables.
    preprocess: bool
        If true, the input data will be passed through preprocess_options() to
        fill in any missing options before being returned. The default is False.
    """
    try:
        flops_data: dict = FLOPS_Test_Data[case_name]
    except KeyError:
        flops_data: dict = FLOPS_Lacking_Test_Data[case_name]

    flops_inputs_copy: AviaryValues = flops_data['inputs'].deepcopy()
    if preprocess:
        preprocess_options(flops_inputs_copy, engine_models=[build_engine_deck(flops_inputs_copy)])

    if keys is None:
        options = extract_options(flops_inputs_copy)
    else:
        options = extract_options(keys)

    return options


def get_flops_outputs(case_name: str, keys: str = None) -> AviaryValues:
    """
    Returns an AviaryValues object containing output data for the named FLOPS validation case.

    Parameters
    ----------
    case_name : str
        Name of the case being run. Output data will be looked up from
        the corresponding case in the FLOPS validation data collection.
    keys : str, or iter of str
        List of variables whose values will be transferred from the output data.
        The default is all variables.
    """
    flops_data: dict = FLOPS_Test_Data[case_name]
    flops_outputs_copy: AviaryValues = flops_data['outputs'].deepcopy()
    if keys is None:
        return flops_outputs_copy
    keys_list = _assure_is_list(keys)

    return AviaryValues({key: flops_outputs_copy.get_item(key) for key in keys_list})


def get_flops_case_names(omit: list = None, only: list = None) -> list:
    """
    Returns a list of case names in the FLOPS validation database.

    Parameters
    ----------
    omit : str or list
        The case name or list of case names to omit from the returned list.
    only : str or list
        The case name or list of case names to be considered for inclusion
        in the returned list. Note that parameters only and omit cannot be
        used together.
    """
    if omit is not None and only is not None:
        raise ValueError('cannot use both only and omit keywords')

    omit_list = _assure_is_list(omit)
    only_list = _assure_is_list(only, backup=FLOPS_Test_Data)

    return [key for key in FLOPS_Test_Data if key in only_list and key not in omit_list]


def print_case(testcase_func, param_num, param: param):
    """
    Returns a formatted case name for unit testing with decorator @parameterized.expand().
    It is intended to be used when expand() is called with a list of strings
    representing test case names.

    Parameters
    ----------
    testcase_func : Any
        This parameter is ignored.
    param_num : Any
        This parameter is ignored.
    param : param
        The param object containing the case name to be formatted.
    """
    return 'test_case_' + param.args[0]


def _assure_is_list(keys, backup=None):
    if isinstance(keys, str):
        return [keys]
    elif not keys:
        if backup is None:
            backup = []
        return backup
    return keys
