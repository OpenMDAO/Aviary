import warnings

import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.named_values import get_keys
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Settings
from aviary.utils.test_utils.variable_test import get_names_from_hierarchy
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData


def preprocess_options(aviary_options: AviaryValues, **kwargs):
    """
    Run all preprocessors on provided AviaryValues object

    Parameters
    ----------
    aviary_options : AviaryValues
        Options to be updated
    """
    try:
        engine_models = kwargs['engine_models']
    except KeyError:
        engine_models = None

    preprocess_crewpayload(aviary_options)
    preprocess_propulsion(aviary_options, engine_models)


def preprocess_crewpayload(aviary_options: AviaryValues):
    """
    Calculates option values that are derived from other options, and are not direct inputs.
    This function modifies the entries in the supplied collection, and for convenience also
    returns the modified collection.
    """
    verbosity = aviary_options.get_val(Settings.VERBOSITY)
    pax_provided = False
    design_pax_provided = False

    pax_keys = [
        Aircraft.CrewPayload.NUM_PASSENGERS,
        Aircraft.CrewPayload.NUM_FIRST_CLASS,
        Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
        Aircraft.CrewPayload.NUM_TOURIST_CLASS,
    ]

    design_pax_keys = [
        Aircraft.CrewPayload.Design.NUM_PASSENGERS,
        Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
        Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS,
        Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS,
    ]

    for key in pax_keys:
        if key in aviary_options:
            # mark that the user provided any information on mission passenger count
            pax_provided = True
        else:
            # default all non-provided passenger info to 0
            aviary_options.set_val(key, 0)

    for key in design_pax_keys:
        if key in aviary_options:
            # mark that the user provided any information on design passenger count
            design_pax_provided = True
        else:
            # default all non-provided passenger info to 0
            aviary_options.set_val(key, 0)

    # no passenger info provided
    if not pax_provided and not design_pax_provided:
        if verbosity >= 1:
            UserWarning(
                "No passenger information has been provided for the aircraft, assuming "
                "that this aircraft is not designed to carry passengers."
            )
    # only mission passenger info provided
    if pax_provided and not design_pax_provided:
        if verbosity >= 1:
            UserWarning(
                "Passenger information for the aircraft as designed was not provided. "
                "Assuming that the design passenger count is the same as the passenger "
                "count for the flown mission."
            )
        # set design passengers to mission passenger values
        for i in range(len(pax_keys)):
            mission_val = aviary_options.get_val(pax_keys[i])
            aviary_options.set_val(design_pax_keys[i], mission_val)
    # only design passenger info provided
    if not pax_provided and design_pax_provided:
        if verbosity >= 1:
            UserWarning(
                "Passenger information for the flown mission was not provided. "
                "Assuming that the mission passenger count is the same as the design "
                "passenger count."
            )
        # set mission passengers to design passenger values
        for i in range(len(pax_keys)):
            design_val = aviary_options.get_val(design_pax_keys[i])
            aviary_options.set_val(pax_keys[i], design_val)

    # check that passenger sums match individual seat class values
    design_pax = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
    mission_pax = aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)

    mission_sum = 0
    design_sum = 0
    for i in range(1, len(pax_keys)):
        mission_sum += aviary_options.get_val(pax_keys[i])
        design_sum += aviary_options.get_val(design_pax_keys[i])

    # Resolve conflicts between seat class totals and num_passengers
    # Specific beats general - trust the individual class counts over the total provided, unless
    #    the class counts are all zero, in which case trust the total provided and assume all economy
    # design num_passengers does not match sum of seat classes for design
    if design_pax != design_sum:
        # if sum of seat classes is zero (design pax is not), assume all economy
        if design_sum == 0:
            if verbosity >= 1:
                UserWarning(
                    "Information on seat class distribution for aircraft design was "
                    "not provided - assuming that all passengers are economy class."
                )
            aviary_options.set_val(
                Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS, design_pax
            )
        else:
            if verbosity >= 1:
                UserWarning(
                    "Sum of all passenger classes does not equal total number of "
                    "passengers provided for aircraft design. Overriding "
                    "Aircraft.CrewPayload.Design.NUM_PASSENGERS with the sum of "
                    "passenger classes for design."
                )
            aviary_options.set_val(
                Aircraft.CrewPayload.Design.NUM_PASSENGERS, design_sum
            )

    # mission num_passengers does not match sum of seat classes for mission
    if mission_pax != mission_sum:
        # if sum of seat classes is zero (mission pax is not), assume all economy
        if mission_sum == 0:
            if verbosity >= 1:
                UserWarning(
                    "Information on seat class distribution for current mission was "
                    "not provided - assuming that all passengers are economy class."
                )
            aviary_options.set_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS, mission_pax)
        else:
            if verbosity >= 1:
                UserWarning(
                    "Sum of all passenger classes does not equal total number of "
                    "passengers provided for current mission. Overriding "
                    "Aircraft.CrewPayload.NUM_PASSENGERS with the sum of "
                    "passenger classes for the flown mission."
                )
            aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, mission_sum)

    # Check cases where mission passengers are greater than design passengers
    design_pax = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
    mission_pax = aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
    if mission_pax > design_pax:
        if verbosity >= 1:
            # TODO Should this be an error?
            UserWarning(
                f"The aircraft is designed for {design_pax} passengers but the current "
                f"mission is being flown with {mission_pax} passengers. The mission "
                "will be flown using the mass of these extra passengers and baggage, "
                "but this mission may not be realistic due to lack of room."
            )
    # First Class
    if aviary_options.get_val(
        Aircraft.CrewPayload.Design.NUM_FIRST_CLASS
    ) < aviary_options.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS):
        if verbosity >= 1:
            UserWarning(
                "More first class passengers are flying in this mission than there are "
                "available first class seats on the aircraft. Assuming these passengers "
                "have the same mass as other first class passengers, but are sitting in "
                "different seats."
            )
    # Business Class
    if aviary_options.get_val(
        Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
    ) < aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS):
        if verbosity >= 1:
            UserWarning(
                "More business class passengers are flying in this mission than there are "
                "available business class seats on the aircraft. Assuming these passengers "
                "have the same mass as other business class passengers, but are sitting in "
                "different seats."
            )
    # Economy Class
    if aviary_options.get_val(
        Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
    ) < aviary_options.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS):
        if verbosity >= 1:
            UserWarning(
                "More tourist class passengers are flying in this mission than there are "
                "available tourist class seats on the aircraft. Assuming these passengers "
                "have the same mass as other tourist class passengers, but are sitting in "
                "different seats."
            )

    # Check flight attendants
    if Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS not in aviary_options:
        flight_attendants_count = 0  # assume no passengers

        if 0 < design_pax:
            if design_pax < 51:
                flight_attendants_count = 1

            else:
                flight_attendants_count = design_pax // 40 + 1

        aviary_options.set_val(
            Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, flight_attendants_count)

    if Aircraft.CrewPayload.NUM_GALLEY_CREW not in aviary_options:
        galley_crew_count = 0  # assume no passengers

        if 150 < design_pax:
            galley_crew_count = design_pax // 250 + 1

        aviary_options.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, galley_crew_count)

    if Aircraft.CrewPayload.NUM_FLIGHT_CREW not in aviary_options:
        flight_crew_count = 3

        if design_pax < 151:
            flight_crew_count = 2

        aviary_options.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, flight_crew_count)

    if (Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER not in aviary_options and
            Mission.Design.RANGE in aviary_options):
        design_range = aviary_options.get_val(Mission.Design.RANGE, 'nmi')

        if design_range <= 900.0:
            baggage_mass_per_pax = 35.0
        elif design_range <= 2900.0:
            baggage_mass_per_pax = 40.0
        else:
            baggage_mass_per_pax = 44.0

        aviary_options.set_val(Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER,
                               val=baggage_mass_per_pax, units='lbm')

    return aviary_options


def preprocess_propulsion(aviary_options: AviaryValues, engine_models: list = None):
    '''
    Updates AviaryValues object with values taken from provided EngineModels.

    If no EngineModels are provided, either in engine_models or included in
    aviary_options, an EngineDeck is created using avaliable inputs and options in
    aviary_options.

    Vectorizes variables in aviary_options in the correct order for vehicles with
    heterogeneous engines.

    Performs basic sanity checks on inputs that are universal to all EngineModels.

    !!! WARNING !!!
    Values in aviary_options can be overwritten with corresponding values from
    engine_models!

    Parameters
    ----------
    aviary_options : AviaryValues
        Options to be updated. EngineModels (provided or generated) are added, and
        Aircraft:Engine:* and Aircraft:Nacelle:* variables are vectorized as numpy arrays

    engine_models : <list of EngineModels> (optional)
        EngineModel objects to be added to aviary_options. Replaced existing EngineModels
        in aviary_options
    '''
    # TODO add verbosity check to warnings
    ##############################
    # Vectorize Engine Variables #
    ##############################
    # Only vectorize variables user has defined in some way or engine model has calculated
    # Combine aviary_options and all engine options into single AviaryValues
    # It is assumed that all EngineModels are up-to-date at this point and will NOT
    # be changed later on (otherwise preprocess_propulsion must be run again)
    num_engine_type = len(engine_models)

    complete_options_list = AviaryValues(aviary_options)
    for engine in engine_models:
        complete_options_list.update(engine.options)

    # update_list has keys of all variables that are already defined, and must
    # be vectorized
    update_list = list(get_keys(complete_options_list))

    # Vectorize engine variables. Only update variables in update_list that are relevant
    # to engines (defined by _get_engine_variables())
    for var in _get_engine_variables():
        if var in update_list:
            dtype = _MetaData[var]['types']
            default_value = _MetaData[var]['default_value']
            # type is optionally specified, fall back to type of default value
            if dtype is None:
                if isinstance(default_value, np.ndarray):
                    dtype = default_value.dtype
                elif default_value is None:
                    # With no default value, we cannot determine a dtype.
                    dtype = None
                else:
                    dtype = type(default_value)

            # if dtype has multiple options, use type of default value
            elif isinstance(dtype, (list, tuple)):
                # if default value is a list/tuple, find type inside that
                if isinstance(default_value, (list, tuple)):
                    dtype = type(default_value[0])
                elif default_value is None:
                    # With no default value, we cannot determine a dtype.
                    dtype = None
                else:
                    dtype = type(default_value)

            # if var is supposed to be a unique array per engine model, assemble flat
            # vector manually to avoid ragged arrays (such as for wing engine locations)
            if isinstance(default_value, (list, np.ndarray)):
                vec = np.zeros(0, dtype=dtype)
            elif type(default_value) is tuple:
                vec = ()
            else:
                vec = [default_value] * num_engine_type

            units = _MetaData[var]['units']

            # priority order is (checked per engine):
            # 1. EngineModel.options
            # 2. aviary_options
            # 3. default value from metadata
            for i, engine in enumerate(engine_models):
                # test to see if engine has this variable - if so, use it
                try:
                    # variables in engine models are known to be "safe", will only
                    # contain data for that engine
                    engine_val = engine.get_val(var, units)
                    if isinstance(default_value, (list, np.ndarray)):
                        vec = np.append(vec, engine_val)
                    elif isinstance(default_value, tuple):
                        vec = vec + (engine_val,)
                    else:
                        vec[i] = engine_val
                # if the variable is not in the engine model, pull from aviary options
                except KeyError:
                    # check if variable is defined in aviary options (for this engine's
                    # index) - if so, use it
                    try:
                        aviary_val = aviary_options.get_val(var, units)
                        # if aviary_val is an iterable, just grab val for this engine
                        if isinstance(aviary_val, (list, np.ndarray, tuple)):
                            aviary_val = aviary_val[i]
                        if isinstance(default_value, (list, np.ndarray)):
                            vec = np.append(vec, aviary_val)
                        elif isinstance(default_value, tuple):
                            vec = vec + (aviary_val,)
                        else:
                            vec[i] = aviary_val
                    # if not, use default value from _MetaData
                    except (KeyError, IndexError):
                        if isinstance(default_value, (list, np.ndarray)):
                            vec = np.append(vec, default_value)
                        else:
                            # default value is aleady in array
                            continue
                # TODO update each engine's options with "new" values? Allows each engine
                #      to have a copy of all options/inputs, beyond what it was
                #      originally initialized with

            # update aviary options and outputs with new vectors
            # if data is numerical, store in a numpy array
            # keep tuples as tuples, lists get converted to numpy arrays
            if type(vec[0]) in (int, float, np.int64, np.float64)\
               and type(vec) is not tuple:
                vec = np.array(vec, dtype=dtype)
            aviary_options.set_val(var, vec, units)

    ###################################
    # Input/Option Consistency Checks #
    ###################################
    # Make sure number of engines based on mount location match expected total
    try:
        num_engines_all = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
    except KeyError:
        num_engines_all = np.zeros(num_engine_type).astype(int)
    try:
        num_fuse_engines_all = aviary_options.get_val(
            Aircraft.Engine.NUM_FUSELAGE_ENGINES)
    except KeyError:
        num_fuse_engines_all = np.zeros(num_engine_type).astype(int)
    try:
        num_wing_engines_all = aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)
    except KeyError:
        num_wing_engines_all = np.zeros(num_engine_type).astype(int)

    for i, engine in enumerate(engine_models):
        num_engines = num_engines_all[i]
        num_fuse_engines = num_fuse_engines_all[i]
        num_wing_engines = num_wing_engines_all[i]
        total_engines_calc = num_fuse_engines + num_wing_engines

        # if engine mount type is not specified at all, default to wing
        if total_engines_calc == 0:
            eng_name = engine.name
            num_wing_engines_all[i] = num_engines
            # TODO is a warning overkill here? It can be documented wing mounted engines
            # are assumed default
            warnings.warn(
                f'Mount location for engines of type <{eng_name}> not specified. '
                'Wing-mounted engines are assumed.')

        # If wing mount type are specified but inconsistent, handle it
        elif total_engines_calc > num_engines:
            # more defined engine locations than number of engines - increase num engines
            eng_name = engine.name
            num_engines_all[i] = total_engines_calc
            warnings.warn(
                'Sum of aircraft:engine:num_fueslage_engines and '
                'aircraft:engine:num_wing_engines do not match '
                f'aircraft:engine:num_engines for EngineModel <{eng_name}>. Overwriting '
                'with the sum of wing and fuselage mounted engines.')
        elif total_engines_calc < num_engines:
            # fewer defined locations than num_engines - assume rest are wing mounted
            eng_name = engine.name
            num_wing_engines_all[i] = num_engines - num_fuse_engines
            warnings.warn(
                'Mount location was not defined for all engines of EngineModel '
                f'<{eng_name}> - unspecified engines are assumed wing-mounted.')

    aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, num_engines_all)
    aviary_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, num_wing_engines_all)
    aviary_options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, num_fuse_engines_all)

    if Mission.Summary.FUEL_FLOW_SCALER not in aviary_options:
        aviary_options.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)

    num_engines = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
    total_num_engines = int(sum(num_engines_all))
    total_num_fuse_engines = int(sum(num_fuse_engines_all))
    total_num_wing_engines = int(sum(num_wing_engines_all))

    # compute propulsion-level engine count totals here
    aviary_options.set_val(
        Aircraft.Propulsion.TOTAL_NUM_ENGINES, total_num_engines)
    aviary_options.set_val(
        Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, total_num_fuse_engines)
    aviary_options.set_val(
        Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, total_num_wing_engines)


def _get_engine_variables():
    '''
    Yields all propulsion-related variables in Aircraft that need to be vectorized
    '''
    for item in get_names_from_hierarchy(Aircraft.Engine):
        yield item

    for item in get_names_from_hierarchy(Aircraft.Nacelle):
        yield item
