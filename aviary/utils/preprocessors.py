"""
Preprocessors are utility functions that handle issues with Aviary inputs before model
setup and execution. These tasks include consistency checking between related variables,.

"""

import warnings

import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.named_values import get_keys
from aviary.utils.test_utils.variable_test import get_names_from_hierarchy
from aviary.utils.utils import isiterable
from aviary.variable_info.enums import LegacyCode, ProblemType, Verbosity
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Mission, Settings


# TODO document what kwargs are used, and by which preprocessors in docstring?
# TODO preprocess needed for design range vs phase_info range in sizing missions? (should be the same)
def preprocess_options(aviary_options: AviaryValues, meta_data=_MetaData, verbosity=None, **kwargs):
    """
    Run all preprocessors on provided AviaryValues object.

    Parameters
    ----------
    aviary_options : AviaryValues
        Options to be updated

    meta_data : dict
        Variable metadata being used with this set of aviary_options
    """
    try:
        engine_models = kwargs['engine_models']
    except KeyError:
        engine_models = None

    if verbosity is None:
        if Settings.VERBOSITY in aviary_options:
            verbosity = aviary_options.get_val(Settings.VERBOSITY)
        else:
            verbosity = meta_data[Settings.VERBOSITY]['default_value']
            aviary_options.set_val(Settings.VERBOSITY, verbosity)

    preprocess_crewpayload(aviary_options, meta_data, verbosity)
    preprocess_fuel_capacities(aviary_options, verbosity)

    if engine_models is not None:
        preprocess_propulsion(aviary_options, engine_models, meta_data, verbosity)


def remove_preprocessed_options(aviary_options):
    """
    Remove options whose values will be computed in the preprocessors.

    Parameters
    ----------
    aviary_options : AviaryValues
        Options to be updated
    """
    pre_opt = [
        Aircraft.CrewPayload.NUM_FLIGHT_CREW,
        Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS,
        Aircraft.CrewPayload.NUM_GALLEY_CREW,
        Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER,
    ]

    for option in pre_opt:
        aviary_options.delete(option)


def preprocess_crewpayload(aviary_options: AviaryValues, meta_data=_MetaData, verbosity=None):
    """
    Calculates option values that are derived from other options, and are not direct inputs.
    This function modifies the entries in the supplied collection, and for convenience also
    returns the modified collection.

    Parameters
    ----------
    aviary_options : AviaryValues
        Options to be updated

    meta_data : dict
        Variable metadata being used with this set of aviary_options

    Verbosity, optional
        Sets level of printouts for this function.
    """
    if verbosity is not None:
        # compatibility with being passed int for verbosity
        verbosity = Verbosity(verbosity)
    else:
        verbosity = aviary_options.get_val(Settings.VERBOSITY)

    # Some tests, but not all, do not correctly set default values
    # # so we need to ensure all these values are available.

    for key in (
        Aircraft.CrewPayload.NUM_PASSENGERS,
        Aircraft.CrewPayload.NUM_FIRST_CLASS,
        Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
        Aircraft.CrewPayload.NUM_TOURIST_CLASS,
        Aircraft.CrewPayload.Design.NUM_PASSENGERS,
        Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
        Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS,
        Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS,
    ):
        if key not in aviary_options:
            aviary_options.set_val(key, meta_data[key]['default_value'])

    # Sum passenger Counts for later checks and assignments
    passenger_count = 0
    for key in (
        Aircraft.CrewPayload.NUM_FIRST_CLASS,
        Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
        Aircraft.CrewPayload.NUM_TOURIST_CLASS,
    ):
        passenger_count += aviary_options.get_val(key)
    design_passenger_count = 0
    for key in (
        Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
        Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS,
        Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS,
    ):
        design_passenger_count += aviary_options.get_val(key)

    # Create summary value (num_pax) if it was not assigned by the user
    # or if it was set to it's default value of zero
    if passenger_count != 0 and aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS) == 0:
        aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, passenger_count)
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'User has specified supporting values for NUM_PASSENGERS but has left '
                'NUM_PASSENGERS=0. Replacing NUM_PASSENGERS with passenger_count.'
            )
    if (
        design_passenger_count != 0
        and aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS) == 0
    ):
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, design_passenger_count)
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'User has specified supporting values for Design.NUM_PASSENGERS but has '
                'left Design.NUM_PASSENGERS=0. Replacing Design.NUM_PASSENGERS with '
                'design_passenger_count.'
            )

    num_pax = aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
    design_num_pax = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    # Check summary data against individual data if individual data was entered
    if passenger_count != 0 and num_pax != passenger_count:
        if verbosity > verbosity.BRIEF:
            warnings.warn(
                'Total passenger count ('
                f'{aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)}) does not '
                'equal the sum of first class + business class + tourist class passengers '
                f'(total of {passenger_count}). Setting total number of passengers to '
                f'{passenger_count}.'
            )
        aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, passenger_count)
    if design_passenger_count != 0 and design_num_pax != design_passenger_count:
        if verbosity > verbosity.BRIEF:
            warnings.warn(
                'Design total passenger count ('
                f'{aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)}) '
                'does not equal the sum of design first class + business class + tourist '
                f'class passengers (total of {design_passenger_count}). Setting total number of '
                f'design passengers to {design_passenger_count}.'
            )
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, design_passenger_count)

    # TODO these don't have to be errors, we can recover in some cases, for example
    # defaulting to all economy class if passenger seat info is not provided. See the
    # engine count checks for an example of this.
    # Fail if incorrect data sets were provided:
    # have you give us enough info to determine where people were sitting vs. designed seats
    if num_pax != 0 and design_passenger_count != 0 and passenger_count == 0:
        raise UserWarning(
            'The user has specified CrewPayload.NUM_PASSENGERS, and how many of what '
            'types of seats are on the aircraft. However, the user has not specified '
            'where those passengers are sitting. User must specify '
            'CrewPayload.FIRST_CLASS, CrewPayload.NUM_BUSINESS_CLASS, NUM_TOURIST_CLASS '
            'in aviary_values.'
        )
        # where are the people sitting? is first class full? We know how many seats are in each class.
    if design_num_pax != 0 and passenger_count != 0 and design_passenger_count == 0:
        raise UserWarning(
            'The user has specified Design.NUM_PASSENGERS, and has specified how many '
            'people are sitting in each class of seats. However, the user has not '
            'specified how many seats of each class exist in the aircraft. User must '
            'specify Design.FIRST_CLASS, Design.NUM_BUSINESS_CLASS, '
            'Design.NUM_TOURIST_CLASS in aviary_values.'
        )
        # we don't know which classes this aircraft has been design for. How many 1st class seats are there?

    # Copy data over if only one set of data exists
    # User has given detailed values for as-flown and NO design values at all
    if passenger_count != 0 and design_num_pax == 0 and design_passenger_count == 0:
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'User has not input design passengers data. Assuming design is equal to '
                'as-flown passenger data.'
            )
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, passenger_count)
        aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
            aviary_options.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS),
        )
        aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS,
            aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS),
        )
        aviary_options.set_val(
            Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS,
            aviary_options.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS),
        )
    # user has not supplied detailed information on design but has supplied summary information on passengers
    elif num_pax != 0 and design_num_pax == 0:
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'User has specified as-flown NUM_PASSENGERS but not how many passengers '
                'the aircraft was designed for in Design.NUM_PASSENGERS. Assuming they '
                'are equal.'
            )
        aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, num_pax)
    elif design_passenger_count != 0 and num_pax == 0 and passenger_count == 0:
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'User has specified Design.NUM_* passenger values but CrewPyaload.NUM_* '
                'category has been left blank or set to zero. Assuming they are equal '
                'to maintain backwards compatibility with converted GASP and FLOPS. '
                'If you intended to have no passengers on this flight, set '
                'Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS to zero in aviary_values.'
            )
        aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, design_passenger_count)
        aviary_options.set_val(
            Aircraft.CrewPayload.NUM_FIRST_CLASS,
            aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS),
        )
        aviary_options.set_val(
            Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
            aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS),
        )
        aviary_options.set_val(
            Aircraft.CrewPayload.NUM_TOURIST_CLASS,
            aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS),
        )
    # user has not supplied detailed information on design but has supplied summary information on passengers
    elif design_num_pax != 0 and num_pax == 0:
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'User has specified Design.NUM_PASSENGERS but '
                'CrewPayload.NUM_PASSENGERS has been left blank or set to zero. '
                'Assuming they are equal to maintain backwards compatibility with '
                'converted GASP and FLOPS files. If you intended to have no passengers '
                'on this flight, set Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS to zero in '
                'aviary_values'
            )
        aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, design_num_pax)

    # Perform checks on the final data tables to ensure Design is always larger then As-Flown
    if aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS) < aviary_options.get_val(
        Aircraft.CrewPayload.NUM_FIRST_CLASS
    ):
        raise UserWarning(
            'NUM_FIRST_CLASS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS)}) is larger '
            'than the number of seats set by Design.NUM_FIRST_CLASS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)})'
        )
    if aviary_options.get_val(
        Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
    ) < aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS):
        raise UserWarning(
            'NUM_BUSINESS_CLASS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS)}) is '
            'larger than the number of seats set by Design.NUM_BUSINESS_CLASS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS)})'
        )
    if aviary_options.get_val(
        Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
    ) < aviary_options.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS):
        raise UserWarning(
            'NUM_TOURIST_CLASS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.NUM_TOURIST_CLASS)}) is '
            'larger than the number of seats set by Design.NUM_TOURIST_CLASS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS)})'
        )
    if aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS) < aviary_options.get_val(
        Aircraft.CrewPayload.NUM_PASSENGERS
    ):
        raise UserWarning(
            'NUM_PASSENGERS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)}) is larger '
            'than the number of seats set by Design.NUM_PASSENGERS ('
            f'{aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)})'
        )

    # Check and process cargo variables - confirm mass method
    if Settings.MASS_METHOD in aviary_options:
        mass_method = aviary_options.get_val(Settings.MASS_METHOD)
    else:
        raise UserWarning('MASS_METHOD not specified. Cannot preprocess cargo inputs.')

    # Process GASP based cargo variables
    if mass_method == LegacyCode.GASP:
        try:
            cargo = aviary_options.get_val(Aircraft.CrewPayload.CARGO_MASS, 'lbm')
        except KeyError:
            cargo = None
        try:
            max_cargo = aviary_options.get_val(Aircraft.CrewPayload.Design.MAX_CARGO_MASS, 'lbm')
        except KeyError:
            max_cargo = None
        try:
            des_cargo = aviary_options.get_val(Aircraft.CrewPayload.Design.CARGO_MASS, 'lbm')
        except KeyError:
            des_cargo = None

        if Settings.PROBLEM_TYPE in aviary_options:
            problem_type = aviary_options.get_val(Settings.PROBLEM_TYPE)
        else:
            problem_type = ProblemType.SIZING

        if cargo is not None:
            if max_cargo is not None:
                if des_cargo is not None:
                    if problem_type == ProblemType.SIZING and cargo != des_cargo:
                        # user has set all three check if self consistent
                        cargo = des_cargo
                        if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                            warnings.warn(
                                f'Aircraft.CrewPayload.CARGO_MASS ({cargo}) does not equal '
                                f'Aircraft.CrewPayload.Design.CARGO_MASS ({des_cargo}) for SIZING '
                                'mission. Setting as-flown CARGO_MASS equal to Design.CARGO_MASS '
                                f'({des_cargo})'
                            )
                else:
                    # user has set cargo & max: assume des = max
                    des_cargo = max_cargo
                    if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                        warnings.warn(
                            'Aircraft.CrewPayload.Design.CARGO_MASS missing, assume '
                            f'Design.CARGO_MASS = Design.MAX_CARGO_MASS ({max_cargo})'
                        )
            elif des_cargo is not None:
                # user has set cargo & des: assume max = des
                max_cargo = des_cargo
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                    warnings.warn(
                        'Aircraft.CrewPayload.Design.MAX_CARGO_MASS is missing, assuming '
                        f'Design.MAX_CARGO_MASS equals Design.CARGO_MASS ({des_cargo})'
                    )
            else:
                # user has set cargo only: assume design and max cargo is equal to flown cargo
                des_cargo = max_cargo = cargo

        elif max_cargo is not None:
            if des_cargo is not None:
                # user has set max & des: assume flown = 0
                cargo = 0
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG:
                    warnings.warn(
                        'Aircraft.CrewPayload.CARGO_MASS is missing, assuming CARGO_MASS = 0'
                    )
            else:
                # user has set max only: assume flown = des = 0
                cargo = des_cargo = 0
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG:
                    warnings.warn(
                        'Aircraft.CrewPayload.CARGO_MASS and Aircraft.CrewPayload.Design.CARGO_MASS '
                        'missing, assume CARGO_MASS and Design.CARGO_MASS = 0. No Cargo is flown on '
                        'mission.'
                    )

        elif des_cargo is not None:
            # user has only input des: assume max = des and flown = 0
            max_cargo = des_cargo
            cargo = 0
            if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG:
                warnings.warn(
                    'Aircraft.CrewPayload.CARGO_MASS and Aircraft.CrewPayload.Design.MAX_CARGO_MASS '
                    'missing, assume CARGO_MASS = 0 and Design.MAX_CARGO_MASS = Design.CARGO_MASS '
                    f'({des_cargo}).'
                )

        else:
            # user has input no cargo information
            cargo = max_cargo = des_cargo = 0
            if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG:
                warnings.warn(
                    'No CARGO variables detected, assume CARGO_MASS, Design.MAX_CARGO_MASS, and '
                    'Design.CARGO_MASS equal to 0.'
                )

        if cargo > max_cargo or des_cargo > max_cargo:
            raise UserWarning(
                f'Aircraft.CrewPayload.CARGO_MASS ({cargo}) and/or '
                f'Aircraft.CrewPayload.Design.CARGO_MASS ({des_cargo}) is greater than'
                f'Aircraft.CrewPayload.Design.MAX_CARGO_MASS ({max_cargo})'
            )

        # calculate passenger mass with bags based on user inputs.
        try:
            pax_mass_with_bag = aviary_options.get_val(
                Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, 'lbm'
            )
        except KeyError:
            pax_mass = aviary_options.get_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 'lbm')
            bag_mass = aviary_options.get_val(
                Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 'lbm'
            )
            pax_mass_with_bag = pax_mass + bag_mass
            aviary_options.set_val(
                Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, pax_mass_with_bag, 'lbm'
            )

        # calculate and check total payload
        # NOTE this is only used for error messaging the calculations for analysis are subsystems/mass/gasp_based
        design_passenger_payload_mass = design_num_pax * pax_mass_with_bag
        max_payload = design_passenger_payload_mass + max_cargo
        num_pax = aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
        as_flown_passenger_payload_mass = num_pax * pax_mass_with_bag
        as_flown_payload = as_flown_passenger_payload_mass + cargo
        if as_flown_payload > max_payload and verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
            warnings.warn(
                f'As-flown payload ({as_flown_payload}) is greater than maximum payload '
                f'({max_payload}). The aircraft will be undersized for this payload!'
            )

        # set assumed cargo mass variables:
        aviary_options.set_val(Aircraft.CrewPayload.CARGO_MASS, cargo, 'lbm')
        aviary_options.set_val(Aircraft.CrewPayload.Design.MAX_CARGO_MASS, max_cargo, 'lbm')
        aviary_options.set_val(Aircraft.CrewPayload.Design.CARGO_MASS, des_cargo, 'lbm')

    if Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS not in aviary_options:
        flight_attendants_count = 0  # assume no passengers

        if 0 < passenger_count:
            if passenger_count < 51:
                flight_attendants_count = 1

            else:
                flight_attendants_count = passenger_count // 40 + 1

        aviary_options.set_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, flight_attendants_count)

    if Aircraft.CrewPayload.NUM_GALLEY_CREW not in aviary_options:
        galley_crew_count = 0  # assume no passengers

        if 150 < passenger_count:
            galley_crew_count = passenger_count // 250 + 1

        aviary_options.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, galley_crew_count)

    if Aircraft.CrewPayload.NUM_FLIGHT_CREW not in aviary_options:
        flight_crew_count = 3

        if passenger_count < 151:
            flight_crew_count = 2

        aviary_options.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, flight_crew_count)

    if (
        Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER not in aviary_options
        and Mission.Design.RANGE in aviary_options
    ):
        design_range = aviary_options.get_val(Mission.Design.RANGE, 'nmi')

        if design_range <= 900.0:
            baggage_mass_per_pax = 35.0
        elif design_range <= 2900.0:
            baggage_mass_per_pax = 40.0
        else:
            baggage_mass_per_pax = 44.0

        aviary_options.set_val(
            Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER,
            val=baggage_mass_per_pax,
            units='lbm',
        )

    return aviary_options


def preprocess_fuel_capacities(aviary_options: AviaryValues, verbosity=None):
    """
    Preprocesses the AviaryValues object to ensure the user has provided a consistent set of fuel capacity overrides.

    Parameters
    ----------
    aviary_options : AviaryValues
        Options to be updated

    """
    if verbosity is not None:
        # compatibility with being passed int for verbosity
        verbosity = Verbosity(verbosity)
    else:
        verbosity = aviary_options.get_val(Settings.VERBOSITY)

    if Settings.MASS_METHOD in aviary_options:
        mass_method = aviary_options.get_val(Settings.MASS_METHOD)
    else:
        raise UserWarning('MASS_METHOD not specified. Cannot preprocess fuel capacity inputs.')

    if mass_method == LegacyCode.FLOPS:
        # find which fuel capacity variables the user has set:
        if Aircraft.Fuel.TOTAL_CAPACITY not in aviary_options:
            # Aviary will need to calculate the total capacity and can only do so if we assume any missing subsystem capacities are zero
            if Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY not in aviary_options:
                aviary_options.set_val(Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0, 'lbm')
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(f'Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY is missing assume = 0')

            if Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY not in aviary_options:
                aviary_options.set_val(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0, 'lbm')
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(f'Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY is missing assume = 0')
        else:
            total_capacity = aviary_options.get_val(Aircraft.Fuel.TOTAL_CAPACITY, 'lbm')
            try:
                wing_capacity = aviary_options.get_val(Aircraft.Fuel.WING_FUEL_CAPACITY, 'lbm')
            except KeyError:
                wing_capacity = None
            try:
                fuselage_capacity = aviary_options.get_val(
                    Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 'lbm'
                )
            except KeyError:
                fuselage_capacity = None
            try:
                auxiliary_capacity = aviary_options.get_val(
                    Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 'lbm'
                )
            except KeyError:
                auxiliary_capacity = None

            capacity_count = sum(
                1
                for capacity in [wing_capacity, fuselage_capacity, auxiliary_capacity]
                if capacity is not None
            )
            capacity_check = sum(
                capacity
                for capacity in [wing_capacity, fuselage_capacity, auxiliary_capacity]
                if capacity is not None
            )

            # check if the user inputs are self consistent (as far as possible at this stage!) Aviary can still calculate -ve outputs at runtime.
            if capacity_count == 3 and capacity_check != total_capacity:
                raise UserWarning(
                    f'Aircraft.Fuel.TOTAL_CAPACITY ({total_capacity}) != Aircraft.Fuel.WING_FUEL_CAPACITY ({wing_capacity})'
                    f'+ Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY ({fuselage_capacity}) + Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY ({auxiliary_capacity})'
                    f' = {capacity_check}'
                )
            elif capacity_count < 3 and capacity_check > total_capacity:
                raise UserWarning(
                    f'Aircraft.Fuel.TOTAL_CAPACITY ({total_capacity}) < Aircraft.Fuel.WING_FUEL_CAPACITY ({wing_capacity})'
                    f' + Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY ({fuselage_capacity}) + Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY ({auxiliary_capacity})'
                    f' = {capacity_check}'
                )

    return aviary_options


def preprocess_propulsion(
    aviary_options: AviaryValues,
    engine_models: list = None,
    meta_data=_MetaData,
    verbosity=None,
):
    """
    Updates AviaryValues object with values taken from provided EngineModels.

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
    """
    if verbosity is not None:
        # compatibility with being passed int for verbosity
        verbosity = Verbosity(verbosity)
    else:
        if verbosity in aviary_options:
            verbosity = aviary_options.get_val(Settings.VERBOSITY)
        else:
            verbosity = Verbosity.BRIEF

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
            dtype = meta_data[var]['types']
            default_value = meta_data[var]['default_value']
            multivalue = meta_data[var]['multivalue']
            units = meta_data[var]['units']

            # If dtype has multiple options, prefer type of default value
            # Otherwise, use the first type in the tuple
            if isinstance(dtype, tuple):
                if default_value is not None:
                    dtype = type(default_value)
                else:
                    dtype = dtype[0]

            if isiterable(meta_data[var]['types']):
                typeset = meta_data[var]['types']
            else:
                typeset = (meta_data[var]['types'],)

            # Variables are multidimensional if their base types have iterables, and are
            # flagged as `multivalue`
            multidimensional = set(typeset) & set((list, tuple, np.ndarray)) and multivalue

            # vec is where the vectorized engine data is stored - always a list right
            # now, converted to other types like np array later
            vec = []

            # Vectorize variable "var" from available sources #

            # If var is supposed to be a unique array per engine model, assemble flat
            # vector manually to avoid ragged arrays (such as for wing engine locations)

            # Priority order is (checked per engine):
            # 1. EngineModel.options
            # 2. aviary_options
            # 3. default value from metadata
            for i, engine in enumerate(engine_models):
                # test to see if engine has this variable - if so, use it
                try:
                    # variables in engine models are trusted to be "safe", and only
                    # contain data for that engine
                    engine_val = engine.get_val(var, units)
                # if the variable is not in the engine model, pull from aviary options
                except KeyError:
                    # check if variable is defined in aviary options (for this engine's
                    # index) - if so, use it
                    try:
                        aviary_val = aviary_options.get_val(var, units)
                    # if the variable is not in aviary_options, use default from metadata
                    except (KeyError, IndexError):
                        vec.append(default_value)
                    else:
                        # save value from aviary_options
                        if isiterable(aviary_val):
                            if multidimensional:
                                vec.extend(aviary_val)
                            else:
                                # if aviary_val is an iterable, just grab val for this engine
                                vec.append(aviary_val[i])
                        else:
                            vec.append(aviary_val)
                else:
                    # save value from EngineModel
                    if isiterable(engine_val) and multidimensional:
                        vec.extend(engine_val)
                    else:
                        vec.append(engine_val)
                # TODO update each engine's options with "new" values? Allows each engine
                #      to have a copy of all options/inputs, beyond what it was
                #      originally initialized with

            # Update aviary options with new vectors
            # If data is numerical, store in a numpy array, else use a list
            # Some machines default to specific-bit np array types, so we have to
            # check for those too
            if type(vec[0]) in (int, float, np.int32, np.int64, np.float32, np.float64):
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
        num_fuse_engines_all = aviary_options.get_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES)
    except KeyError:
        num_fuse_engines_all = np.zeros(num_engine_type).astype(int)
    try:
        num_wing_engines_all = aviary_options.get_val(Aircraft.Engine.NUM_WING_ENGINES)
    except KeyError:
        num_wing_engines_all = np.zeros(num_engine_type).astype(int)

    for i, engine in enumerate(engine_models):
        eng_name = engine.name
        num_engines = num_engines_all[i]
        num_fuse_engines = num_fuse_engines_all[i]
        num_wing_engines = num_wing_engines_all[i]
        total_engines_calc = num_fuse_engines + num_wing_engines

        # If engine mount type is not specified at all, default to wing (unless there is
        # only one engine, in which case default to fuselage)
        if total_engines_calc == 0:
            if num_engines > 1:
                num_wing_engines_all[i] = num_engines
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                    warnings.warn(
                        f'Mount location for engines of type <{eng_name}> not '
                        'specified. Wing-mounted engines are assumed.'
                    )
            else:
                num_fuse_engines_all[i] = num_engines
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                    warnings.warn(
                        f'Mount location for single engine of type <{eng_name}> not '
                        'specified. Assuming it is fuselage-mounted.'
                    )

        # If wing mount type are specified but inconsistent, handle it
        elif total_engines_calc > num_engines:
            # more defined engine locations than number of engines - increase num engines
            num_engines_all[i] = total_engines_calc
            if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                warnings.warn(
                    'Sum of user-specified fuselage-mounted and wing-mounted engines do '
                    f'not match total number of engines for EngineModel <{eng_name}>. '
                    'Overwriting total number of engines with the sum of wing and '
                    'fuselage mounted engines.'
                )
        elif total_engines_calc < num_engines:
            # fewer defined locations than num_engines - assume rest are wing mounted
            # (unless there is just one prospective wing engine, then fuselage mount it)
            num_unspecified_engines = num_engines - num_fuse_engines - num_wing_engines
            if num_unspecified_engines > 1:
                num_wing_engines_all[i] = num_wing_engines + num_unspecified_engines
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                    warnings.warn(
                        'Mount location was not defined for all engines of EngineModel '
                        f'<{eng_name}> - unspecified engines are assumed wing-mounted.'
                    )
            elif num_unspecified_engines == 1 and num_wing_engines != 0:
                num_fuse_engines_all[i] = num_fuse_engines + num_unspecified_engines
                if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                    warnings.warn(
                        'Mount location was not defined for all engine of EngineModel '
                        f'<{eng_name}> - unspecified engine is assumed fuselage-mounted.'
                    )

        if num_wing_engines % 2 == 1:
            if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
                warnings.warn(
                    'Odd number of wing engines are specified for EngineModel '
                    f'<{eng_name}> - this may cause issues with some mass and geometry '
                    'components that assume symmetric wing engine distribution.'
                )

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
    aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, total_num_engines)
    aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES, total_num_fuse_engines)
    aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, total_num_wing_engines)


def _get_engine_variables():
    """Yields all propulsion-related variables in Aircraft that need to be vectorized."""
    for item in get_names_from_hierarchy(Aircraft.Engine):
        yield item

    for item in get_names_from_hierarchy(Aircraft.Nacelle):
        yield item
