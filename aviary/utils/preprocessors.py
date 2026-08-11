"""
Preprocessors are utility functions that handle issues with Aviary inputs before model
setup and execution. These tasks include consistency checking between related variables.

A general rule of thumb for verbosity is VERBOSE or higher is preferred if no user-set values are
being replaced. BRIEF is the default, so it should only be used when it is actually important the
user is notified at run-time that something happened.
"""

import warnings

import numpy as np

from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import get_names_from_hierarchy
from aviary.utils.utils import isiterable
from aviary.variable_info.enums import AircraftTypes, LegacyCode, ProblemType, Verbosity
from aviary.variable_info.variable_meta_data import CoreMetaData
from aviary.variable_info.variables import Aircraft, Mission, Settings


# TODO document what kwargs are used, and by which preprocessors in docstring?
# TODO preprocess needed for design range vs phase_info range in sizing missions? (should be the same)
def preprocess_options(
    aviary_options: AviaryValues, meta_data=CoreMetaData, verbosity=None, **kwargs
):
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

    # TODO move these into their own subfunction (preprocess options is a collector for preprocessor
    #      functions only, and should not change options on its own)

    # TODO the zero value behavior should probably be covered in fortran_to_aviary, and the preprocessor sets up the default value if it is not provided in the input file at all
    if Aircraft.Wing.ASPECT_RATIO_REFERENCE in aviary_options:
        arref = aviary_options.get_val(Aircraft.Wing.ASPECT_RATIO_REFERENCE)
        if (np.isscalar(arref) and arref == 0) or (not np.isscalar(arref) and arref[0] == 0):
            if Aircraft.Wing.ASPECT_RATIO in aviary_options:
                ar = aviary_options.get_val(Aircraft.Wing.ASPECT_RATIO)
                aviary_options.set_val(Aircraft.Wing.ASPECT_RATIO_REFERENCE, ar, 'unitless')
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(
                        'Setting Aircraft.Wing.ASPECT_RATIO_REFERENCE to the same value as '
                        f'Aircraft.Wing.ASPECT_RATIO ({ar}).'
                    )

    # TODO also move zero value behavior to fortran_to_aviary and rewrite this case to cover when variable is not provided
    if Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE in aviary_options:
        tcref = aviary_options.get_val(Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE)
        if (np.isscalar(tcref) and tcref == 0) or (not np.isscalar(tcref) and tcref[0] == 0):
            if Aircraft.Wing.THICKNESS_TO_CHORD in aviary_options:
                tc = aviary_options.get_val(Aircraft.Wing.THICKNESS_TO_CHORD)
                aviary_options.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE, tc, 'unitless')
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(
                        'Setting Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE to the same value as '
                        f'Aircraft.Wing.THICKNESS_TO_CHORD ({tc}).'
                    )

    if Aircraft.Design.PERCENT_EXCRESCENCE_DRAG not in aviary_options:
        # In FLOPS, excrescence drag percentage is not able to be set via the input file
        # Therefore, it appears to have been hardcoded into the method
        # Here we set the default value to that fixed value
        if (
            Settings.AERODYNAMICS_METHOD in aviary_options
            and aviary_options.get_val(Settings.AERODYNAMICS_METHOD) is LegacyCode.FLOPS
        ):
            aviary_options.set_val(Aircraft.Design.PERCENT_EXCRESCENCE_DRAG, 0.06)

    if Settings.MASS_METHOD in aviary_options:
        mass_method = aviary_options.get_val(Settings.MASS_METHOD)
    else:
        raise UserWarning('MASS_METHOD not specified. Cannot preprocess inputs.')

    if Aircraft.Design.TYPE in aviary_options:
        design_type = aviary_options.get_val(Aircraft.Design.TYPE)
    else:
        design_type = AircraftTypes.TRANSPORT
        if verbosity > Verbosity.BRIEF:
            warnings.warn('Setting Aircraft.Design.TYPE = AircraftTypes.TRANSPORT.')

    if Aircraft.Fuselage.SIMPLE_LAYOUT not in aviary_options:
        if mass_method == LegacyCode.FLOPS:
            aviary_options.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, True, 'unitless')
            simple_layout = True
        elif mass_method == LegacyCode.GASP:
            if design_type == AircraftTypes.BLENDED_WING_BODY:
                simple_layout = False
            else:
                simple_layout = True
            aviary_options.set_val(Aircraft.Fuselage.SIMPLE_LAYOUT, simple_layout, 'unitless')
    else:
        simple_layout = aviary_options.get_val(Aircraft.Fuselage.SIMPLE_LAYOUT)

    if simple_layout == False:
        # Check seat widths are set
        if mass_method == LegacyCode.FLOPS:
            if design_type == AircraftTypes.TRANSPORT:
                if Aircraft.Fuselage.SEAT_WIDTH_ECONOMY not in aviary_options:
                    aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.0, 'inch')
                    if verbosity >= Verbosity.BRIEF:
                        warnings.warn(
                            'Aircraft.Fuselage.SEAT_WIDTH_ECONOMY not set, '
                            'assuming default 20.0 inches.'
                        )
                if Aircraft.Fuselage.SEAT_WIDTH_BUSINESS not in aviary_options:
                    aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH_BUSINESS, 22.0, 'inch')
                    if verbosity >= Verbosity.BRIEF:
                        warnings.warn(
                            'Aircraft.Fuselage.SEAT_WIDTH_BUSINESS not set, '
                            'assuming default 22.0 inches.'
                        )
                if Aircraft.Fuselage.SEAT_WIDTH_FIRST not in aviary_options:
                    aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH_FIRST, 25.0, 'inch')
                    if verbosity >= Verbosity.BRIEF:
                        warnings.warn(
                            'Aircraft.Fuselage.SEAT_WIDTH_FIRST not set, '
                            'assuming default 25.0 inches.'
                        )
        elif mass_method == LegacyCode.GASP:
            if design_type == AircraftTypes.BLENDED_WING_BODY:
                if Aircraft.Fuselage.SEAT_WIDTH_ECONOMY not in aviary_options:
                    aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH_ECONOMY, 20.0, 'inch')
                    if verbosity >= Verbosity.BRIEF:
                        warnings.warn(
                            'Aircraft.Fuselage.SEAT_WIDTH_ECONOMY not set, '
                            'assuming default 20.0 inches.'
                        )
                if Aircraft.Fuselage.SEAT_WIDTH_BUSINESS not in aviary_options:
                    aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH_BUSINESS, 22.0, 'inch')
                    if verbosity >= Verbosity.BRIEF:
                        warnings.warn(
                            'Aircraft.Fuselage.SEAT_WIDTH_BUSINESS not set, '
                            'assuming default 22.0 inches.'
                        )
                if Aircraft.Fuselage.SEAT_WIDTH_FIRST not in aviary_options:
                    aviary_options.set_val(Aircraft.Fuselage.SEAT_WIDTH_FIRST, 28.0, 'inch')
                    if verbosity >= Verbosity.BRIEF:
                        warnings.warn(
                            'Aircraft.Fuselage.SEAT_WIDTH_FIRST not set, '
                            'assuming default 28.0 inches.'
                        )

    # preprocess atmosphere / GRAV??
    # Set the gravity model based on the atmosphere model to enable calculation of weight from mass
    from aviary.subsystems.atmosphere.utils.get_atmosphere_data import get_atmosphere_data

    if Settings.ATMOSPHERE_MODEL not in aviary_options:
        aviary_options.set_val(
            Settings.ATMOSPHERE_MODEL, meta_data[Settings.ATMOSPHERE_MODEL]['default_value']
        )
    _, _, _, planet_gravity, sea_level_density = get_atmosphere_data(
        aviary_options.get_val(Settings.ATMOSPHERE_MODEL)
    )

    if Mission.GRAVITY not in aviary_options:  # Check to see if the user has set a gravity profile.
        # No gravity profile is set, set it now.
        aviary_options.set_val(
            Mission.GRAVITY, val=planet_gravity[0], units=planet_gravity[1]
        )  # contains both value and units as a tuple.

    if Mission.SEA_LEVEL_DENSITY not in aviary_options:
        aviary_options.set_val(
            Mission.SEA_LEVEL_DENSITY, val=sea_level_density[0], units=sea_level_density[1]
        )


def preprocess_crewpayload(aviary_options: AviaryValues, meta_data=CoreMetaData, verbosity=None):
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
    # Check and process cargo variables - confirm mass method
    if Settings.MASS_METHOD in aviary_options:
        mass_method = aviary_options.get_val(Settings.MASS_METHOD)
    else:
        raise UserWarning('MASS_METHOD not specified. Cannot preprocess cargo inputs.')

    if verbosity is not None:
        # compatibility with being passed int for verbosity
        verbosity = Verbosity(verbosity)
    else:
        verbosity = aviary_options.get_val(Settings.VERBOSITY)
    pax_provided = False
    pax_sum_provided = False
    design_pax_provided = False
    design_pax_sum_provided = False

    if mass_method == LegacyCode.FLOPS:
        pax_keys = [
            Aircraft.CrewPayload.NUM_PASSENGERS,
            Aircraft.CrewPayload.NUM_FIRST_CLASS,
            Aircraft.CrewPayload.NUM_BUSINESS_CLASS,
            Aircraft.CrewPayload.NUM_ECONOMY_CLASS,
        ]

        design_pax_keys = [
            Aircraft.CrewPayload.Design.NUM_PASSENGERS,
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS,
            Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS,
            Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS,
        ]
    elif mass_method is LegacyCode.GASP:
        pax_keys = [Aircraft.CrewPayload.NUM_PASSENGERS]

        design_pax_keys = [Aircraft.CrewPayload.Design.NUM_PASSENGERS]

    for key in pax_keys:
        if key in aviary_options:
            # mark that the user provided any information on mission passenger count
            pax_provided = True
            if key == Aircraft.CrewPayload.NUM_PASSENGERS:
                pax_sum_provided = True
        else:
            # default all non-provided passenger info to 0
            aviary_options.set_val(key, 0)

    for key in design_pax_keys:
        if key in aviary_options:
            # mark that the user provided any information on design passenger count
            design_pax_provided = True
            if key == Aircraft.CrewPayload.NUM_PASSENGERS:
                design_pax_sum_provided = True
        else:
            # default all non-provided passenger info to 0
            aviary_options.set_val(key, 0)

    # no passenger info provided
    if not pax_provided and not design_pax_provided:
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'No passenger information has been provided for the aircraft, assuming that this '
                'aircraft is not designed to carry passengers.'
            )
    # only mission passenger info provided
    if pax_provided and not design_pax_provided:
        if verbosity >= Verbosity.BRIEF:
            warnings.warn(
                'Passenger information for the aircraft as designed was not provided. Assuming that '
                'the design passenger count is the same as the passenger count for the flown mission.'
            )
        # set design passengers to mission passenger values
        for i in range(len(pax_keys)):
            mission_val = aviary_options.get_val(pax_keys[i])
            aviary_options.set_val(design_pax_keys[i], mission_val)

    # only design passenger info provided
    if not pax_provided and design_pax_provided:
        # Assuming mission uses design pax is less serious than assuming design uses mission pax
        # Only warning for VERBOSE or higher
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                'Passenger information for the flown mission was not provided. Assuming that the '
                'mission passenger count is the same as the design passenger count.'
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
    if design_pax != design_sum:
        # if sum of seat classes is zero (design pax is not), assume all economy
        if design_sum == 0:
            if verbosity >= Verbosity.VERBOSE:
                warnings.warn(
                    'Information on seat class distribution for aircraft design was not provided - '
                    f'assuming all {design_pax} passengers are economy class.'
                )
            aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS, design_pax)
        else:
            if design_pax_sum_provided and verbosity >= Verbosity.BRIEF:
                warnings.warn(
                    f'Sum of all passenger classes ({design_sum}) does not equal total number of '
                    f'passengers provided for aircraft design {design_pax}. Overriding '
                    'Aircraft.CrewPayload.Design.NUM_PASSENGERS with the sum of '
                    f'passenger classes for design ({design_sum}).'
                )
            aviary_options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, design_sum)

    # mission num_passengers does not match sum of seat classes for mission
    if mission_pax != mission_sum:
        # if sum of seat classes is zero (mission pax is not), assume all economy
        if mission_sum == 0:
            if verbosity >= Verbosity.VERBOSE:
                warnings.warn(
                    'Information on seat class distribution for current mission was not provided - '
                    f'assuming that all {mission_pax} passengers are economy class.'
                )
            aviary_options.set_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS, mission_pax)
        else:
            if pax_sum_provided and verbosity >= Verbosity.BRIEF:
                warnings.warn(
                    f'Sum of all passenger classes ({mission_sum}) does not equal total number of '
                    f'passengers provided for current mission ({mission_pax}). Setting '
                    'Aircraft.CrewPayload.NUM_PASSENGERS with the sum of passenger classes for flown '
                    f'mission ({mission_sum}).'
                )
            aviary_options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, mission_sum)

    # Check cases where mission passengers are greater than design passengers
    design_pax = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS)
    mission_pax = aviary_options.get_val(Aircraft.CrewPayload.NUM_PASSENGERS)
    if mission_pax > design_pax:
        if verbosity >= Verbosity.VERBOSE:
            warnings.warn(
                f'The aircraft is designed for {design_pax} passengers but the current mission is '
                f'being flown with {mission_pax} passengers. The mission will be flown using the '
                'mass of these extra passengers and baggage, and it is assumed there is still room '
                'on the aircraft.'
            )

    if mass_method == LegacyCode.FLOPS:
        # First Class
        des_num_first = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        num_first = aviary_options.get_val(Aircraft.CrewPayload.NUM_FIRST_CLASS)
        if num_first > des_num_first:
            if verbosity >= Verbosity.VERBOSE:
                warnings.warn(
                    f'More first class passengers ({num_first}) are flying in this mission than '
                    f'available first class seats ({des_num_first}). Assuming these passengers have '
                    'the same mass as other first class passengers, but are sitting in different seats.'
                )
        # Business Class
        des_num_business = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS)
        num_business = aviary_options.get_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS)
        if num_business > des_num_business:
            if verbosity >= Verbosity.VERBOSE:
                warnings.warn(
                    f'More business class passengers ({num_business}) are flying in this mission '
                    f'than available business class seats ({des_num_business}). Assuming these '
                    'passengers have the same mass as other business class passengers, but are '
                    'sitting in different seats.'
                )
        # Economy Class
        des_num_econ = aviary_options.get_val(Aircraft.CrewPayload.Design.NUM_ECONOMY_CLASS)
        num_econ = aviary_options.get_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS)
        if num_econ > des_num_econ:
            if verbosity >= Verbosity.VERBOSE:
                warnings.warn(
                    f'More economy class passengers ({num_econ}) are flying in this mission than '
                    f'available economy class seats ({des_num_econ}). Assuming these passengers '
                    'have the same mass as other economy class passengers, but are sitting in '
                    'different seats.'
                )

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

        # TODO GASP cargo preprocessing does not fully account for *which* mission type is being
        #      flown (make sure that as-flown cargo matches design cargo for sizing missions)
        if cargo is not None:
            if max_cargo is not None:
                if des_cargo is not None:
                    if problem_type == ProblemType.SIZING and cargo != des_cargo:
                        # user has set all three check, if self consistent
                        cargo = des_cargo
                        if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                            warnings.warn(
                                f'Aircraft.CrewPayload.CARGO_MASS ({cargo} lbm) does not equal '
                                f'Aircraft.CrewPayload.Design.CARGO_MASS ({des_cargo} lbm) for '
                                'SIZING mission. Setting as-flown CARGO_MASS equal to '
                                f'Design.CARGO_MASS ({des_cargo} lbm)'
                            )
                else:
                    # user has set cargo & max: assume des = max
                    des_cargo = max_cargo
                    if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
                        warnings.warn(
                            'Aircraft.CrewPayload.Design.CARGO_MASS not set, assuming design cargo '
                            f'mass equals Design.MAX_CARGO_MASS ({max_cargo} lbm)'
                        )
            elif des_cargo is not None:
                # user has set cargo & des: assume max = des
                max_cargo = des_cargo
                if verbosity >= Verbosity.VERBOSE:  # BRIEF, VERBOSE, DEBUG
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
                if verbosity >= Verbosity.VERBOSE:  # BRIEF, VERBOSE, DEBUG:
                    warnings.warn(
                        'Aircraft.CrewPayload.CARGO_MASS not set, assuming no cargo flown on mission'
                    )
            else:
                # user has set max only: assume flown = des = 0
                cargo = des_cargo = 0
                if verbosity >= Verbosity.VERBOSE:  # BRIEF, VERBOSE, DEBUG:
                    warnings.warn(
                        'Aircraft has a maximum cargo capacity but no design cargo capacity. No '
                        'cargo will be flown on the mission.'
                    )

        elif des_cargo is not None:
            # user has only input des: assume max = des and flown = 0
            max_cargo = des_cargo
            cargo = 0
            if verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG:
                warnings.warn(
                    'Aircraft.CrewPayload.CARGO_MASS and Aircraft.CrewPayload.Design.MAX_CARGO_MASS '
                    'not set, assuming maximum cargo capacity equals design cargo mass'
                    f'({des_cargo} lbm).'
                )

        else:
            # user has input no cargo information
            cargo = max_cargo = des_cargo = 0

        # TODO add logic to handle these edge cases. Max cargo should probably get set to higher
        #      value (with warning)
        if cargo > max_cargo:
            raise UserWarning(
                f'Aircraft.CrewPayload.CARGO_MASS ({cargo} lbm) is greater than'
                f'Aircraft.CrewPayload.Design.MAX_CARGO_MASS ({max_cargo} lbm)'
            )
        if des_cargo > max_cargo:
            raise UserWarning(
                f'Aircraft.CrewPayload.Design.CARGO_MASS ({des_cargo} lbm) is greater than'
                f'Aircraft.CrewPayload.Design.MAX_CARGO_MASS ({max_cargo} lbm)'
            )

        # calculate passenger mass with bags based on user inputs.
        try:
            pax_mass_with_bag = aviary_options.get_val(
                Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, 'lbm'
            )
        except KeyError:
            pax_mass = aviary_options.get_val(Aircraft.CrewPayload.MASS_PER_PASSENGER, 'lbm')
            bag_mass = aviary_options.get_val(
                Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER, 'lbm'
            )
            pax_mass_with_bag = pax_mass + bag_mass
            aviary_options.set_val(
                Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, pax_mass_with_bag, 'lbm'
            )

        # calculate and check total payload
        # NOTE this is only used for error messaging, the calculations for analysis are subsystems/mass/gasp_based
        # No actual "preprocessing" happens here - a simple error message might be better suited
        # inside the calculation component itself (with a flag to make sure it only triggers once)
        design_passenger_payload_mass = design_pax * pax_mass_with_bag
        max_payload = design_passenger_payload_mass + max_cargo
        as_flown_passenger_payload_mass = mission_pax * pax_mass_with_bag
        as_flown_payload = as_flown_passenger_payload_mass + cargo
        if as_flown_payload > max_payload and verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
            warnings.warn(
                f'As-flown payload ({as_flown_payload} lbm) is greater than maximum payload '
                f'({max_payload} lbm). The aircraft will be undersized for this payload!'
            )

        # set assumed cargo mass variables:
        aviary_options.set_val(Aircraft.CrewPayload.CARGO_MASS, cargo, 'lbm')
        aviary_options.set_val(Aircraft.CrewPayload.Design.MAX_CARGO_MASS, max_cargo, 'lbm')
        aviary_options.set_val(Aircraft.CrewPayload.Design.CARGO_MASS, des_cargo, 'lbm')

    # Process FLOPS based crew variables
    if mass_method == LegacyCode.FLOPS:
        # TODO: check if this is excuted.
        # Check flight attendants
        if (
            Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS not in aviary_options
            or aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS) < 0
        ):
            flight_attendants_count = 0  # assume no passengers

            if 0 < design_pax:
                if design_pax < 51:
                    flight_attendants_count = 1

                else:
                    flight_attendants_count = design_pax // 40 + 1

            aviary_options.set_val(
                Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS, flight_attendants_count
            )

        if (
            Aircraft.CrewPayload.NUM_GALLEY_CREW not in aviary_options
            or aviary_options.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW) < 0
        ):
            galley_crew_count = 0  # assume no galley_crew

            if 150 < design_pax:
                galley_crew_count = design_pax // 250 + 1

            aviary_options.set_val(Aircraft.CrewPayload.NUM_GALLEY_CREW, galley_crew_count)
        else:
            galley_crew_count = aviary_options.get_val(Aircraft.CrewPayload.NUM_GALLEY_CREW)

        if Aircraft.CrewPayload.NUM_FLIGHT_CREW not in aviary_options:
            flight_crew_count = 3

            if design_pax < 151:
                flight_crew_count = 2

            aviary_options.set_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW, flight_crew_count)
        else:
            flight_crew_count = aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)

        # check cabin crew against flight attendants & galley crew
        if Aircraft.CrewPayload.NUM_CABIN_CREW in aviary_options:
            flight_attendants_count = aviary_options.get_val(
                Aircraft.CrewPayload.NUM_FLIGHT_ATTENDANTS
            )
            cabin_crew_calc = flight_attendants_count + galley_crew_count
            cabin_crew_provided = aviary_options.get_val(Aircraft.CrewPayload.NUM_CABIN_CREW)
            if cabin_crew_calc != cabin_crew_provided:
                if verbosity >= Verbosity.BRIEF:
                    warnings.warn(
                        f'Total number of cabin crew specified ({cabin_crew_provided}) does not '
                        f'equal sum of flight attendants and galley crew ({cabin_crew_calc}).'
                        f'Setting aircraft:crew_and_payload:num_cabin_crew to {cabin_crew_calc}'
                    )
                aviary_options.set_val(Aircraft.CrewPayload.NUM_CABIN_CREW, cabin_crew_calc)

        if Aircraft.CrewPayload.BAGGAGE_MASS_PER_PASSENGER not in aviary_options:
            baggage_mass_per_pax = 35.0
            if Aircraft.Design.RANGE in aviary_options:
                design_range = aviary_options.get_val(Aircraft.Design.RANGE, 'nmi')

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

        if Aircraft.Engine.NUM_FUSELAGE_ENGINES in aviary_options:
            num_fulselage_engines = aviary_options.get_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES)
            if isinstance(num_fulselage_engines, (list, np.ndarray)):
                num_fulselage_engines = num_fulselage_engines[0]

        if Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION not in aviary_options:
            if (
                Aircraft.Engine.NUM_FUSELAGE_ENGINES in aviary_options
                and num_fulselage_engines > 1
                and aviary_options.get_val(Aircraft.Design.TYPE) == AircraftTypes.TRANSPORT
            ):
                HHT = 1
                if verbosity >= Verbosity.BRIEF:
                    warnings.warn(
                        'Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION not specified and '
                        'fuselage-mounted engines present. Assuming T-Tail configuration '
                        '(Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION = 1)'
                    )
            else:
                # TODO this is the default, so it probably doesn't need to be called out here
                HHT = 0
                if verbosity >= Verbosity.VERBOSE:
                    warnings.warn(
                        'Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION not specified - '
                        'assuming conventional tail configuration '
                        '(Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION = 0)'
                    )
            aviary_options.set_val(
                Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION,
                val=HHT,
                units='unitless',
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
            # TODO these are default values, double check they need to actually be set here
            if Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY not in aviary_options:
                aviary_options.set_val(Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY, 0.0, 'lbm')

            if Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY not in aviary_options:
                aviary_options.set_val(Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY, 0.0, 'lbm')
        else:
            total_capacity = aviary_options.get_val(Aircraft.Fuel.TOTAL_CAPACITY, 'lbm')
            try:
                wing_capacity = aviary_options.get_val(Aircraft.Fuel.WING_FUEL_MASS_CAPACITY, 'lbm')
            except KeyError:
                wing_capacity = None
            try:
                fuselage_capacity = aviary_options.get_val(
                    Aircraft.Fuel.FUSELAGE_FUEL_MASS_CAPACITY, 'lbm'
                )
            except KeyError:
                fuselage_capacity = None
            try:
                auxiliary_capacity = aviary_options.get_val(
                    Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY, 'lbm'
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

            # check if the user inputs are self consistent (as far as possible at this stage!) Aviary can still calculate outputs at runtime.
            if capacity_count == 3 and capacity_check != total_capacity:
                raise UserWarning(
                    f'Aircraft.Fuel.TOTAL_CAPACITY ({total_capacity} lbm) is not equal to sum of '
                    f'Aircraft.Fuel.WING_FUEL_CAPACITY ({wing_capacity} lbm), '
                    f'+ Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY ({fuselage_capacity} lbm), and'
                    f'Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY ({auxiliary_capacity} lbm)'
                    f'(total of {capacity_check} lbm)'
                )
            elif capacity_count < 3 and capacity_check > total_capacity:
                raise UserWarning(
                    f'Aircraft.Fuel.TOTAL_CAPACITY ({total_capacity}) is less than sum of '
                    f'Aircraft.Fuel.WING_FUEL_CAPACITY ({wing_capacity} lbm), '
                    f'Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY ({fuselage_capacity} lbm), and '
                    f'Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY ({auxiliary_capacity} lbm) '
                    f'(total of {capacity_check} lbm)'
                )

    return aviary_options


def preprocess_propulsion(
    aviary_options: AviaryValues,
    engine_models: list = None,
    meta_data=CoreMetaData,
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
    update_list = list(complete_options_list.keys())

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
                    if isinstance(default_value, (tuple, list)):
                        dtype = type(default_value[0])
                    else:
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
                        if isiterable(aviary_val) and len(aviary_val) > 1:
                            if multidimensional:
                                vec.extend(aviary_val)
                            else:
                                try:
                                    # check variable exists at this index - if not, use default
                                    aviary_val[i]
                                except IndexError:
                                    vec.append(default_value)
                                else:
                                    # if aviary_val is an iterable, just grab val for this engine
                                    vec.append(aviary_val[i])
                        else:
                            if isiterable(aviary_val):
                                vec.extend(aviary_val)
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
    sum_engines_provided = False
    try:
        num_engines_all = aviary_options.get_val(Aircraft.Engine.NUM_ENGINES)
        sum_engines_provided = True
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
                if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
                    warnings.warn(
                        f'Mount location for engines of type <{eng_name}> not specified. Wing-'
                        'mounted engines are assumed.'
                    )
            else:
                num_fuse_engines_all[i] = num_engines
                if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
                    warnings.warn(
                        f'Mount location for single engine of type <{eng_name}> not specified.'
                        'Assuming it is fuselage-mounted.'
                    )

        # If wing mount type are specified but inconsistent, handle it
        elif total_engines_calc > num_engines:
            # more defined engine locations than number of engines - increase num engines
            num_engines_all[i] = total_engines_calc
            if sum_engines_provided and verbosity >= Verbosity.BRIEF:  # BRIEF, VERBOSE, DEBUG
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
                if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
                    warnings.warn(
                        'Mount location was not defined for all engines of EngineModel '
                        f'<{eng_name}> - unspecified engines are assumed wing-mounted.'
                    )
            elif num_unspecified_engines == 1 and num_wing_engines != 0:
                num_fuse_engines_all[i] = num_fuse_engines + num_unspecified_engines
                if verbosity >= Verbosity.VERBOSE:  # VERBOSE, DEBUG
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
