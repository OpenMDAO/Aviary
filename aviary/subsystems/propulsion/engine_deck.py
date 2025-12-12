"""
Define utilities for building engine decks.

Classes
-------
EngineDeck : the interface for an engine deck builder.

Aliases
-------
accepted_headers : dict
    The strings that are accepted as valid header names after converted to all lowercase
    with all whitespace removed, mapped to the enum EngineModelVariables.

default_required_variables : set
    Variables that must be present in an EngineDeck's DATA_FILE (Mach, altitude, etc.).
    Can be replaced by user-provided list.

required_options : tuple
    Options that must be present in an EngineDeck's options attribute.

dependent_options : dict
    Options that may or may not be required based on the presence or value of other
    provided options.
"""

import math
import warnings

import numpy as np
import openmdao.api as om
from openmdao.utils.units import convert_units

from aviary.interface.utils import round_it
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.engine_scaling import EngineScaling
from aviary.subsystems.propulsion.engine_sizing import SizeEngine
from aviary.subsystems.propulsion.utils import (
    EngineModelVariables,
    UncorrectData,
    convert_geopotential_altitude,
    default_units,
    max_variables,
)
from aviary.utils.aviary_values import AviaryValues, NamedValues, get_items, get_keys
from aviary.utils.csv_data_file import read_data_file
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

MACH = EngineModelVariables.MACH
ALTITUDE = EngineModelVariables.ALTITUDE
THROTTLE = EngineModelVariables.THROTTLE
HYBRID_THROTTLE = EngineModelVariables.HYBRID_THROTTLE
THRUST = EngineModelVariables.THRUST
TAILPIPE_THRUST = EngineModelVariables.TAILPIPE_THRUST
GROSS_THRUST = EngineModelVariables.GROSS_THRUST
SHAFT_POWER = EngineModelVariables.SHAFT_POWER
SHAFT_POWER_CORRECTED = EngineModelVariables.SHAFT_POWER_CORRECTED
RAM_DRAG = EngineModelVariables.RAM_DRAG
FUEL_FLOW = EngineModelVariables.FUEL_FLOW
ELECTRIC_POWER_IN = EngineModelVariables.ELECTRIC_POWER_IN
NOX_RATE = EngineModelVariables.NOX_RATE
TEMPERATURE = EngineModelVariables.TEMPERATURE_T4
RPM = EngineModelVariables.RPM

# EXIT_AREA = EngineModelVariables.EXIT_AREA

# EngineDeck assumes all aliases point to an enum, these are used internally only
aliases = {
    # whitespaces are replaced with underscores converted to lowercase before
    # comparison with keys
    MACH: ['m', 'mn', 'mach', 'mach_number'],
    ALTITUDE: ['altitude', 'alt', 'h'],
    THROTTLE: ['throttle', 'power_code', 'pc'],
    HYBRID_THROTTLE: [
        'hybrid_throttle',
        'hpc',
        'hybrid_power_code',
        'electric_throttle',
    ],
    THRUST: ['thrust', 'net_thrust'],
    GROSS_THRUST: ['gross_thrust'],
    RAM_DRAG: ['ram_drag'],
    FUEL_FLOW: ['fuel', 'fuel_flow', 'fuel_flow_rate'],
    ELECTRIC_POWER_IN: ['electric_power_in', 'electric_power'],
    NOX_RATE: ['nox', 'nox_rate'],
    TEMPERATURE: ['t4', 'temp', 'temperature', 'temperature_t4', 't4_temperature'],
    SHAFT_POWER: ['shaft_power', 'shp'],
    SHAFT_POWER_CORRECTED: ['shaft_power_corrected', 'shpcor', 'corrected_horsepower'],
    TAILPIPE_THRUST: ['tailpipe_thrust'],
    RPM: ['rpm', 'rotations_per_minute'],
}

# these variables must be present in engine performance data
default_required_variables = {MACH, ALTITUDE, THROTTLE, THRUST}

# EngineDecks internally require these options to have values. Input checks will set
# these options to default values in self.options if they are not provided
required_options = (
    Aircraft.Engine.SCALE_PERFORMANCE,
    Aircraft.Engine.IGNORE_NEGATIVE_THRUST,
    Aircraft.Engine.GEOPOTENTIAL_ALT,
    Aircraft.Engine.GENERATE_FLIGHT_IDLE,
    Aircraft.Engine.INTERPOLATION_METHOD,
    Aircraft.Engine.INTERPOLATION_SORT,
    # TODO fuel flow scaler is required for the EngineScaling component but does not need
    #      to be defined on a per-engine basis, so it could exist only in the problem-
    #      level aviary_options without issue. Is this a propulsion_preprocessor task?
    Mission.Summary.FUEL_FLOW_SCALER,
)

# options that are only required based on the value of another option
dependent_options = {
    Aircraft.Engine.GENERATE_FLIGHT_IDLE: (
        Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION,
        Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION,
        Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION,
    )
}


class EngineDeck(EngineModel):
    """
    EngineModel that obtains performance data from a tabular input file or memory.

    Attributes
    ----------
    name : str ('engine')
        Object label.
    options : AviaryValues (<empty>)
        Inputs and options related to engine model.
    data : NamedVaues (<empty>), optional
        Engine performance data (optional). If provided, used instead of tabular data
        file.
    required_variables : set, optional
        A set of required variables (from EngineModelVariables) for this EngineDeck.
        Defaults to the required set {ALTITUDE, MACH, THROTTLE, THRUST}.

    Methods
    -------
    build_pre_mission
    build_mission
    get_val
    set_val
    update
    """

    def __init__(
        self,
        name='engine_deck',
        options: AviaryValues = None,
        data: NamedValues = None,
        required_variables: set = default_required_variables,
        meta_data: dict = _MetaData,
    ):
        if data is not None:
            self.read_from_file = False
        else:
            self.read_from_file = True
            # TODO update default name to be based on filename

        # also calls _preprocess_inputs() as part of EngineModel __init__
        super().__init__(name, options, meta_data=meta_data)

        # custom error messages depending on data type
        if self.read_from_file:
            self.error_message = f'<{self.get_val(Aircraft.Engine.DATA_FILE)}>'
        else:
            self.error_message = f'EngineDeck <{self.name}>'

        # copy of raw data read from data_file or memory, never modified or used outside
        #     EngineDeck
        self._original_data = {key: np.array([]) for key in EngineModelVariables}
        # working copy of engine performance data, is modified during data pre-processing
        self.data = {key: np.array([]) for key in EngineModelVariables}

        # number of data points in engine data
        self.model_length = 0

        self.throttle_min = 0.0
        self.throttle_max = 1.0
        self.hybrid_throttle_min = 0.0
        self.hybrid_throttle_max = 1.0

        # absolute tolerance for how far apart two points must be to be counted as unique
        self.mach_tol = 0.01
        self.alt_tol = 10.0  # ft
        self.thrust_tol = 1  # lbf

        # Create dict for variables present in engine data with associated units
        self.engine_variables = {}

        if Aircraft.Engine.GLOBAL_THROTTLE in options:
            self.global_throttle = self.options.get_val(Aircraft.Engine.GLOBAL_THROTTLE)
        else:
            default = meta_data[Aircraft.Engine.GLOBAL_THROTTLE]['default_value']
            self.options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, default)
            self.global_throttle = default
        if Aircraft.Engine.GLOBAL_HYBRID_THROTTLE in options:
            self.global_hybrid_throttle = self.options.get_val(
                Aircraft.Engine.GLOBAL_HYBRID_THROTTLE
            )
        else:
            default = meta_data[Aircraft.Engine.GLOBAL_HYBRID_THROTTLE]['default_value']
            self.options.set_val(Aircraft.Engine.GLOBAL_HYBRID_THROTTLE, default)
            self.global_hybrid_throttle = default

        # ensure required variables are a set
        self.required_variables = {*required_variables}

        self._setup(data)

    def _preprocess_inputs(self):
        """
        Checks that provided options are valid and logically consistent. Raises errors
        for non-recoverable issues, issues warnings for minor problems that are fixed at
        runtime.

        Raises
        ------
        TypeError
            If provided options are not an instance of AviaryValues (or None).
        FileNotFoundError
            If the provided DATA_FILE cannot be found.
        """
        super()._preprocess_inputs()

        options = self.options

        # CHECK FOR REQUIRED OPTIONS
        additional_options = ()
        if self.read_from_file:
            additional_options = (Aircraft.Engine.DATA_FILE,)

        for key in additional_options + required_options:
            if key not in options:
                val = self.meta_data[key]['default_value']
                units = self.meta_data[key]['units']

                if self.get_val(Settings.VERBOSITY) > Verbosity.BRIEF:
                    warnings.warn(
                        f'<{key}> is a required option for EngineDecks, but has not been '
                        f'specified for EngineDeck <{self.name}>. The default value '
                        f'{val}{" " + units if units != "unitless" else ""} will '
                        'be used.'
                    )

                self.set_val(key, val, units)

        # check dependent options
        for key in dependent_options:
            if self.get_val(key):
                for item in dependent_options[key]:
                    if item not in options:
                        val = self.meta_data[item]['default_value']
                        units = self.meta_data[item]['units']
                        self.set_val(item, val, units)

        # LOGIC CHECKS
        if self.get_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE):
            idle_min = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION)
            idle_max = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION)
            # Allowing idle fractions to be equal, i.e. fixing flight idle conditions
            # instead of extrapolation
            if idle_min > idle_max:
                if self.get_val(Settings.VERBOSITY).value >= 1:
                    warnings.warn(
                        f'EngineDeck <{self.name}>: Minimum flight idle fraction '
                        'exceeds maximum flight idle fraction. Values for min and max '
                        'fraction will be flipped.'
                    )
                self.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, val=idle_max)
                self.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, val=idle_min)

        # check that sufficient information on engine scaling is provided
        # default behavior is to calculate scale factor based on thrust target
        engine_mapping = get_keys(self.options)

        # check if scale factor and thrust target are user defined and check consistency
        scale_performance = self.get_val(Aircraft.Engine.SCALE_PERFORMANCE)
        scale_factor_provided = False
        thrust_provided = False
        # was scale factor originally provided? (Not defaulted)
        if Aircraft.Engine.SCALE_FACTOR in engine_mapping:
            # if scale factor is 1, doesn't conflict with performance scaling turned off
            if self.options.get_val(Aircraft.Engine.SCALE_FACTOR) == 1:
                scale_factor_provided = False
            else:
                scale_factor_provided = True
        # was scaled thrust originally provided? (Not defaulted)
        if Aircraft.Engine.SCALED_SLS_THRUST in engine_mapping:
            thrust_provided = True

        # user provided target thrust or scale factor, but performance scaling is off
        if (
            not scale_performance
            and (scale_factor_provided or thrust_provided)
            and self.get_val(Settings.VERBOSITY).value >= 1
        ):
            warnings.warn(
                f'EngineDeck <{self.name}>: Scaling targets are provided, but will be '
                'ignored because performance scaling is disabled. Set '
                'aircraft:engine:scale_performance to True to enable scaling.'
            )

        # Check validity of interp_sort.
        # TODO: support this as an enum instead.
        interp_sort = self.get_val(Aircraft.Engine.INTERPOLATION_SORT)
        if interp_sort not in ['mach', 'altitude']:
            raise ValueError(
                f'EngineDeck <{self.name}>: Invalid value of Aircraft.Engine.INTERPOLATION_SORT.'
                f' Expected "altitude" or "mach", but found "{interp_sort}".'
            )

    def _set_variable_flags(self):
        """
        Sets flags in EngineDeck to communicate which (non-required) variables are
        available to greater propulsion module.
        """
        engine_variables = self.engine_variables

        # these flags are shortcuts for common checks for the presence of specific
        # variables in the engine deck
        self.use_thrust = (
            THRUST in engine_variables
            or TAILPIPE_THRUST in engine_variables
            or (GROSS_THRUST in engine_variables and RAM_DRAG in engine_variables)
        )
        self.use_hybrid_throttle = HYBRID_THROTTLE in engine_variables
        self.use_t4 = TEMPERATURE in engine_variables
        self.use_shaft_power = (
            SHAFT_POWER in engine_variables or SHAFT_POWER_CORRECTED in engine_variables
        )

    def _setup(self, data):
        """
        Read in and process engine data.
        - Check data consistency.
        - Convert altitudes to geometric.
        - Sort and pack data.
        - Determine reference thrust.
        - Normalize throttles & hybrid throttles.
        - Fill flight idle points if requested.
        """
        self._read_data(data)

        # perform consistency checks on data
        self._check_data()

        # convert geopotential altitude to geometric if required
        if self.get_val(Aircraft.Engine.GEOPOTENTIAL_ALT):
            self.data[ALTITUDE] = convert_geopotential_altitude(self.data[ALTITUDE])

        # sort and organize data
        self._pack_data()

        if self.use_thrust:
            # assign reference sls thrust from engine deck, perform sanity checks
            self._set_reference_thrust()

        # normalize throttle and hybrid throttle (if included) to |0-1| scale
        self._normalize_throttle()

        # extrapolate flight idle data if requested
        if self.get_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE):
            self._generate_flight_idle()

    def _read_data(self, raw_data: NamedValues):
        """
        Import tabular engine data; either from memory or from a data file.

        Parameters
        ----------
        raw_data : NamedValues (optional)
            Data provided via a NamedValues object. Will be used as data source instead
            of Aircraft.Engine.DATA_FILE if self.read_from_file is False

        Raises
        ------
        ValueError
            If non-numerical data found in DATA_FILE (not including header or comments).
        """
        # get data (as NamedValues object) from data file
        if self.read_from_file:
            data_file = self.get_val(Aircraft.Engine.DATA_FILE)

            # read csv file - currently not saving comments
            raw_data, self.inputs, self.outputs = read_data_file(data_file, aliases=aliases)

        else:
            # run provided data through aliases
            # create dict of what names to change, modify outside of loop
            alias_dict = {}
            for item in get_items(raw_data):
                var = item[0]
                val = item[1][0]
                units = item[1][1]
                # quick "reverse" lookup of aliases
                for key in aliases:
                    if var in aliases[key]:
                        alias_dict[var] = key
                        break

            # replace old names with aliased ones
            for name in alias_dict:
                val, units = raw_data.get_item(name)
                raw_data.delete(name)
                raw_data.set_val(alias_dict[name], val, units)

        # Loop through all variables in provided data. Track which valid variables are
        #    included with the data and save raw data for reference
        for key in get_keys(raw_data):
            val, units = raw_data.get_item(key)
            if key in aliases:
                # Convert data to expected units. Required so settings like tolerances
                # that assume units work as expected
                try:
                    val = np.array([convert_units(i, units, default_units[key]) for i in val])
                except TypeError:
                    raise TypeError(
                        f"{self.error_message}: units of '{units}' provided for "
                        f'<{key.name}> are not compatible with expected '
                        f'units of {default_units[key]}'
                    )

                # Engine_variables currently only used to store "valid" engine variables
                # as defined in EngineModelVariables Enum
                self.engine_variables[key] = default_units[key]

            else:
                if self.get_val(Settings.VERBOSITY) >= Verbosity.BRIEF:
                    warnings.warn(
                        f'{self.error_message}: header <{key}> was not recognized, and will be skipped'
                    )

            # save all data in self._original_data, including skipped variables
            self._original_data[key] = val

        if not self.engine_variables:
            raise UserWarning(f'No valid engine variables found in data for {self.error_message}')

        # Copy data from original data (never modified) to working data (changed through
        #    sorting, generating missing data, etc.)
        for key in self.data:
            self.data[key] = self._original_data[key]

    def _check_data(self):
        """
        Checks for consistency of provided thrust and drag data, ensures no required
        variables are missing, fills unused variables with a default value of zero, and
        removes negative thrusts if requested.

        Raises
        ------
        UserWarning
            If provided net thrust does not match difference between provided gross
            thrust and ram drag within tolerance.
        UserWarning
            If required variables are not present in the provided engine data.
        """
        original_data = self._original_data
        data = self.data

        engine_variables = self.engine_variables

        # Handle ram drag, net and gross thrust and potential conflicts in value or units
        # Warn user if they provide partial info for calculated thrust
        # Not a fail state if net thrust is still provided
        # If both net thrust and components for calculated thrust both provided, a sanity
        #   check that they match is done after reading data
        if THRUST in engine_variables:
            # if thrust is present, but gross thrust or ram drag also present raise warning
            if GROSS_THRUST in engine_variables and RAM_DRAG not in engine_variables:
                warnings.warn(
                    f'{self.error_message} contains both net and gross thrust. Only net thrust will be used.'
                )
            if GROSS_THRUST not in engine_variables and RAM_DRAG in engine_variables:
                warnings.warn(
                    f'{self.error_message} contains both net thrust '
                    'and ram drag. Only net thrust will be used.'
                )

        if RAM_DRAG in engine_variables and GROSS_THRUST in engine_variables:
            # Check that units are the same. Variables have already been checked for valid
            # units, so it is assumed they are convertible. Prioritizes thrust units
            if engine_variables[RAM_DRAG] != engine_variables[GROSS_THRUST]:
                data[RAM_DRAG] = convert_units(
                    original_data[RAM_DRAG],
                    engine_variables[RAM_DRAG],
                    engine_variables[GROSS_THRUST],
                )
                engine_variables[RAM_DRAG] = engine_variables[GROSS_THRUST]

            net_thrust_calc = data[GROSS_THRUST] - data[RAM_DRAG]
            # prefer using directly provided values for net thrust vs. calculating
            if THRUST in engine_variables:
                res = abs(net_thrust_calc - original_data[THRUST])
                if np.any(self.thrust_tol > res):
                    raise UserWarning(
                        'Provided net thrust is not equal to difference '
                        '(within tolerance) between gross thrust and ram '
                        f'drag in {self.error_message}'
                    )
            else:
                # store net thrust in THRUST key instead of gross thrust
                data[THRUST] = net_thrust_calc
                engine_variables[THRUST] = engine_variables[GROSS_THRUST]

        if TAILPIPE_THRUST in engine_variables:
            # tailpipe thrust is not bookept separately in Aviary. Add to net thrust.
            if THRUST in engine_variables:
                # Check that units are the same. Variables have already been checked for valid
                # units, so it is assumed they are convertible. Prioritizes thrust units
                if engine_variables[THRUST] != engine_variables[TAILPIPE_THRUST]:
                    data[TAILPIPE_THRUST] = convert_units(
                        data,
                        engine_variables[TAILPIPE_THRUST],
                        engine_variables[GROSS_THRUST],
                    )
                    engine_variables[THRUST] = engine_variables[TAILPIPE_THRUST]
                data[THRUST] = data[THRUST] + data[TAILPIPE_THRUST]
            else:
                data[THRUST] = original_data[TAILPIPE_THRUST]
                engine_variables[THRUST] = engine_variables[TAILPIPE_THRUST]

        # remove now unneeded dependent variables from engine_variables and self.data
        if RAM_DRAG in engine_variables:
            engine_variables.pop(RAM_DRAG)
            self.data.pop(RAM_DRAG)
        if GROSS_THRUST in engine_variables:
            engine_variables.pop(GROSS_THRUST)
            self.data.pop(GROSS_THRUST)
        if TAILPIPE_THRUST in engine_variables:
            engine_variables.pop(TAILPIPE_THRUST)
            self.data.pop(TAILPIPE_THRUST)

        # Handle shaft power (corrected and uncorrected). It is not possible to compare
        # them for consistency, as that requires information not available during setup
        # (freestream air temp and pressure). Instead, we must trust the source and
        # assume either data set is valid and can be used.
        if (
            SHAFT_POWER in engine_variables
            and SHAFT_POWER_CORRECTED in engine_variables
            and self.get_val(Settings.VERBOSITY) >= Verbosity.BRIEF
        ):
            warnings.warn(
                'Both corrected and uncorrected shaft horsepower are '
                f'present in {self.error_message}. The two cannot be validated for '
                'consistency, and only uncorrected shaft power will be used.'
            )
            engine_variables.pop(SHAFT_POWER_CORRECTED)
            self.data.pop(SHAFT_POWER_CORRECTED)

        self._set_variable_flags()

        # check that all required variables are present in engine data
        if not self.required_variables.issubset(engine_variables):
            # gather all missing required variables
            missing_variables = set()
            for var in engine_variables:
                if var in self.required_variables:
                    missing_variables.add(var)

            # if missing_variables is not empty
            if not missing_variables:
                raise UserWarning(
                    f'Required variables {missing_variables} are missing from {self.error_message}'
                )

        # removes data points with negative thrust if requested
        if self.get_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST):
            keep_idx = np.where(data[THRUST] >= 0)
            for key in data:
                data[key] = data[key][keep_idx]

        self.model_length = len(self.data[ALTITUDE])

        # Set all unused variables to default value of zero
        for key in data:
            if not len(data[key]):
                data[key] = np.zeros(self.model_length)

        # set flags using updated engine_variables
        self._set_variable_flags()

    def _generate_flight_idle(self):
        """
        Generate flight idle data via extrapolation from lowest points in data set,
        bound by upper and lower constraints set by user.

        Requires sorted, packed data with normalized throttles.

        Modifies unpacked data in place, updates packed data.
        """

        def _extrapolate(array):
            """
            Linearly extrapolate variable to idle thrust point.

            Parameters
            ----------
            array : numpy.ndarray
                Data used for extrapolation.

            Returns
            -------
            rvalue : float
                Extrapolated flight idle value.
            """
            y0 = array[0]
            y1 = array[1]

            if y0 == 0 and y1 == 0:
                return 0

            rvalue = y0 + (y1 - y0) * extrap_term

            return rvalue

        idle_thrust_fract = self.get_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION)
        idle_min_fract = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION)
        idle_max_fract = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION)

        packed_data = self.packed_data

        # variables whose idle value is directly calculated based on FLIGHT_IDLE_THRUST_FRACTION
        direct_calc_vars = []
        if THRUST in self.engine_variables:
            direct_calc_vars.append(THRUST)
        if SHAFT_POWER_CORRECTED in self.engine_variables:
            direct_calc_vars.append(SHAFT_POWER_CORRECTED)
        if SHAFT_POWER in self.engine_variables:
            direct_calc_vars.append(SHAFT_POWER)

        # stored information about packed data
        mach_max_count = self.mach_max_count
        alt_max_count = self.alt_max_count
        data_indices = self.data_indices

        # Throttle is already normalized from 0 to 1. Set flight idle to -0.1, which will
        # get re-normalized to 0
        # -0.1 is chosen to avoid stretching out the data range while at the same time
        # avoiding "discontinuities" in engine data from arbitrarily small negative
        # throttle (e.g. -1e-6). Basically, this is an arbitrary number
        throttle_idle = -0.1
        hybrid_throttle_idle = 0

        idle_points = {key: np.empty(0) for key in packed_data}

        # Normally, only one idle point is needed - however, when hybrid throttle is
        # present, there needs to be a sweep of points for a given Mach/alt/throttle
        # to satisfy the interpolator's requirements for at least 3 points per dimension
        # The data values at each point in the sweep are kept identical (e.g. same thrust,
        # fuel flow, etc. as calculated by extrapolation)
        num_points = 1
        if self.use_hybrid_throttle:
            num_points = 3
            # How far apart the "fake" points should be from the actual idle point
            # This time, we want an arbitrarily small number
            h_tol = 1e-4

        for M in range(mach_max_count):
            for A in range(alt_max_count):
                # if no data at this Mach, alt index combination, skip
                if data_indices[M, A] == 0:
                    continue

                # don't generate flight idle points if thrust is already zero or negative
                # at lowest index
                if packed_data[THRUST][M, A, 0] <= self.thrust_tol:
                    continue

                # define known data for idle point (independent variables)
                idle_points[MACH] = np.append(
                    idle_points[MACH], [packed_data[MACH][M, A, 0]] * num_points
                )
                idle_points[ALTITUDE] = np.append(
                    idle_points[ALTITUDE], [packed_data[ALTITUDE][M, A, 0]] * num_points
                )
                idle_points[THROTTLE] = np.append(
                    idle_points[THROTTLE], [throttle_idle] * num_points
                )
                if self.use_hybrid_throttle:
                    hybrid_throttle_range = np.linspace(
                        hybrid_throttle_idle - h_tol,
                        hybrid_throttle_idle + h_tol,
                        num_points,
                    )
                    idle_points[HYBRID_THROTTLE] = np.append(
                        idle_points[HYBRID_THROTTLE], hybrid_throttle_range
                    )
                else:
                    idle_points[HYBRID_THROTTLE] = np.append(
                        idle_points[HYBRID_THROTTLE], hybrid_throttle_idle
                    )

                # if there is only one data point at this Mach, alt combination, use
                # thrust fraction instead of extrapolation
                if data_indices[M, A] == 1:
                    # Find the point closest to hybrid throttle idle (0) if hybrid throttle is present
                    if self.use_hybrid_throttle:
                        hybrid_throttle_data = packed_data[HYBRID_THROTTLE][M, A]
                        # Find index of point closest to hybrid throttle idle (0)
                        idle_idx = np.argmin(np.abs(hybrid_throttle_data))
                    else:
                        idle_idx = 0

                    for key in packed_data:
                        if (
                            key
                            not in [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE] + direct_calc_vars
                        ):
                            idle_value = packed_data[key][M, A, idle_idx] * idle_thrust_fract
                            var_min = packed_data[key][M, A, -1] * idle_min_fract
                            var_max = packed_data[key][M, A, -1] * idle_max_fract

                            if idle_value < var_min:
                                idle_value = var_min
                            elif idle_value > var_max:
                                idle_value = var_max

                            idle_points[key] = np.append(
                                idle_points[key], [idle_value] * num_points
                            )

                    # thrust, shaft powers do not get idle_min/max checks
                    for var in direct_calc_vars:
                        idle_points[var] = np.append(
                            idle_points[var],
                            [[packed_data[var][M, A, idle_idx] * idle_thrust_fract]] * num_points,
                        )
                    # move to next data point
                    continue

                # calculate idle thrust, shaft powers as a percentage of max thrust at Mach, alt point
                for var in direct_calc_vars:
                    idle_calc_value = (
                        packed_data[var][M, A, data_indices[M, A] - 1] * idle_thrust_fract
                    )

                    # add this point to idle_points
                    idle_points[var] = np.append(idle_points[var], [idle_calc_value] * num_points)

                    # Calculate term for linear extrapolation - shaft power has highest
                    # "preference" since it is last in the list, followed by corrected
                    # shaft power then finally thrust. This is designed for compatibility
                    # with turboshaft engine decks in TurbopropModels.
                    # Only one extrapolation term can be used for all dependent vars
                    extrap_term = (idle_calc_value - packed_data[var][M, A, 0]) / (
                        packed_data[var][M, A, 1] - packed_data[var][M, A, 0]
                    )

                # compute idle data
                for key in packed_data:
                    # skip independent variables or thrust, which is already calculated
                    if key not in [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE] + direct_calc_vars:
                        # extrapolate to idle from lowest two throttle points in data
                        idle_value = _extrapolate(packed_data[key][M, A])

                        # idle cannot be below or above user-set limits
                        var_min = packed_data[key][M, A, -1] * idle_min_fract
                        var_max = packed_data[key][M, A, -1] * idle_max_fract

                        if idle_value < var_min:
                            idle_value = var_min
                        elif idle_value > var_max:
                            idle_value = var_max

                        # store newly computed idle point
                        idle_points[key] = np.append(idle_points[key], [idle_value] * num_points)

        # add idle points to data
        for key in packed_data:
            self.data[key] = np.append(self.data[key], idle_points[key])

        # update model length
        self.model_length = len(self.data[ALTITUDE])

        # save idle points, in case they are wanted later
        self.idle_points = idle_points

        # Re-sort and re-pack data with flight idle information to keep data
        # structures consistent
        self._pack_data()

        # Re-normalize throttle since "dummy" idle values were used
        self._normalize_throttle()

    def build_pre_mission(self, aviary_inputs, **kwargs) -> om.ExplicitComponent:
        """
        Build components to be added to pre-mission propulsion subsystem.

        Returns
        -------
            SizeEngine component specific to this EngineDeck, used for calculating engine
            scaling factors.
        """
        return SizeEngine()

    def _build_engine_interpolator(self, num_nodes, aviary_inputs):
        """
        Builds the OpenMDAO metamodel component for the engine deck.
        Currently only the semistructured model is supported.
        """
        interp_method = self.get_val(Aircraft.Engine.INTERPOLATION_METHOD)
        interp_sort = self.get_val(Aircraft.Engine.INTERPOLATION_SORT)
        # interpolator object for engine data
        engine = om.MetaModelSemiStructuredComp(
            method=interp_method, extrapolate=True, vec_size=num_nodes
        )

        units = default_units
        for key in self.engine_variables:
            units[key] = self.engine_variables[key]
        self.engine_variable_units = units

        # add inputs and outputs to interpolator
        # independent variables that currently MUST be inputs
        if interp_sort == 'altitude':
            independent_variables = [ALTITUDE, MACH, THROTTLE, HYBRID_THROTTLE]
        else:
            independent_variables = [MACH, ALTITUDE, THROTTLE, HYBRID_THROTTLE]

        if self.inputs == []:
            self.inputs = independent_variables
        else:
            for var in independent_variables:
                if var in self.outputs:
                    raise UserWarning(
                        f'Variable {var} is defined as an output in {self.error_message}, but '
                        'Aviary requires it to be an input.'
                    )

        no_scale_variables = [TEMPERATURE, RPM]

        # Add the first table inputs in the requested order.
        for variable in independent_variables:
            if variable in self.inputs and variable in self.engine_variables:
                engine.add_input(
                    variable.value,
                    self.data[variable],
                    units=default_units[variable],
                )

        # Add the remaining variables.
        for variable in self.engine_variables:
            if variable in self.inputs:
                if variable in independent_variables:
                    # Already handled above.
                    continue

                engine.add_input(
                    variable.value,
                    self.data[variable],
                    units=default_units[variable],
                )

            else:
                # don't append 'unscaled' to variables that will not be passed to scaling
                if variable in no_scale_variables:
                    var_name = variable.value
                else:
                    var_name = variable.value + '_unscaled'
                engine.add_output(
                    var_name,
                    self.data[variable],
                    units=default_units[variable],
                )

        return engine

    def build_mission(self, num_nodes, aviary_inputs, **kwargs) -> om.Group:
        """
        Creates interpolator objects to be added to mission-level propulsion subsystem.
        Interpolators must be re-generated for each ODE due to potentially different
        num_nodes in each mission segment.

        Parameters
        ----------
        num_nodes : int
            Number of nodes present in the current Dymos phase of mission analysis.

        Returns
        -------
        engine_group : openmdao.core.Group
            An OpenMDAO group containing engine data interpolators, an EngineScaling
            component, and max throttle/max hybrid_throttle generating components as
            needed for this EngineDeck.
        """
        interp_method = self.get_val(Aircraft.Engine.INTERPOLATION_METHOD)
        interp_sort = self.get_val(Aircraft.Engine.INTERPOLATION_SORT)

        engine_group = om.Group()

        engine = self._build_engine_interpolator(num_nodes, aviary_inputs)
        units = self.engine_variable_units

        # Create copy of interpolation component that computes max thrust/shp for current
        # flight condition
        # NOTE max thrust is assumed to occur at maximum throttle and hybrid throttle
        #      for each flight condition
        # TODO Use solver to find throttle/hybrid throttle for maximum thrust at given flight condition?
        #      Pre-solve max throttle/hybrid throttle for each flight condition, interpolate on
        #      reduced data set?
        if self.use_thrust or self.use_shaft_power:
            if self.global_throttle or (self.global_hybrid_throttle and self.use_hybrid_throttle):
                # create IndepVarComp to pass maximum throttle is to max thrust interpolator
                fixed_throttles = om.IndepVarComp()
                if self.global_throttle:
                    fixed_throttles.add_output(
                        'throttle_max',
                        val=np.ones(num_nodes) * self.throttle_max,
                        units='unitless',
                        desc='Engine maximum throttle',
                    )
                if self.global_hybrid_throttle and self.use_hybrid_throttle:
                    fixed_throttles.add_output(
                        'hybrid_throttle_max',
                        val=np.ones(num_nodes) * self.hybrid_throttle_max,
                        units='unitless',
                        desc='Engine maximum hybrid throttle',
                    )
            if not (
                self.global_throttle or (self.global_hybrid_throttle and self.use_hybrid_throttle)
            ):
                interp_throttles = om.MetaModelSemiStructuredComp(
                    method=interp_method, extrapolate=False, vec_size=num_nodes
                )

                packed_data = self.packed_data
                mach_table = np.array([])
                alt_table = np.array([])

                for M in range(self.mach_max_count):
                    for A in range(self.alt_max_count):
                        if self.data_indices[M, A] != 0:
                            mach_table = np.append(mach_table, packed_data[MACH][M, A, 0])
                            alt_table = np.append(alt_table, packed_data[ALTITUDE][M, A, 0])

                # add inputs and outputs to interpolator
                if interp_sort == 'altitude':
                    interp_throttles.add_input(
                        Dynamic.Mission.ALTITUDE,
                        alt_table,
                        units=units[ALTITUDE],
                        desc='Current flight altitude',
                    )
                    interp_throttles.add_input(
                        Dynamic.Atmosphere.MACH,
                        mach_table,
                        units='unitless',
                        desc='Current flight Mach number',
                    )
                else:
                    interp_throttles.add_input(
                        Dynamic.Atmosphere.MACH,
                        mach_table,
                        units='unitless',
                        desc='Current flight Mach number',
                    )
                    interp_throttles.add_input(
                        Dynamic.Mission.ALTITUDE,
                        alt_table,
                        units=units[ALTITUDE],
                        desc='Current flight altitude',
                    )

                if not self.global_throttle:
                    interp_throttles.add_output(
                        'throttle_max',
                        self.throttle_max,
                        units='unitless',
                        desc='max throttle available at current flight condition',
                    )
                if not self.global_hybrid_throttle and self.use_hybrid_throttle:
                    interp_throttles.add_output(
                        'hybrid_throttle_max',
                        self.hybrid_throttle_max,
                        units='unitless',
                        desc='max hybrid throttle available at current flight condition',
                    )

            # Calculation of max thrust currently done with a duplicate of the engine
            # model and scaling components
            max_thrust_engine = om.MetaModelSemiStructuredComp(
                method=interp_method, extrapolate=False, vec_size=num_nodes
            )

            if interp_sort == 'altitude':
                max_thrust_engine.add_input(
                    Dynamic.Mission.ALTITUDE,
                    self.data[ALTITUDE],
                    units=units[ALTITUDE],
                    desc='Current flight altitude',
                )
                max_thrust_engine.add_input(
                    Dynamic.Atmosphere.MACH,
                    self.data[MACH],
                    units='unitless',
                    desc='Current flight Mach number',
                )
            else:
                max_thrust_engine.add_input(
                    Dynamic.Atmosphere.MACH,
                    self.data[MACH],
                    units='unitless',
                    desc='Current flight Mach number',
                )
                max_thrust_engine.add_input(
                    Dynamic.Mission.ALTITUDE,
                    self.data[ALTITUDE],
                    units=units[ALTITUDE],
                    desc='Current flight altitude',
                )

            # replace throttle coming from mission with max value based on flight condition
            max_thrust_engine.add_input(
                'throttle_max',
                self.data[THROTTLE],
                units='unitless',
                desc='Current engine throttle',
            )
            if self.use_hybrid_throttle:
                # replace hybrid throttle coming from mission with max value based on
                # flight condition
                max_thrust_engine.add_input(
                    'hybrid_throttle_max',
                    self.data[HYBRID_THROTTLE],
                    units='unitless',
                    desc='Current engine hybrid throttle',
                )
            max_thrust_engine.add_output(
                'thrust_net_max_unscaled',
                self.data[THRUST],
                units=units[THRUST],
                desc='maximum thrust that can currently be produced',
            )
        if self.use_shaft_power:
            if SHAFT_POWER in self.engine_variables:
                max_thrust_engine.add_output(
                    'shaft_power_max_unscaled',
                    self.data[SHAFT_POWER],
                    units=units[SHAFT_POWER],
                    desc='maximum shaft power that can currently be produced',
                )
            else:
                max_thrust_engine.add_output(
                    'shaft_power_corrected_max_unscaled',
                    self.data[SHAFT_POWER_CORRECTED],
                    units=units[SHAFT_POWER_CORRECTED],
                    desc='maximum corrected shaft power that can currently be produced',
                )

        # add created subsystems to engine_group
        outputs = []
        if getattr(self, 'use_t4', False):
            outputs.append(Dynamic.Vehicle.Propulsion.TEMPERATURE_T4)

        engine_group.add_subsystem(
            'interpolation', engine, promotes_inputs=['*'], promotes_outputs=outputs
        )

        # check if uncorrection component is needed
        uncorrect_shp = False
        if (
            SHAFT_POWER_CORRECTED in self.engine_variables
            and SHAFT_POWER not in self.engine_variables
        ):
            uncorrect_shp = True
            engine_group.add_subsystem(
                'uncorrect_shaft_power',
                subsys=UncorrectData(num_nodes=num_nodes),
                promotes_inputs=[
                    Dynamic.Atmosphere.TEMPERATURE,
                    Dynamic.Atmosphere.STATIC_PRESSURE,
                    Dynamic.Atmosphere.MACH,
                ],
            )

            engine_group.connect(
                'interpolation.shaft_power_corrected_unscaled',
                'uncorrect_shaft_power.corrected_data',
            )

        if self.use_thrust or self.use_shaft_power:
            if self.global_throttle or (self.global_hybrid_throttle and self.use_hybrid_throttle):
                engine_group.add_subsystem(
                    'fixed_max_throttles', fixed_throttles, promotes_outputs=['*']
                )

            if not (
                self.global_throttle or (self.global_hybrid_throttle and self.use_hybrid_throttle)
            ):
                engine_group.add_subsystem(
                    'interp_max_throttles',
                    interp_throttles,
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

            engine_group.add_subsystem(
                'max_interpolation', max_thrust_engine, promotes_inputs=['*']
            )

            if uncorrect_shp:
                engine_group.add_subsystem(
                    'uncorrect_max_shaft_power',
                    subsys=UncorrectData(
                        num_nodes=num_nodes,
                    ),
                    promotes_inputs=[
                        Dynamic.Atmosphere.TEMPERATURE,
                        Dynamic.Atmosphere.STATIC_PRESSURE,
                        Dynamic.Atmosphere.MACH,
                    ],
                )

                engine_group.connect(
                    'max_interpolation.shaft_power_corrected_max_unscaled',
                    'uncorrect_max_shaft_power.corrected_data',
                )

        engine_outputs = self.engine_variables.copy()
        if SHAFT_POWER_CORRECTED in engine_outputs:
            shp_units = engine_outputs.pop(SHAFT_POWER_CORRECTED)
            engine_outputs[SHAFT_POWER] = shp_units

        engine_group.add_subsystem(
            'engine_scaling',
            subsys=EngineScaling(
                num_nodes=num_nodes,
                engine_variables=engine_outputs,
            ),
            promotes_inputs=[Aircraft.Engine.SCALE_FACTOR, Dynamic.Atmosphere.MACH],
            promotes_outputs=['*'],
        )

        # manually connect unscaled variables, since we do not want them promoted
        # skip variables that are not outputs of scaling component
        skipped_variables = [
            MACH,
            ALTITUDE,
            THROTTLE,
            HYBRID_THROTTLE,
            TEMPERATURE,
            SHAFT_POWER_CORRECTED,
            RPM,
        ]

        for variable in self.engine_variables:
            if variable not in skipped_variables:
                engine_group.connect(
                    'interpolation.' + variable.value + '_unscaled',
                    'engine_scaling.' + variable.value + '_unscaled',
                )
            if variable in max_variables:
                engine_group.connect(
                    'max_interpolation.' + variable.value + '_max_unscaled',
                    'engine_scaling.' + variable.value + '_max_unscaled',
                )

        if self.use_shaft_power and SHAFT_POWER not in self.engine_variables:
            engine_group.connect(
                'uncorrect_shaft_power.uncorrected_data',
                'engine_scaling.shaft_power_unscaled',
            )
            engine_group.connect(
                'uncorrect_max_shaft_power.uncorrected_data',
                'engine_scaling.shaft_power_max_unscaled',
            )

        return engine_group

    def get_parameters(self):
        params = {
            Aircraft.Engine.SCALE_FACTOR: {
                'val': 1.0,
                'units': 'unitless',
                'static_target': True,
            }
        }
        return params

    def report(self, problem, reports_file, **kwargs):
        meta_data = kwargs['meta_data']
        engine_idx = kwargs['engine_idx']

        outputs = [
            Aircraft.Engine.NUM_ENGINES,
            Aircraft.Engine.SCALED_SLS_THRUST,
            Aircraft.Engine.SCALE_FACTOR,
        ]

        # modified version of markdown table until adjusted to handle engine decks
        with open(reports_file, mode='a') as f:
            f.write(f'\n### {self.name}')
            f.write('\n| Variable Name | Value | Units |\n')
            f.write('| :- | :- | :- |\n')
            for var_name in outputs:
                # get default units from metadata
                try:
                    units = meta_data[var_name]['units']
                except KeyError:
                    units = None
                # try to get value from engine
                try:
                    if units:
                        val = self.get_val(var_name, units)
                    else:
                        val, units = self.get_item(var_name)
                        if (val, units) == (None, None):
                            raise KeyError
                except KeyError:
                    # get value from problem
                    try:
                        if units:
                            val = problem.get_val(var_name, units)
                        else:
                            # TODO find units for variable in problem?
                            val = problem.get_val(var_name)
                            units = 'unknown'
                    # variable not in problem, get from aviary_inputs instead
                    except KeyError:
                        try:
                            if units:
                                val = problem.aviary_inputs.get_val(var_name, units)
                            else:
                                val, units = problem.aviary_inputs.get_item(var_name)
                                if (val, units) == (None, None):
                                    raise KeyError
                        except KeyError:
                            val = 'Not Found in Model'
                            units = None
                        else:
                            val = val[engine_idx]
                    else:
                        val = val[engine_idx]
                # handle rounding + formatting
                if isinstance(val, (np.ndarray, list, tuple)):
                    val = [round_it(item) for item in val]
                    # if an interable with a length of 1, remove bracket/parentheses, etc.
                    if len(val) == 1:
                        val = val[0]
                else:
                    round_it(val)
                if not units:
                    units = 'unknown'
                if units == 'unitless':
                    units = '-'
                summary_line = f'| {var_name} | {val} | {units} |\n'
                f.write(summary_line)

    def _set_reference_thrust(self):
        """
        Determine maximum sea-level static thrust produced by the engine (unscaled).

        Reference thrust can instead be directly provided in options, intended for use in
        cases where the deck does not include a SLS point or if a different point would
        make more physical sense for scaling.

        Perform consistency checks on thrust-scaling options based on new reference
        thrust.
        """
        engine_mapping = get_keys(self.options)

        # Find reference thrust if not provided - assumed user-provided value is "best"
        # estimate of reference thrust
        if Aircraft.Engine.REFERENCE_SLS_THRUST not in engine_mapping:
            alt_tol = self.alt_tol
            mach_tol = self.mach_tol
            # NOTE This fails if there is no data point at SLS (within tolerance)
            sea_level_idx = np.intersect1d(
                np.where(-alt_tol < self.data[ALTITUDE])[0],
                np.where(self.data[ALTITUDE] <= alt_tol)[0],
            )
            static_idx = np.intersect1d(
                np.where(-mach_tol < self.data[MACH]),
                np.where(self.data[MACH] < self.mach_tol),
            )
            sls_idx = np.intersect1d(sea_level_idx, static_idx)

            if sls_idx.size == 0:
                raise UserWarning(
                    'Could not find sea-level static max thrust point for EngineDeck '
                    f'<{self.name}>. Please review the data file '
                    f'<{self.get_val(Aircraft.Engine.DATA_FILE)}> or manually specify '
                    'aircraft:engine:reference_sls_thrust in EngineDeck options'
                )

            reference_sls_thrust = max(self.data[THRUST][sls_idx])

            if self.get_val(Settings.VERBOSITY) >= Verbosity.VERBOSE:
                print(
                    f'EngineDeck <{self.name}>: found reference SLS thrust of '
                    f'{reference_sls_thrust} {self.engine_variables[THRUST]} in '
                    'provided performance data'
                )

            self.set_val(
                Aircraft.Engine.REFERENCE_SLS_THRUST,
                reference_sls_thrust,
                units=self.engine_variables[THRUST],
            )

        # Update SCALED_SLS_THRUST if required based on scaling information provided
        scale_performance = self.get_val(Aircraft.Engine.SCALE_PERFORMANCE)
        scale_factor_provided = False
        thrust_provided = False
        # was scale factor originally provided? (Not defaulted)
        if Aircraft.Engine.SCALE_FACTOR in engine_mapping:
            scale_factor_provided = True
        # was scaled thrust originally provided? (Not defaulted)
        if Aircraft.Engine.SCALED_SLS_THRUST in engine_mapping:
            thrust_provided = True
        ref_thrust = self.get_val(Aircraft.Engine.REFERENCE_SLS_THRUST, 'lbf')

        # logic tree when scale factor provided
        if scale_factor_provided:
            scale_factor = self.get_val(Aircraft.Engine.SCALE_FACTOR)
            # both scale factor and target thrust provided:
            if thrust_provided:
                scaled_thrust = self.get_val(Aircraft.Engine.SCALED_SLS_THRUST, 'lbf')
                if scale_performance:
                    # Check if target thrust and ref thrust * scale factor match using
                    # rough tolerance
                    # Tolerance is arbitrary, but designed to handle thrusts in the
                    # hundreds of thousands via rel_tol, and the hundreds via abs_tol
                    if not math.isclose(
                        scaled_thrust,
                        ref_thrust * scale_factor,
                        abs_tol=1e-1,
                        rel_tol=1e-4,
                    ):
                        # user wants scaling but provided conflicting inputs,
                        # cannot be resolved
                        raise AttributeError(
                            f'EngineDeck <{self.name}>: Conflicting values provided '
                            'for aircraft:engine:scale_factor and '
                            'aircraft:engine:scaled_sls_thrust when compared against '
                            'aircraft:engine:reference_sls_thrust'
                        )
                    # get thrust target & scale factor matching exactly. Scale factor is
                    # design variable, so don't touch it!! Instead change output thrust
                    else:
                        target_thrust = ref_thrust * scale_factor
                        if self.get_val(Settings.VERBOSITY) >= Verbosity.VERBOSE:
                            warnings.warn(
                                f'EngineDeck <{self.name}>: '
                                'aircraft:engine:scaled_sls_thrust and '
                                'product of aircraft:engine_scale_factor and '
                                'aircraft:engine:reference_sls_thrust are not an exact '
                                'match but within tolerance. Setting scaled thrust '
                                f'target to calculated value of {target_thrust} lbf.'
                            )
                        self.set_val(Aircraft.Engine.SCALED_SLS_THRUST, target_thrust, 'lbf')
                else:
                    # engine is not scaled: make sure scaled thrust = ref thrust and
                    # scale factor = 1
                    self.set_val(Aircraft.Engine.SCALED_SLS_THRUST, ref_thrust, 'lbf')
                    self.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

            # scale factor provided, but not target thrust
            else:
                # calculate new scaled thrust value
                scaled_thrust = ref_thrust * scale_factor
                self.set_val(Aircraft.Engine.SCALED_SLS_THRUST, scaled_thrust, 'lbf')

        # target thrust provided, but not scale factor
        elif thrust_provided:
            scaled_thrust = self.get_val(Aircraft.Engine.SCALED_SLS_THRUST, 'lbf')
            if scale_performance:
                scale_factor = scaled_thrust / ref_thrust
                if self.get_val(Settings.VERBOSITY) >= Verbosity.VERBOSE:
                    warnings.warn(
                        f'EngineDeck <{self.name}>: aircraft:engine:scale_factor has '
                        'been indirectly set by the ratio of '
                        'aircraft:engine:scaled_sls_thrust and '
                        f'aircraft:engine:reference_sls_thrust to {scale_factor}'
                    )
                self.set_val(Aircraft.Engine.SCALE_FACTOR, scale_factor)
            else:
                # engine is not scaled: just make sure scale_factor = 1
                self.set_val(Aircraft.Engine.SCALE_FACTOR, 1.0)

        # neither scale factor nor target thrust are provided
        elif not thrust_provided:
            if scale_performance:
                # user wants to scale, but provided no scaling info: default to
                # scale factor of 1, set scaled thrust = ref thrust
                scale_factor = 1
                self.set_val(Aircraft.Engine.SCALE_FACTOR, scale_factor)
                self.set_val(Aircraft.Engine.SCALED_SLS_THRUST, ref_thrust, 'lbf')
            else:
                # engine is not scaled: just make sure scaled thrust = ref thrust
                scaled_thrust = ref_thrust
                self.set_val(Aircraft.Engine.SCALED_SLS_THRUST, scaled_thrust, 'lbf')

    def _normalize_throttle(self):
        """
        Normalize throttle and hybrid throttle options. Requires packed data.

        Throttle is normalized to [0, 1], while hybrid throttle is normalized to two
        separate scales. Negative hybrid throttles are normalized to [-1, 0) and positive
        hybrid throttles to (0, 1], with the zero point representing an assumed "idle"
        condition based on the provided data.

        Normalization can be "global" (using max and min values from entire data set), or
        "local" (using the max and min values from each individual flight condition).
        """

        def _hybrid_throttle_norm(hybrid_throttle_list):
            """
            Normalizes hybrid throttle.

            Uses the scale:
            [-1 (minimum negative hybrid throttle)
            <-> 0 (idle hybrid throttle)
            <-> 1 (max positive hybrid throttle)]

            Negative normalized hybrid throttles only appear if negative hybrid throttle
            values are provided in engine data. Positive normalized hybrid throttle values
            only appear if positive hybrid throttle values are provided in engine data.

            Parameters
            ----------
            hybrid_throttle_list : (list, numpy.ndarray)
                Hybrid throttle data to be normalized.

            Returns
            -------
            norm_hybrid_list : numpy.ndarray
                Normalized hybrid throttle data from hybrid_throttle_list.
            """
            norm_hybrid_list = np.array(hybrid_throttle_list)
            # Split throttle into positive and negative components
            # (track index to preserve order)
            # Throttle points at zero do not need to be tracked - they are already
            # "normalized", and zero is always assumed to be in the normalization range
            # (max or min)
            hybrid_throttle_neg_idx = np.where(norm_hybrid_list < 0)
            if not hybrid_throttle_neg_idx[0].size == 0:
                hybrid_throttle_neg = norm_hybrid_list[hybrid_throttle_neg_idx]

                # normalize negative component from -1 to 0
                hybrid_throttle_neg_norm = normalize(hybrid_throttle_neg, maximum=0) - 1
                norm_hybrid_list[hybrid_throttle_neg_idx] = hybrid_throttle_neg_norm

            hybrid_throttle_pos_idx = np.where(norm_hybrid_list > 0)
            if not hybrid_throttle_pos_idx[0].size == 0:
                hybrid_throttle_pos = norm_hybrid_list[hybrid_throttle_pos_idx]

                # normalize positive component from 0 to 1
                hybrid_throttle_pos_norm = normalize(hybrid_throttle_pos, minimum=0)

                norm_hybrid_list[hybrid_throttle_pos_idx] = hybrid_throttle_pos_norm

            return norm_hybrid_list

        # vectors of the normalized throttle data
        normalized_throttle = np.array([])
        normalized_hybrid_throttle = np.array([])

        # Min and max throttle value vectors
        # If using "global" throttle setting, max & min are single values and these are unused,
        # otherwise they contain min or max throttle per unique flight condition
        throttle_min = np.array([])
        throttle_max = np.array([])
        hybrid_throttle_min = np.array([])
        hybrid_throttle_max = np.array([])

        # information on packed data
        packed_throttle = self.packed_data[THROTTLE]
        packed_hybrid_throttle = self.packed_data[HYBRID_THROTTLE]
        data_indices = self.data_indices

        # for each unique flight condition...
        for M in range(self.mach_max_count):
            for A in range(self.alt_max_count):
                if data_indices[M, A] == 0:
                    # skip point if there is no data
                    continue

                if not self.global_throttle:
                    # normalize throttles for this flight condition from 0 to 1
                    throttle_list = normalize(packed_throttle[M, A][: data_indices[M, A] + 1])
                    normalized_throttle = np.append(normalized_throttle, throttle_list)
                    throttle_min = np.append(throttle_min, min(throttle_list))
                    throttle_max = np.append(throttle_max, max(throttle_list))

                if not self.global_hybrid_throttle and self.use_hybrid_throttle:
                    # normalize hybrid throttles for this flight condition
                    hybrid_throttle_list = _hybrid_throttle_norm(
                        packed_hybrid_throttle[M, A][: data_indices[M, A] + 1]
                    )
                    normalized_hybrid_throttle = np.append(
                        normalized_hybrid_throttle, hybrid_throttle_list
                    )
                    hybrid_throttle_min = np.append(hybrid_throttle_min, min(hybrid_throttle_list))
                    hybrid_throttle_max = np.append(hybrid_throttle_max, max(hybrid_throttle_list))

        # store normalized throttle data
        if self.global_throttle:
            self.data[THROTTLE] = normalize(self.data[THROTTLE])
            self.throttle_min = min(self.data[THROTTLE])
            self.throttle_max = max(self.data[THROTTLE])
        else:
            self.data[THROTTLE] = normalized_throttle
            self.throttle_min = throttle_min
            self.throttle_max = throttle_max

        # store normalized hybrid throttle data
        if self.use_hybrid_throttle:
            if self.global_hybrid_throttle:
                norm_hybrid_throttle = _hybrid_throttle_norm(self.data[HYBRID_THROTTLE])

                self.hybrid_throttle_min = min(self.data[HYBRID_THROTTLE])
                self.hybrid_throttle_max = max(self.data[HYBRID_THROTTLE])
                self.data[HYBRID_THROTTLE] = norm_hybrid_throttle
            else:
                self.data[HYBRID_THROTTLE] = normalized_hybrid_throttle
                self.hybrid_throttle_min = hybrid_throttle_min
                self.hybrid_throttle_max = hybrid_throttle_max

        # repack data to keep it up to date
        self._pack_data()

    def _sort_data(self):
        """
        Sort unpacked engine data in order based on Aircraft.Engine.INTERPOLATION_SORT. When this
        is set to "mach", sort by Mach number, altitude, throttle. When it is set to "altitude",
        sort by altitude, Mach number, throttle.
        hybrid throttle.
        """
        interp_sort = self.get_val(Aircraft.Engine.INTERPOLATION_SORT)
        engine_data = self.data

        # sort engine data to ensure independent variables are always in
        # ascending order as required by metamodel interpolator

        # convert engine_data from dict to list so it can be sorted
        sorted_values = np.array([engine_data[key] for key in engine_data]).transpose()

        if interp_sort == 'altitude':
            # Sort by mach, then altitude, then throttle, then hybrid throttle
            sort_keys = [
                engine_data[HYBRID_THROTTLE],
                engine_data[THROTTLE],
                engine_data[MACH],
                engine_data[ALTITUDE],
            ]

        else:
            # Sort by mach, then altitude, then throttle, then hybrid throttle
            sort_keys = [
                engine_data[HYBRID_THROTTLE],
                engine_data[THROTTLE],
                engine_data[ALTITUDE],
                engine_data[MACH],
            ]

        sorted_values = sorted_values[np.lexsort(sort_keys)]

        for idx, var in enumerate(engine_data):
            engine_data[var] = sorted_values[:, idx]

        self.data = engine_data

    def _pack_data(self):
        """
        Reorganize data from a dictionary of flat 2d arrays to a dictionary of 3d arrays
        organized by Mach, altitude, and data for each engine variable.
        Data is an array with a length equal to the number of unique data points at
        that Mach, alt point.
        """
        # method requires sorted data
        self._sort_data()
        # get updated data count
        self._count_data()

        mach_max_count = self.mach_max_count
        alt_max_count = self.alt_max_count
        data_max_count = self.data_max_count
        data_indices = self.data_indices

        packed_data = self.packed_data = {}
        idx = 0

        for key in self.data:
            packed_data[key] = np.zeros((mach_max_count, alt_max_count, data_max_count))

        for M in range(mach_max_count):
            for A in range(alt_max_count):
                if data_indices[M, A] == 0:
                    # skip point if there is no data
                    continue

                # number of data points is index+1
                for D in range(data_indices[M, A] + 1):
                    for key in self.data:
                        unpacked_data = self.data[key]
                        if idx < len(unpacked_data):
                            packed_data[key][M, A, D] = unpacked_data[idx]
                    idx += 1

    def _count_data(self):
        """
        Count unique data entries in the engine data for each Mach, altitude combination.
        Requires that data is sorted.

        Raises
        ------
        UserWarning
            If insufficient number of altitude points (<2) provided for a given Mach
            number.
        """
        interp_sort = self.get_val(Aircraft.Engine.INTERPOLATION_SORT)

        mach_count = 0
        # First Mach number must have at least one altitude associated with it
        alt_count = 1
        max_alt_count = 0
        # First Mach number must have at least one data point associated with it
        data_count = 1
        max_data_count = 0

        # data_indices stores how many data points there are for a given Mach/alt combo
        data_indices = np.array([[]])

        curr_mach = curr_alt = np.inf

        mach_numbers = self.data[MACH]
        altitudes = self.data[ALTITUDE]

        # Loop through data. Keep track of last unique value (curr_*) to compare each new
        #   value with
        # Count number of altitudes per mach, number of data points per
        #   mach/altitude combination, compare with max_count
        for idx in range(self.model_length):
            mach_num = mach_numbers[idx]
            alt = altitudes[idx]

            if math.isclose(mach_num, curr_mach, abs_tol=self.mach_tol):
                if math.isclose(alt, curr_alt, abs_tol=self.alt_tol):
                    data_indices[mach_count - 1, alt_count - 1] = data_count
                    data_count += 1

                else:
                    # new altitude for this Mach number, count it

                    # if there are less than two Machs for this altitude, quit
                    if interp_sort == 'altitude':
                        if mach_count < 2 and alt_count > 0:
                            raise UserWarning(
                                'Only one Mach provided for altitude '
                                f'{altitudes[alt_count]:6.3f} in engine data '
                                'file '
                                f'<{self.get_val(Aircraft.Engine.DATA_FILE).name}>'
                            )

                    curr_alt = alt
                    alt_count += 1
                    data_indices = extend_array(data_indices, [mach_count, alt_count])

                    if data_count > max_data_count:
                        max_data_count = data_count
                    # new altitude means reset data counter
                    data_count = 1
                    # count data associated with new altitude
                    data_indices[mach_count - 1, alt_count - 1] = 1

            else:
                # new Mach number

                # if there are less than two altitudes for this Mach number, quit
                if interp_sort == 'mach':
                    if alt_count < 2 and mach_count > 0:
                        raise UserWarning(
                            'Only one altitude provided for Mach number '
                            f'{mach_numbers[mach_count]:6.3f} in engine data '
                            'file '
                            f'<{self.get_val(Aircraft.Engine.DATA_FILE).name}>'
                        )

                # record and count Mach numbers
                curr_mach = mach_num
                mach_count += 1

                # new mach comes with new altitude, record and count it
                if alt_count > max_alt_count:
                    max_alt_count = alt_count
                # new mach means reset altitude counter
                curr_alt = alt
                alt_count = 1
                data_indices = extend_array(data_indices, [mach_count, alt_count])

                if data_count > max_data_count:
                    max_data_count = data_count
                # new mach means reset data counter
                data_count = 1
                # count data associated with new altitude
                data_indices[mach_count - 1, alt_count - 1] = 1

        self.mach_max_count = mach_count
        self.alt_max_count = max_alt_count
        self.data_max_count = max_data_count
        self.data_indices = data_indices.astype(int)


#####################
# UTILITY FUNCTIONS #
#####################
"""
Functions that do not directly use attributes of EngineDeck (do not require self) are
located here. These functions are currently only used for EngineDecks and are not
applicable to other EngineModels. If any of these functions become useful to other
EngineModels besides EngineDeck, move them to propulsion utils.
"""


def normalize(base_list, maximum=None, minimum=None):
    """
    Normalize the given list from 0 to 1.
    Maximum or minimum of data range can be overwritten, otherwise range of list
    is assumed to contain full range of data that must be normalized.

    Parameters
    ----------
    base_list : (list, numpy.ndarray)
        Data that is to be normalized.
    maximum : float
        Overwritten maximum value of data that will scale to 1 when normalized.
    minimum : float
        Overwritten minimum value of data that will scale to 0 when normalized.

    Returns
    -------
    norm_list : numpy.ndarray
        Normalized data from base_list.
    """
    if maximum is None:
        maximum = max(base_list)
    if minimum is None:
        minimum = min(base_list)

    norm_list = np.array([(x - minimum) / (maximum - minimum) for x in base_list])

    return norm_list


def extend_array(inp_array, size):
    """
    Extends input array such that it is at least as large as the target size in each dimension.
    Works on arrays of any dimension.
    Returns copy of input array extended in required dimensions with newly created points set to 0.
    If requested size is smaller than inp_array in any dimensions, original inp_array is not
    modified along that dimension.

    Parameters
    ----------
    inp_array : (list, numpy.ndarray)
        Array that needs to be checked for expansion.
    size : list
        List containing desired minimum length of inp_array along each dimension.

    Returns
    -------
    inp_array : numpy.ndarray
        The provided array extended in the desired dimensions with newly created points set to 0.
    """
    # Convert input to numpy array if it isn't already
    inp_array = np.asarray(inp_array)

    # Calculate padding needed for each dimension
    current_shape = np.array(inp_array.shape)
    target_shape = np.array(size)

    # If target has more dimensions, add new dimensions to input array
    if len(target_shape) > len(current_shape):
        # Create new shape with additional dimensions of size 1
        new_shape = np.ones(len(target_shape), dtype=int)
        new_shape[: len(current_shape)] = current_shape
        # Reshape input array to add new dimensions
        inp_array = inp_array.reshape(new_shape)
        current_shape = np.array(inp_array.shape)

    padding = np.maximum(0, target_shape - current_shape)

    # Create padding tuple for np.pad
    pad_width = tuple((0, p) for p in padding)

    # Pad array with zeros
    return np.pad(inp_array, pad_width, mode='constant', constant_values=0)
