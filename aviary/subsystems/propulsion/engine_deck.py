"""
Define utilities for building engine decks.

Classes
-------
EngineDeck : the interface for an engine deck builder.

Attributes
----------
accepted_headers : dict
    The strings that are accepted as valid header names after converted to all lowercase
    with all whitespace removed, mapped to the enum EngineModelVariables.

required_variables : set
    Variables that must be present in an EngineDeck's DATA_FILE (Mach, altitude, etc.)

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

from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.subsystems.propulsion.engine_scaling import EngineScaling
from aviary.subsystems.propulsion.engine_sizing import SizeEngine
from aviary.subsystems.propulsion.utils import (EngineModelVariables,
                                                convert_geopotential_altitude,
                                                default_units)
from aviary.utils.named_values import NamedValues, get_keys, get_items
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.utils.csv_data_file import read_data_file


MACH = EngineModelVariables.MACH
ALTITUDE = EngineModelVariables.ALTITUDE
THROTTLE = EngineModelVariables.THROTTLE
HYBRID_THROTTLE = EngineModelVariables.HYBRID_THROTTLE
THRUST = EngineModelVariables.THRUST
GROSS_THRUST = EngineModelVariables.GROSS_THRUST
RAM_DRAG = EngineModelVariables.RAM_DRAG
FUEL_FLOW = EngineModelVariables.FUEL_FLOW
ELECTRIC_POWER = EngineModelVariables.ELECTRIC_POWER
NOX_RATE = EngineModelVariables.NOX_RATE
TEMPERATURE = EngineModelVariables.TEMPERATURE_ENGINE_T4
# EXIT_AREA = EngineModelVariables.EXIT_AREA

# EngineDeck assumes all aliases point to an enum, these are used internally only
aliases = {
    # whitespaces are replaced with underscores converted to lowercase before
    # comparison with keys
    MACH: ['m', 'mn', 'mach', 'mach_number'],
    ALTITUDE: ['altitude', 'alt', 'h'],
    THROTTLE: ['throttle', 'power_code', 'pc'],
    HYBRID_THROTTLE: ['hybrid_throttle', 'hpc', 'hybrid_power_code', 'electric_throttle'],
    THRUST: ['thrust', 'net_thrust'],
    GROSS_THRUST: ['gross_thrust'],
    RAM_DRAG: ['ram_drag'],
    FUEL_FLOW: ['fuel', 'fuel_flow', 'fuel_flow_rate'],
    ELECTRIC_POWER: 'electric_power',
    NOX_RATE: ['nox', 'nox_rate'],
    TEMPERATURE: ['t4', 'temp', 'temperature']
}

# these variables must be present in engine performance data
required_variables = {
    MACH,
    ALTITUDE,
    THROTTLE,
    THRUST
}

# EngineDecks internally require these options to have values. Input checks will set
# these options to default values in self.options if they are not provided
required_options = (
    Aircraft.Engine.SCALE_PERFORMANCE,
    Aircraft.Engine.IGNORE_NEGATIVE_THRUST,
    Aircraft.Engine.GEOPOTENTIAL_ALT,
    Aircraft.Engine.GENERATE_FLIGHT_IDLE,
    # TODO fuel flow scaler is required for the EngineScaling component but does not need
    #      to be defined on a per-engine basis, so it could exist only in the problem-
    #      level aviary_options without issue. Is this a propulsion_preprocessor task?
    Mission.Summary.FUEL_FLOW_SCALER
)

# options that are only required based on the value of another option
dependent_options = {
    Aircraft.Engine.GENERATE_FLIGHT_IDLE: (Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION,
                                           Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION,
                                           Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION,)
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
    data : NamedVaues (<empty>)
        Engine performance data (optional). If provided, used instead of tabular data 
        file.

    Methods
    -------
    build_pre_mission
    build_mission
    get_val
    set_val
    update
    """

    def __init__(self, name='engine_deck', options=None, data: NamedValues = None):
        if data is not None:
            self.read_from_file = False
        else:
            self.read_from_file = True
            # TODO update default name to be based on filename

        # also calls _preprocess_inputs() as part of EngineModel __init__
        super().__init__(name, options)

        # copy of raw data read from data_file or memory, never modified or used outside
        #     EngineDeck
        self._original_data = {key: np.array([]) for key in EngineModelVariables}
        # working copy of engine performance data, is modified during data pre-processing
        self.data = {key: np.array([]) for key in EngineModelVariables}
        # gross thrust and ram drag are not used outside of EngineDeck, remove from
        #     working data
        self.data.pop(GROSS_THRUST)
        self.data.pop(RAM_DRAG)

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

        # TODO make this an option - disabling global throttle ranges is better to \
        #      prevent unintended extrapolation, but breaks missions using gasp-based
        #      engines that have uneven throttle ranges (need t4 constraint on mission
        #      to truly fix)
        self.global_throttle = True
        self.global_hybrid_throttle = True

        self._set_variable_flags()

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
                warnings.warn(
                    f'<{key}> is a required option for EngineDecks, but has not been '
                    f'specified for EngineDeck <{self.name}>. The default value will be '
                    'used.')

                val = _MetaData[key]['default_value']
                units = _MetaData[key]['units']
                self.set_val(key, val, units)
        # check dependent options
        for key in dependent_options:
            if self.get_val(key):
                for item in dependent_options[key]:
                    if item not in options:
                        val = _MetaData[item]['default_value']
                        units = _MetaData[item]['units']
                        self.set_val(item, val, units)

        # LOGIC CHECKS
        if self.get_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE):
            idle_min = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION)
            idle_max = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION)
            # Allowing idle fractions to be equal, i.e. fixing flight idle conditions
            # instead of extrapolation
            if idle_min > idle_max:
                warnings.warn(
                    f'EngineDeck <{self.name}>: Minimum flight idle fraction exceeds maximum '
                    f'flight idle fraction. Values for min and max fraction will be flipped.'
                )
                self.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION,
                             val=idle_max)
                self.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION,
                             val=idle_min)

        # check that sufficient information on engine scaling is provided
        # default behavior is to calculate scale factor based on thrust target
        engine_mapping = get_keys(self.options)

        # check if scale factor and thrust target are user defined and check consistency
        scale_performance = self.get_val(Aircraft.Engine.SCALE_PERFORMANCE)
        scale_factor_provided = False
        thrust_provided = False
        # was scale factor originally provided? (Not defaulted)
        if Aircraft.Engine.SCALE_FACTOR in engine_mapping:
            scale_factor_provided = True
        # was scaled thrust originally provided? (Not defaulted)
        if Aircraft.Engine.SCALED_SLS_THRUST in engine_mapping:
            thrust_provided = True

        # user provided target thrust or scale factor, but performance scaling is off
        if scale_performance and (scale_factor_provided or thrust_provided):
            UserWarning(
                f'EngineDeck <{self.name}>: Scaling targets are provided, but will be '
                'ignored because performance scaling is disabled. Set '
                'aircraft:engine:SCALE_PERFORMANCE to True to enable scaling.'
            )

    def _set_variable_flags(self):
        """
        Sets flags in EngineDeck to communicate which (non-required) variables are 
        avaliable to greater propulsion module.
        """
        engine_variables = self.engine_variables

        # TODO many of these may not be required, just independents?
        self.use_thrust = THRUST in engine_variables
        self.use_fuel = FUEL_FLOW in engine_variables
        self.use_electricity = ELECTRIC_POWER in engine_variables
        self.use_hybrid_throttle = HYBRID_THROTTLE in engine_variables
        self.use_nox = NOX_RATE in engine_variables
        self.use_t4 = TEMPERATURE in engine_variables
        # self.use_exit_area = EXIT_AREA in engine_variables

    def _setup(self, data):
        """
        Read in and process engine data:

            Check data consistency.

            Convert altitudes to geometric.

            Sort and pack data.

            Determine reference thrust.

            Normalize throttles/hybrid throttles.

            Fill flight idle points.
        """
        self._read_data(data)

        # perform consistency checks on data
        self._check_data()

        # convert geopotential altitude to geometric if required
        if self.get_val(Aircraft.Engine.GEOPOTENTIAL_ALT):
            self.data[ALTITUDE] = convert_geopotential_altitude(
                self.data[ALTITUDE])

        # sort and organize data
        self._pack_data()

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
        # custom error messages depending on data type
        if self.read_from_file:
            message = f'<{self.get_val(Aircraft.Engine.DATA_FILE)}>'
        else:
            message = f'EngineDeck <{self.name}>'

        # get data (as NamedValues object) from data file
        if self.read_from_file:
            data_file = self.get_val(Aircraft.Engine.DATA_FILE)

            # read csv file - currently not saving comments
            raw_data = read_data_file(data_file, aliases=aliases)

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
                    val = np.array([convert_units(i, units, default_units[key])
                                   for i in val])
                except TypeError:
                    raise TypeError(f"{message}: units of '{units}' provided for "
                                    f'<{key.name}> are not compatible with expected units '
                                    f'of {default_units[key]}')

                # Engine_variables currently only used to store "valid" engine variables
                # as defined in EngineModelVariables Enum
                self.engine_variables[key] = units

            else:
                warnings.warn(
                    f'{message}: header <{key}> was not recognized, and will be skipped')

            # save all data in self._original_data, including skipped variables
            self._original_data[key] = val

        if not self.engine_variables:
            raise UserWarning(f'No valid engine variables found in data for {message}')

        # set flags using updated engine_variables
        self._set_variable_flags()

        # Copy data from original data (never modified) to working data (changed through
        #    sorting, generating missing data, etc.)
        # self.data contains all keys in EngineModelVariables except for ram drag and
        #    gross thrust
        for key in self.data:
            self.data[key] = self._original_data[key]

    def _check_data(self):
        """
        Checks for consistency of provided thrust and drag data, ensures no required
        variables are missing, fills unused variabes with a default value of zero, and
        removes negative thrusts if requested.

        Raises
        ------
        UserWarning
            If provided net thrust does not match difference between provided gross
            thrust and ram drag within tolerance.
        """
        # custom error messages depending on data type
        if self.read_from_file:
            message = f'<{self.get_val(Aircraft.Engine.DATA_FILE)}>'
        else:
            message = f'EngineDeck <{self.name}>'

        engine_variables = self.engine_variables

        # Handle ram drag, net and gross thrust and potential conflicts in value or units
        # Warn user if they provide partial info for calulated thrust
        # Not a fail state if net thrust is still provided
        # If both net thrust and components for calculated thrust both provided, a sanity
        #   check that they match is done after reading data
        if THRUST in engine_variables:
            # if thrust is present, but gross thrust or ram drag also present raise warning
            if GROSS_THRUST in engine_variables and not RAM_DRAG in engine_variables:
                warnings.warn(f'{message} contains both net and '
                              'gross thrust. Only net thrust will be used.')
            if not GROSS_THRUST in engine_variables and RAM_DRAG in engine_variables:
                warnings.warn(f'{message} contains both net thrust '
                              'and ram drag. Only net thrust will be used.')

        if RAM_DRAG in engine_variables and GROSS_THRUST in engine_variables:
            # Check that units are the same. Variables have already been checked for valid
            # units, so it is assumed they are convertable. Prioritizes thrust units
            if engine_variables[RAM_DRAG] != engine_variables[GROSS_THRUST]:
                self.data[RAM_DRAG] = convert_units(self.data,
                                                    engine_variables[RAM_DRAG],
                                                    engine_variables[GROSS_THRUST])
                engine_variables[RAM_DRAG] = engine_variables[GROSS_THRUST]

            net_thrust_calc = self._original_data[GROSS_THRUST] \
                - self._original_data[RAM_DRAG]
            # prefer using directly provided values for net thrust vs. calculating
            if THRUST in engine_variables:
                res = abs(net_thrust_calc - self._original_data[THRUST])
                if np.any(self.thrust_tol > res):
                    raise UserWarning('Provided net thrust is not equal to difference '
                                      '(within tolerance) between gross thrust and ram '
                                      f'drag in {message}')
            else:
                # store net thrust in THRUST key instead of gross thrust
                self.data[THRUST] = net_thrust_calc
                engine_variables[THRUST] = engine_variables[GROSS_THRUST]

        # remove unneeded dependent variables from engine_variables
        if RAM_DRAG in engine_variables:
            engine_variables.pop(RAM_DRAG)
        if GROSS_THRUST in engine_variables:
            engine_variables.pop(GROSS_THRUST)

        self._set_variable_flags()

        self.model_length = len(self.data[ALTITUDE])

        # check that all required variables are present in engine data
        if not required_variables.issubset(engine_variables):
            # gather all missing required variables
            missing_variables = set()
            for var in engine_variables:
                if var in required_variables:
                    missing_variables.add(var)

            # if missing_variables is not empty
            if not missing_variables:
                raise UserWarning(f'Required variables {missing_variables} are '
                                  f'missing from {message}'
                                  )

        # Set all unused variables to default value of zero
        model = self.data
        for key in model:
            if not len(model[key]):
                model[key] = np.zeros(self.model_length)

        # removes data points with negative thrust if requested
        if self.get_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST):
            keep_idx = np.where(model[THRUST] >= 0)
            for key in model:
                model[key] = model[key][keep_idx]

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

            rvalue = (
                y0 + (y1 - y0) * thrust_extrap_term
            )

            return rvalue

        idle_thrust_fract = self.get_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION)
        idle_min_fract = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION)
        idle_max_fract = self.get_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION)

        packed_data = self.packed_data

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
            # how far apart the "fake" points should be from the actual idle point
            # this time, we want an arbitrarily small number
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
                    idle_points[MACH], [packed_data[MACH][M, A, 0]] * num_points)
                idle_points[ALTITUDE] = np.append(
                    idle_points[ALTITUDE], [packed_data[ALTITUDE][M, A, 0]] * num_points)
                idle_points[THROTTLE] = np.append(
                    idle_points[THROTTLE], [throttle_idle] * num_points)
                if self.use_hybrid_throttle:
                    hybrid_throttle_range = np.linspace(hybrid_throttle_idle-h_tol,
                                                        hybrid_throttle_idle+h_tol,
                                                        num_points)
                    idle_points[HYBRID_THROTTLE] = np.append(
                        idle_points[HYBRID_THROTTLE], hybrid_throttle_range)
                else:
                    idle_points[HYBRID_THROTTLE] = np.append(
                        idle_points[HYBRID_THROTTLE], hybrid_throttle_idle)

                # if there is only one data point at this Mach, alt combination, use
                # thrust fraction instead of extrapolation
                # TODO idle currently calculated using lowest index data points - this is not
                #      guaranteed to be at hybrid throttle idle point, could be negative
                if data_indices[M, A] == 1:
                    for key in packed_data:
                        if key not in [
                                MACH,
                                ALTITUDE,
                                THROTTLE,
                                HYBRID_THROTTLE,
                                THRUST]:
                            idle_value = packed_data[key][M, A, 0] * idle_thrust_fract
                            var_min = packed_data[key][M, A, -1] * idle_min_fract
                            var_max = packed_data[key][M, A, -1] * idle_max_fract

                            if idle_value < var_min:
                                idle_value = var_min
                            elif idle_value > var_max:
                                idle_value = var_max

                            idle_points[key] = np.append(idle_points[key],
                                                         [idle_value] * num_points)
                            # add Mach, alt combination to idle_points with idle power
                            # codes

                    # thrust does not get idle_min/max checks
                    idle_points[THRUST] = np.append(idle_points[THRUST],
                                                    [[packed_data[THRUST][M, A, 0]
                                                     * idle_thrust_fract]] * num_points)
                    # move to next data point
                    continue

                # calculate idle thrust as a percentage of max thrust at Mach, alt point
                idle_thrust = packed_data[THRUST][M, A, data_indices[M, A] - 1]\
                    * idle_thrust_fract

                # add this thrust point to idle_points
                idle_points[THRUST] = np.append(idle_points[THRUST],
                                                [idle_thrust] * num_points)

                # calculate thrust term for linear extrapolation
                thrust_extrap_term = (idle_thrust - packed_data[THRUST][M, A, 0]) / (
                    packed_data[THRUST][M, A, 1] - packed_data[THRUST][M, A, 0])

                # compute idle data
                for key in packed_data:
                    # skip independent variables or thrust, which is already calculated
                    if key not in [
                            MACH,
                            ALTITUDE,
                            THROTTLE,
                            HYBRID_THROTTLE,
                            THRUST]:
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
                        idle_points[key] = np.append(idle_points[key],
                                                     [idle_value] * num_points)

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

    def build_pre_mission(self, aviary_inputs):
        """
        Build components to be added to pre-mission propulsion subsystem.

        Returns
        -------
            SizeEngine component specific to this EngineDeck, used for calculating engine
            scaling factors.
        """

        return SizeEngine(aviary_options=self.options)

    def build_mission(self, num_nodes, aviary_inputs):
        """
        Creates interpolator objects to be added to mission-level propulsion subsystem.
        Interpolators must be re-generated for each ODE due to potentialy different
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

        engine_group = om.Group()

        # interpolator object for engine data
        engine = om.MetaModelSemiStructuredComp(
            method=interp_method, extrapolate=True, vec_size=num_nodes)

        units = default_units
        for key in self.engine_variables:
            units[key] = self.engine_variables[key]

        # add inputs and outputs to interpolator
        engine.add_input(Dynamic.Mission.MACH,
                         self.data[MACH],
                         units='unitless',
                         desc='Current flight Mach number')
        engine.add_input(Dynamic.Mission.ALTITUDE,
                         self.data[ALTITUDE],
                         units=units[ALTITUDE],
                         desc='Current flight altitude')
        engine.add_input(Dynamic.Mission.THROTTLE,
                         self.data[THROTTLE],
                         units='unitless',
                         desc='Current engine throttle')
        if self.use_hybrid_throttle:
            engine.add_input(Dynamic.Mission.HYBRID_THROTTLE,
                             self.data[HYBRID_THROTTLE],
                             units='unitless',
                             desc='Current engine hybrid throttle')
        engine.add_output('thrust_net_unscaled',
                          self.data[THRUST],
                          units=units[THRUST],
                          desc='Current net thrust produced (unscaled)')
        engine.add_output('fuel_flow_rate_unscaled',
                          self.data[FUEL_FLOW],
                          units=units[FUEL_FLOW],
                          desc='Current fuel flow rate (unscaled)')
        engine.add_output('electric_power_unscaled',
                          self.data[ELECTRIC_POWER],
                          units=units[ELECTRIC_POWER],
                          desc='Current electric energy rate (unscaled)')
        engine.add_output('nox_rate_unscaled',
                          self.data[NOX_RATE],
                          units=units[NOX_RATE],
                          desc='Current NOx emission rate (unscaled)')
        # if self.use_exit_area:
        # engine.add_output('exit_area_unscaled',
        #                   self.data[EXIT_AREA],
        #                   units='ft**2',
        #                   desc='Current exit area (unscaled)')
        engine.add_output(Dynamic.Mission.TEMPERATURE_ENGINE_T4,
                          self.data[TEMPERATURE],
                          units=units[TEMPERATURE],
                          desc='Current turbine exit temperature')

        # Create copy of interpolation component that computes max thrust for current
        # flight condition
        # NOTE max thrust is assumed to occur at maximum throttle and hybrid throttle
        #      for each flight condition
        # TODO Use solver to find throttle/hybrid throttle for maximum thrust at given flight condition?
        #      Pre-solve max throttle/hybrid throttle for each flight condition, interpolate on
        #      reduced data set?
        if self.use_thrust:
            if self.global_throttle or (self.global_hybrid_throttle
                                        and self.use_hybrid_throttle):
                # create IndepVarComp to pass maximum throttle is to max thrust interpolator
                fixed_throttles = om.IndepVarComp()
                if self.global_throttle:
                    fixed_throttles.add_output('throttle_max',
                                               val=np.ones(num_nodes) *
                                               self.throttle_max,
                                               units='unitless',
                                               desc='Engine maximum throttle')
                if self.global_hybrid_throttle and self.use_hybrid_throttle:
                    fixed_throttles.add_output('hybrid_throttle_max',
                                               val=np.ones(num_nodes) *
                                               self.hybrid_throttle_max,
                                               units='unitless',
                                               desc='Engine maximum hybrid throttle')
            if not (self.global_throttle or (self.global_hybrid_throttle
                                             and self.use_hybrid_throttle)):
                interp_throttles = om.MetaModelSemiStructuredComp(method=interp_method,
                                                                  extrapolate=False,
                                                                  vec_size=num_nodes)

                packed_data = self.packed_data
                mach_table = np.array([])
                alt_table = np.array([])

                for M in range(self.mach_max_count):
                    for A in range(self.alt_max_count):
                        if self.data_indices[M, A] != 0:
                            mach_table = np.append(
                                mach_table, packed_data[MACH][M, A, 0])
                            alt_table = np.append(
                                alt_table, packed_data[ALTITUDE][M, A, 0])

                # add inputs and outputs to interpolator
                interp_throttles.add_input(Dynamic.MACH,
                                           mach_table,
                                           units='unitless',
                                           desc='Current flight Mach number')
                interp_throttles.add_input(Dynamic.ALTITUDE,
                                           alt_table,
                                           units=units[ALTITUDE],
                                           desc='Current flight altitude')
                if not self.global_throttle:
                    interp_throttles.add_output('throttle_max',
                                                self.throttle_max,
                                                units='unitless',
                                                desc='max throttle avaliable at current '
                                                'flight condition')
                if not self.global_hybrid_throttle and self.use_hybrid_throttle:
                    interp_throttles.add_output('hybrid_throttle_max',
                                                self.hybrid_throttle_max,
                                                units='unitless',
                                                desc='max hybrid throttle avaliable at '
                                                     'current flight condition')

            # Calculation of max thrust currently done with a duplicate of the engine
            # model and scaling components
            max_thrust_engine = om.MetaModelSemiStructuredComp(
                method=interp_method, extrapolate=False, vec_size=num_nodes)

            max_thrust_engine.add_input(Dynamic.Mission.MACH,
                                        self.data[MACH],
                                        units='unitless',
                                        desc='Current flight Mach number')
            max_thrust_engine.add_input(Dynamic.Mission.ALTITUDE,
                                        self.data[ALTITUDE],
                                        units=units[ALTITUDE],
                                        desc='Current flight altitude')
            # replace throttle coming from mission with max value based on flight condition
            max_thrust_engine.add_input('throttle_max',
                                        self.data[THROTTLE],
                                        units='unitless',
                                        desc='Current engine throttle')
            if self.use_hybrid_throttle:
                # replace hybrid throttle coming from mission with max value based on
                # flight condition
                max_thrust_engine.add_input('hybrid_throttle_max',
                                            self.data[HYBRID_THROTTLE],
                                            units='unitless',
                                            desc='Current engine hybrid throttle')
            max_thrust_engine.add_output('thrust_net_max_unscaled',
                                         self.data[THRUST],
                                         units=units[THRUST],
                                         desc='Current thrust produced')
        else:
            # If engine does not use thrust, a separate component for max thrust is not
            # necessary.
            # Add unscaled max thrust as output of interpolator, which will have a
            # default value of zero at every flight condition
            engine.add_output('thrust_net_max_unscaled',
                              self.data[THRUST],
                              units=units[THRUST],
                              desc='Current max net thrust produced (unscaled)')

        # add created subsystems to engine_group
        engine_group.add_subsystem('interpolation',
                                   engine,
                                   promotes_inputs=['*'],
                                   promotes_outputs=['*'])
        if self.use_thrust:
            if self.global_throttle or (self.global_hybrid_throttle
                                        and self.use_hybrid_throttle):
                engine_group.add_subsystem('fixed_max_throttles',
                                           fixed_throttles,
                                           promotes_outputs=['*'])

            if not (self.global_throttle or (self.global_hybrid_throttle
                                             and self.use_hybrid_throttle)):
                engine_group.add_subsystem('interp_max_throttles',
                                           interp_throttles,
                                           promotes_inputs=['*'],
                                           promotes_outputs=['*'])

            engine_group.add_subsystem(
                'max_thrust_interpolation',
                max_thrust_engine,
                promotes_inputs=['*'],
                promotes_outputs=['*'])

        engine_group.add_subsystem('engine_scaling',
                                   subsys=EngineScaling(num_nodes=num_nodes,
                                                        aviary_options=self.options),
                                   promotes_inputs=['*'],
                                   promotes_outputs=['*'])

        return engine_group

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

        if Aircraft.Engine.REFERENCE_SLS_THRUST not in engine_mapping:
            alt_tol = self.alt_tol
            mach_tol = self.mach_tol
            # NOTE This fails if there is no data point at SLS (within tolerance)
            sea_level_idx = (np.intersect1d(
                np.where(-alt_tol < self.data[ALTITUDE])[0],
                np.where(self.data[ALTITUDE] <= alt_tol)[0]))
            static_idx = (np.intersect1d(
                np.where(-mach_tol < self.data[MACH]),
                np.where(self.data[MACH] < self.mach_tol)))
            sls_idx = np.intersect1d(sea_level_idx, static_idx)

            if sls_idx.size == 0:
                raise UserWarning('Could not find sea-level static max thrust point for '
                                  f'EngineDeck <{self.name}>. Please review the data file '
                                  f'<{self.get_val(Aircraft.Engine.DATA_FILE)}> or '
                                  'manually specify Aircraft.Engine.REFERENCE_SLS_THRUST '
                                  'in EngineDeck options')

            reference_sls_thrust = max(self.data[THRUST][sls_idx])

            self.set_val(Aircraft.Engine.REFERENCE_SLS_THRUST,
                         reference_sls_thrust, units=self.engine_variables[THRUST])

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
                    if not math.isclose(scaled_thrust/ref_thrust, scale_factor):
                        # user wants scaling but provided conflicting inputs,
                        # cannot be resolved
                        raise AttributeError(
                            f'EngineModel <{self.name}>: Conflicting values provided for '
                            'aircraft:engine:scale_factor and '
                            'aircraft:engine:scaled_sls_thrust'
                        )
                else:
                    # engine is not scaled: just make sure scaled thrust = ref thrust
                    self.set_val(
                        Aircraft.Engine.SCALED_SLS_THRUST, ref_thrust, 'lbf')

            # scale factor provided, but not target thrust
            else:
                # calculate new scaled thrust value
                scaled_thrust = ref_thrust*scale_factor
                self.set_val(Aircraft.Engine.SCALED_SLS_THRUST, scaled_thrust, 'lbf')

        # neither scale factor nor target thrust are provided
        if not scale_factor_provided and not thrust_provided:
            if scale_performance:
                # user wants to scale, but provided no scaling info: default to
                # scale factor of 1, set scaled thrust = ref thrust
                scale_factor = 1
                self.set_val(
                    Aircraft.Engine.SCALE_FACTOR, scale_factor)
                self.set_val(
                    Aircraft.Engine.SCALED_SLS_THRUST, ref_thrust, 'lbf')
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
            Normalize hybrid throttle to the scale:

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

        normalized_throttle = np.array([])
        normalized_hybrid_throttle = np.array([])
        throttle_min = np.array([])
        throttle_max = np.array([])

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
                    throttle_list = normalize(
                        packed_throttle[M, A][:data_indices[M, A]+1])
                    # normalize throttles for this flight condition from 0 to 1
                    normalized_throttle = np.append(normalized_throttle, throttle_list)
                    throttle_min = np.append(throttle_min, min(throttle_list))
                    throttle_max = np.append(throttle_max, max(throttle_list))

                if not self.global_hybrid_throttle and self.use_hybrid_throttle:
                    # normalize hybrid throttles for this flight condition
                    hybrid_throttle_list = _hybrid_throttle_norm(
                        packed_hybrid_throttle[M, A][:data_indices[M, A]+1])
                    normalized_hybrid_throttle = np.append(
                        normalized_hybrid_throttle, hybrid_throttle_list)
                    hybrid_throttle_min = np.append(
                        hybrid_throttle_min, min(hybrid_throttle_list))
                    hybrid_throttle_max = np.append(
                        hybrid_throttle_max, max(hybrid_throttle_list))

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
        Sort unpacked engine data in order of mach number, altitude, throttle,
        hybrid throttle.
        """
        engine_data = self.data

        # sort engine data to ensure independent variables are always in
        # ascending order as required by metamodel interpolator

        # convert engine_data from dict to list so it can be sorted
        sorted_values = np.array([engine_data[key] for key in engine_data]).transpose()

        # Sort by mach, then altitude, then throttle, then hybrid throttle
        sorted_values = sorted_values[np.lexsort(
            [engine_data[HYBRID_THROTTLE],
             engine_data[THROTTLE],
             engine_data[ALTITUDE],
             engine_data[MACH]])]
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
        mach_count = 0
        # First mach number must have at least one altitude associated with it
        alt_count = 1
        max_alt_count = 0
        # First mach number must have at least one data point associated with it
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
                    # new altitude for this mach number, count it
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
                if alt_count < 2 and mach_count > 0:
                    raise UserWarning('Only one altitude provided for Mach number '
                                      f'{mach_numbers[mach_count]:6.3f} in engine data file '
                                      f'<{self.get_val(Aircraft.Engine.DATA_FILE).name}>'
                                      )

                # record and count mach numbers
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
    Extends input array such that it is at least as large as the target size in
    all dimensions. If input array is smaller in any dimension, extends input
    array to match target size in that dimension. Works on arrays up to 3
    dimensions. Returns copy of input array extended in required dimensions with newly
    created points set to 0.

    Parameters
    ----------
    inp_array : (list, numpy.ndarray)
        Array that needs to be checked for expansion.
    size : list
        List containing desired minimum length of inp_array along each dimension.

    Returns
    -------
    inp_array : numpy.ndarray
        The provided array extended in the desired dimensions with with newly created
        points set to 0.
    """
    # TODO may be built-in numpy functions that can replace this and support n dimensions
    dims = np.array(np.shape(inp_array))
    while size[0] > dims[0]:
        inp_array = np.concatenate(
            (inp_array, np.zeros((1, *dims[1:]))), 0)
        dims[0] += 1
    if len(dims) > 1:
        while size[1] > dims[1]:
            inp_array = np.concatenate(
                (inp_array, np.zeros((dims[0], 1, *dims[2:]))), 1)
            dims[1] += 1
        if len(dims) > 2:
            while size[2] > dims[2]:
                inp_array = np.concatenate(
                    (inp_array, np.zeros((*dims[:2], 1))), 2)
                dims[2] += 1

    return inp_array
