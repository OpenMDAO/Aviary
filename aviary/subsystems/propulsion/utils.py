"""
Attributes
----------
default_units : dict
    Matches each EngineModelVariables entry with default units (str)
"""

from enum import Enum
from pathlib import Path

import numpy as np
import openmdao.api as om

import aviary.constants as constants
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.named_values import NamedValues, get_items, get_keys
from aviary.utils.utils import isiterable
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class EngineModelVariables(Enum):
    """Define constants that map to supported variable names in an engine model."""

    MACH = Dynamic.Atmosphere.MACH
    ALTITUDE = Dynamic.Mission.ALTITUDE
    THROTTLE = Dynamic.Vehicle.Propulsion.THROTTLE
    HYBRID_THROTTLE = Dynamic.Vehicle.Propulsion.HYBRID_THROTTLE
    THRUST = Dynamic.Vehicle.Propulsion.THRUST
    TAILPIPE_THRUST = 'tailpipe_thrust'
    GROSS_THRUST = 'gross_thrust'
    SHAFT_POWER = Dynamic.Vehicle.Propulsion.SHAFT_POWER
    SHAFT_POWER_CORRECTED = 'shaft_power_corrected'
    RAM_DRAG = 'ram_drag'
    RPM = Dynamic.Vehicle.Propulsion.RPM
    FUEL_FLOW = Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE
    ELECTRIC_POWER_IN = Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN
    NOX_RATE = Dynamic.Vehicle.Propulsion.NOX_RATE
    TEMPERATURE_T4 = Dynamic.Vehicle.Propulsion.TEMPERATURE_T4
    TORQUE = Dynamic.Vehicle.Propulsion.TORQUE
    # EXIT_AREA = auto()


default_units = {
    EngineModelVariables.MACH: 'unitless',
    EngineModelVariables.ALTITUDE: 'ft',
    EngineModelVariables.THROTTLE: 'unitless',
    EngineModelVariables.HYBRID_THROTTLE: 'unitless',
    EngineModelVariables.THRUST: 'lbf',
    EngineModelVariables.TAILPIPE_THRUST: 'lbf',
    EngineModelVariables.GROSS_THRUST: 'lbf',
    EngineModelVariables.SHAFT_POWER: 'hp',
    EngineModelVariables.SHAFT_POWER_CORRECTED: 'hp',
    EngineModelVariables.RAM_DRAG: 'lbf',
    EngineModelVariables.FUEL_FLOW: 'lb/h',
    EngineModelVariables.ELECTRIC_POWER_IN: 'kW',
    EngineModelVariables.NOX_RATE: 'lb/h',
    EngineModelVariables.TEMPERATURE_T4: 'degR',
    EngineModelVariables.TORQUE: 'ft*lbf',
    EngineModelVariables.RPM: 'rpm',
    # EngineModelVariables.EXIT_AREA: 'ft**2',
}

# variables that have an accompanying max value
max_variables = {
    EngineModelVariables.THRUST: Dynamic.Vehicle.Propulsion.THRUST_MAX,
    EngineModelVariables.SHAFT_POWER: Dynamic.Vehicle.Propulsion.SHAFT_POWER_MAX,
}


def convert_geopotential_altitude(altitude):
    """
    Converts altitudes from geopotential to geometric altitude
    Assumes altitude is provided in feet.

    Parameters
    ----------
    altitude_list : <(float, list of floats)>
        geopotential altitudes (in ft) to be converted.

    Returns
    -------
    altitude_list : <list of floats>
        geometric altitudes (ft).
    """
    try:
        iter(altitude)
    except TypeError:
        altitude = [altitude]

    g = constants.GRAV_METRIC_FLOPS
    radius_earth = constants.RADIUS_EARTH_METRIC
    CM1 = 0.99850  # Center of mass (Earth)? Unknown
    OC2 = 26.76566e-10  # Unknown
    GNS = 9.8236930  # grav_accel_at_surface_earth?

    for i, alt in enumerate(altitude):
        HFT = alt
        HO = HFT * 0.30480  # convert ft to m
        Z = (HFT + (4.37 * (10**-8)) * (HFT**2.00850)) * 0.30480

        DH = float('inf')

        while abs(DH) > 1.0:
            R = radius_earth + Z
            GN = GNS * (radius_earth / R) ** (CM1 + 1.0)
            H = (R * GN * ((R / radius_earth) ** CM1 - 1.0) / CM1 - Z * (R - Z / 2.0) * OC2) / g

            DH = HO - H
            Z += DH

        alt = Z / 0.30480  # convert m to ft
        altitude[i] = alt

    return altitude


# TODO build test for this function
# TODO upgrade to be able to turn vectorized AviaryValues into multiple engine decks
def build_engine_deck(
    options: AviaryValues,
    name: str = None,
    required_variables=None,
    meta_data=_MetaData,
):
    """
    Creates an EngineDeck using available inputs and options in aviary_options.

    Parameter
    ----------
    options : AviaryValues
        Options to use in creation of EngineDeck

    name : str, optional
        The name of the newly created EngineDeck

    required_variables : set, optional
        A set of required variables (from EngineModelVariables) for the newly created
        EngineDeck. Defaults to the required set {ALTITUDE, MACH, THROTTLE, THRUST}.

    meta_data : dict, optional
        The metadata to be used while creating the EngineDeck. If None, use Aviary's
        default metadata.

    Returns
    -------
    EngineDeck
        EngineDeck created using provided options.
    """
    # Required engine vars include one setting from Mission.Summary
    engine_vars = [item for item in Aircraft.Engine.__dict__.values()]
    engine_vars.append(Mission.Summary.FUEL_FLOW_SCALER)

    # Build a single engine deck, currently ignoring vectorization of AviaryValues
    # (use first item in arrays when appropriate)
    engine_options = AviaryValues()
    for var in engine_vars:
        # check if this variable exist with useable metadata
        try:
            units = meta_data[var]['units']
        except (KeyError, TypeError):
            continue
        else:
            try:
                aviary_val = options.get_val(var, units)
            # if not, use default value from _MetaData?
            except KeyError:
                # currently skipping filling "missing" variables with defaults
                # engine_options.set_val(var, meta_data[var]['default_value'], units)
                continue
            # add value from options to engine_options
            else:
                if isiterable(aviary_val):
                    try:
                        aviary_val_0 = aviary_val[0]
                    except TypeError:
                        pass
                    else:
                        # if item in first index is also iterable, or if array only
                        # contains a single value, use that
                        if isiterable(aviary_val_0) or len(aviary_val) == 1:
                            aviary_val = aviary_val_0
                # "Convert" numpy types to standard Python types. Wrap first
                # index in numpy array before calling item() to safeguard against
                # non-standard types, such as objects
                if np.array(aviary_val).ndim == 0:
                    aviary_val = np.array(aviary_val).item()
                engine_options.set_val(var, aviary_val, units)

    # Build EngineDeck object
    # gather arguments for building EngineDeck
    kwargs = {'options': engine_options}

    # name engine deck after filename if name is not provided
    if name is None:
        kwargs['name'] = Path(engine_options.get_val(Aircraft.Engine.DATA_FILE)).stem
    else:
        kwargs['name'] = name

    if required_variables is not None:
        kwargs['required_variables'] = required_variables

    # local import to avoid circular import
    from aviary.subsystems.propulsion.engine_deck import EngineDeck

    return EngineDeck(**kwargs)


# TODO combine with aviary/utils/data_interpolator_builder.py build_data_interpolator
class EngineDataInterpolator(om.Group):
    """
    Group that contains interpolators that get passed training data directly through
    openMDAO connections.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            default=None,
            desc='Collection of Aircraft/Mission specific options',
        )

        self.options.declare(
            'interpolator_inputs',
            types=NamedValues,
            desc='NamedValues object containing data for the independent variables of '
            'interpolation, including name, value, and units',
        )

        self.options.declare(
            'interpolator_outputs',
            types=dict,
            desc='Dictionary describing which variables will be available to the '
            'interpolator as training data at runtime, and their units',
        )

        self.options.declare(
            'interpolation_method',
            values=['slinear', 'lagrange2', 'lagrange3', 'akima'],
            default='slinear',
            desc='Interpolation method for metamodel',
        )

    def setup(self):
        num_nodes = self.options['num_nodes']
        input_data = self.options['interpolator_inputs']
        output_data = self.options['interpolator_outputs']
        interp_method = self.options['interpolation_method']

        # interpolator object for engine data
        engine = om.MetaModelSemiStructuredComp(
            method=interp_method,
            extrapolate=True,
            vec_size=num_nodes,
            training_data_gradients=True,
        )

        # Calculation of max thrust currently done with a duplicate of the engine
        # model and scaling components
        max_thrust_engine = om.MetaModelSemiStructuredComp(
            method=interp_method,
            extrapolate=True,
            vec_size=num_nodes,
            training_data_gradients=True,
        )

        # check that data in table are all vectors of the same length
        for idx, item in enumerate(get_items(input_data)):
            val = item[1][0]
            if idx != 0:
                prev_model_length = model_length
            else:
                prev_model_length = len(val)
            model_length = len(val)
            if model_length != prev_model_length:
                raise IndexError(
                    'Lengths of data provided for engine performance interpolation do not match.'
                )

        # add inputs and outputs to interpolator
        for input in get_keys(input_data):
            values, units = input_data.get_item(input)
            engine.add_input(input, training_data=values, units=units)

            if input == 'throttle':
                input = 'throttle_max'
            if input == 'hybrid_throttle':
                input = 'hybrid_throttle_max'
            max_thrust_engine.add_input(input, training_data=values, units=units)

        for output in output_data:
            engine.add_output(
                output, training_data=np.zeros(model_length), units=output_data[output]
            )
            if output == 'thrust_net':
                max_thrust_engine.add_output(
                    'thrust_net_max',
                    training_data=np.zeros(model_length),
                    units=output_data[output],
                )

        # create IndepVarComp to pass maximum throttle is to max thrust interpolator
        # currently assuming max throttle and max hybrid throttle is always 1 at every
        #   flight condition
        fixed_throttles = om.IndepVarComp()
        fixed_throttles.add_output(
            'throttle_max',
            val=np.ones(num_nodes),
            units='unitless',
            desc='Engine maximum throttle',
        )
        if 'hybrid_throttle' in input_data:
            fixed_throttles.add_output(
                'hybrid_throttle_max',
                val=np.ones(num_nodes),
                units='unitless',
                desc='Engine maximum hybrid throttle',
            )

        # add created subsystems to engine_group
        self.add_subsystem('interpolation', engine, promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('fixed_max_throttles', fixed_throttles, promotes_outputs=['*'])

        self.add_subsystem(
            'max_thrust_interpolation',
            max_thrust_engine,
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )


class UncorrectData(om.Group):
    """Calculations to recover physical parameter values that have been corrected based on ambient atmospheric conditions."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_subsystem(
            'pressure_term',
            om.ExecComp(
                'delta_T = (P0 * (1 + .2*mach**2)**3.5) / P_amb',
                delta_T={'units': 'unitless', 'shape': num_nodes},
                P0={'units': 'psi', 'shape': num_nodes},
                mach={'units': 'unitless', 'shape': num_nodes},
                P_amb={
                    'val': np.full(num_nodes, 14.696),
                    'units': 'psi',
                },
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('P0', Dynamic.Atmosphere.STATIC_PRESSURE),
                ('mach', Dynamic.Atmosphere.MACH),
            ],
            promotes_outputs=['delta_T'],
        )

        self.add_subsystem(
            'temperature_term',
            om.ExecComp(
                'theta_T = T0 * (1 + .2*mach**2)/T_amb',
                theta_T={'units': 'unitless', 'shape': num_nodes},
                T0={'units': 'degR', 'shape': num_nodes},
                mach={'units': 'unitless', 'shape': num_nodes},
                T_amb={
                    'val': np.full(num_nodes, 518.67),
                    'units': 'degR',
                },
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('T0', Dynamic.Atmosphere.TEMPERATURE),
                ('mach', Dynamic.Atmosphere.MACH),
            ],
            promotes_outputs=['theta_T'],
        )

        self.add_subsystem(
            'uncorrection',
            om.ExecComp(
                'uncorrected_data = corrected_data * (delta_T * theta_T**.5)',
                uncorrected_data={'units': 'hp', 'shape': num_nodes},
                delta_T={'units': 'unitless', 'shape': num_nodes},
                theta_T={'units': 'unitless', 'shape': num_nodes},
                corrected_data={'units': 'hp', 'shape': num_nodes},
                has_diag_partials=True,
            ),
            promotes=['*'],
        )


# class InstallationDragFlag(Enum):
#     """
#     Define constants that map to supported options for scaling of installation drag.
#     """
#     OFF = auto()
#     DELTA_MAX_NOZZLE_AREA = auto()
#     MAX_NOZZLE_AREA = auto()
#     REF_NOZZLE_EXIT_AREA = auto()


class PropellerModelVariables(Enum):
    """Define constants that map to supported variable names in a propeller model."""

    HELICAL_MACH = 'Helical_Mach'
    MACH = 'Mach'
    CP = 'CP'  # power coefficient
    CT = 'CT'  # thrust coefficient
    J = 'J'  # advanced ratio


default_propeller_units = {
    PropellerModelVariables.HELICAL_MACH: 'unitless',
    PropellerModelVariables.MACH: 'unitless',
    PropellerModelVariables.CP: 'unitless',
    PropellerModelVariables.CT: 'unitless',
    PropellerModelVariables.J: 'unitless',
}
