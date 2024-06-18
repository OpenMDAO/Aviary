"""
Attributes
----------
default_units : dict
    Matches each EngineModelVariables entry with default units (str)
"""
from enum import Enum, auto
from pathlib import Path

import numpy as np
import openmdao.api as om

import aviary.constants as constants

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft


class EngineModelVariables(Enum):
    '''
    Define constants that map to supported variable names in an engine model.
    '''
    MACH = auto()
    ALTITUDE = auto()
    THROTTLE = auto()
    HYBRID_THROTTLE = auto()
    THRUST = auto()
    TAILPIPE_THRUST = auto()
    GROSS_THRUST = auto()
    SHAFT_POWER = auto()
    SHAFT_POWER_CORRECTED = auto()
    RAM_DRAG = auto()
    FUEL_FLOW = auto()
    ELECTRIC_POWER = auto()
    NOX_RATE = auto()
    TEMPERATURE_ENGINE_T4 = auto()
    TORQUE = auto()
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
    EngineModelVariables.ELECTRIC_POWER: 'kW',
    EngineModelVariables.NOX_RATE: 'lb/h',
    EngineModelVariables.TEMPERATURE_ENGINE_T4: 'degR',
    EngineModelVariables.TORQUE: 'ft*lbf'
    # EngineModelVariables.EXIT_AREA: 'ft**2',
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
    ----------
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

    for (i, alt) in enumerate(altitude):
        HFT = alt
        HO = HFT * .30480  # convert ft to m
        Z = (HFT + (4.37 * (10 ** -8)) * (HFT ** 2.00850)) * .30480

        DH = float('inf')

        while abs(DH) > 1.0:
            R = radius_earth + Z
            GN = GNS * (radius_earth / R) ** (CM1 + 1.0)
            H =\
                (R * GN * ((R / radius_earth)**CM1 - 1.0)
                    / CM1 - Z * (R - Z / 2.0) * OC2) / g

            DH = HO - H
            Z += DH

        alt = Z / .30480  # convert m to ft
        altitude[i] = alt

    return altitude


# TODO build test for this function
def build_engine_deck(aviary_options: AviaryValues):
    '''
    Creates an EngineDeck using avaliable inputs and options in aviary_options.

    Parameter
    ----------
    aviary_options : AviaryValues
        Options to use in creation of EngineDecks.

    Returns
    ----------
    engine_models : <list of EngineDecks>
        List of EngineDecks created using provided aviary_options.
    '''
    # import locally placed to avoid circular import
    from aviary.subsystems.propulsion.engine_deck import EngineDeck

    # Build a single engine deck, currently ignoring vectorization
    # of AviaryValues (use first index)
    engine_options = AviaryValues()
    for var in Aircraft.Engine.__dict__.values():
        # check if this variable exist with useable metadata
        try:
            units = _MetaData[var]['units']
            try:
                # add value from aviary_options to engine_options
                aviary_val = aviary_options.get_val(var, units)
                if isinstance(aviary_val, np.ndarray):
                    # "Convert" numpy types to standard Python types. Wrap first
                    # index in numpy array before calling item() to safeguard against
                    # non-standard types, such as objects
                    aviary_val = np.array(aviary_val[0]).item()
                elif isinstance(aviary_val, (list, tuple)):
                    aviary_val = aviary_val[0]
                engine_options.set_val(var, aviary_val, units)
            # if not, use default value from _MetaData?
            except KeyError:
                # engine_options.set_val(var, _MetaData[var]['default_value'], units)
                continue
        except (KeyError, TypeError):
            continue

    # name engine deck after filename
    return [EngineDeck(Path(engine_options.get_val(Aircraft.Engine.DATA_FILE)).stem, options=engine_options)]


class UncorrectData(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, default=1)
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        num_nodes = self.options['num_nodes']

        self.add_subsystem(
            'pressure_term',
            om.ExecComp(
                'delta_T = (P0 * (1 + .2*mach**2)**3.5) / P_amb',
                delta_T={'units': "unitless", 'shape': num_nodes},
                P0={'units': 'psi', 'shape': num_nodes},
                mach={'units': 'unitless', 'shape': num_nodes},
                P_amb={'val': np.full(num_nodes, 14.696), 'units': 'psi', },
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('P0', Dynamic.Mission.STATIC_PRESSURE),
                ('mach', Dynamic.Mission.MACH),
            ],
            promotes_outputs=['delta_T'],
        )

        self.add_subsystem(
            'temperature_term',
            om.ExecComp(
                'theta_T = T0 * (1 + .2*mach**2)/T_amb',
                theta_T={'units': "unitless", 'shape': num_nodes},
                T0={'units': 'degR', 'shape': num_nodes},
                mach={'units': 'unitless', 'shape': num_nodes},
                T_amb={'val': np.full(num_nodes, 518.67), 'units': 'degR', },
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('T0', Dynamic.Mission.TEMPERATURE),
                ('mach', Dynamic.Mission.MACH),
            ],
            promotes_outputs=['theta_T'],
        )

        self.add_subsystem(
            'uncorrection',
            om.ExecComp(
                'uncorrected_data = corrected_data * (delta_T + theta_T**.5)',
                uncorrected_data={'units': "hp", 'shape': num_nodes},
                delta_T={'units': "unitless", 'shape': num_nodes},
                theta_T={'units': "unitless", 'shape': num_nodes},
                corrected_data={'units': "hp", 'shape': num_nodes},
                has_diag_partials=True,
            ),
            promotes=['*']
        )


# class InstallationDragFlag(Enum):
#     '''
#     Define constants that map to supported options for scaling of installation drag.
#     '''
#     OFF = auto()
#     DELTA_MAX_NOZZLE_AREA = auto()
#     MAX_NOZZLE_AREA = auto()
#     REF_NOZZLE_EXIT_AREA = auto()
