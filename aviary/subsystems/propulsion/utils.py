"""
Attributes
----------
default_units : dict
    Matches each EngineModelVariables entry with default units (str)
"""
from enum import Enum, auto

import aviary.constants as constants


class EngineModelVariables(Enum):
    '''
    Define constants that map to supported variable names in an engine model.
    '''
    MACH = auto()
    ALTITUDE = auto()
    THROTTLE = auto()
    HYBRID_THROTTLE = auto()
    THRUST = auto()
    GROSS_THRUST = auto()
    RAM_DRAG = auto()
    FUEL_FLOW = auto()
    ELECTRIC_POWER = auto()
    NOX_RATE = auto()
    TEMPERATURE_ENGINE_T4 = auto()
    # EXIT_AREA = auto()


default_units = {
    EngineModelVariables.MACH: 'unitless',
    EngineModelVariables.ALTITUDE: 'ft',
    EngineModelVariables.THROTTLE: 'unitless',
    EngineModelVariables.HYBRID_THROTTLE: 'unitless',
    EngineModelVariables.THRUST: 'lbf',
    EngineModelVariables.GROSS_THRUST: 'lbf',
    EngineModelVariables.RAM_DRAG: 'lbf',
    EngineModelVariables.FUEL_FLOW: 'lb/h',
    EngineModelVariables.ELECTRIC_POWER: 'kW',
    EngineModelVariables.NOX_RATE: 'lb/h',
    EngineModelVariables.TEMPERATURE_ENGINE_T4: 'degR'
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


# class InstallationDragFlag(Enum):
#     '''
#     Define constants that map to supported options for scaling of installation drag.
#     '''
#     OFF = auto()
#     DELTA_MAX_NOZZLE_AREA = auto()
#     MAX_NOZZLE_AREA = auto()
#     REF_NOZZLE_EXIT_AREA = auto()
