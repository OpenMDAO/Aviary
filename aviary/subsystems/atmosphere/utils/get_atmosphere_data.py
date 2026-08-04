import numpy as np

from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Cold import atm_data as cold_210A
from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Hot import atm_data as hot_210A
from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Polar import atm_data as polar_210A
from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Tropical import atm_data as tropical_210A
from aviary.subsystems.atmosphere.data.StandardAtm1976 import atm_data as USatm1976
from aviary.subsystems.atmosphere.data.MarsReference2024 import atm_data as MarsReference2024
from aviary.subsystems.atmosphere.data.MarsHellasHot import atm_data as MarsHellasHot
from aviary.subsystems.atmosphere.data.MarsHellasCold import atm_data as MarsHellasCold
from aviary.subsystems.atmosphere.data.MarsEquatorHot import atm_data as MarsEquatorHot
from aviary.subsystems.atmosphere.data.MarsEquatorCold import atm_data as MarsEquatorCold
from aviary.subsystems.atmosphere.data.MarsPolarHot import atm_data as MarsPolarHot
from aviary.subsystems.atmosphere.data.MarsPolarCold import atm_data as MarsPolarCold
from aviary.subsystems.atmosphere.data.VenusReference2021 import atm_data as VenusReference2021
from aviary.variable_info.enums import AtmosphereModel
from aviary.constants import (
    RADIUS_EARTH,
    RADIUS_MARS,
    RADIUS_VENUS,
    GRAV_EARTH,
    GRAV_MARS,
    GRAV_VENUS,
)


def get_atmosphere_data(atmosphere_model=AtmosphereModel.STANDARD):
    """
    Return the atmosphere source data, planet name, and gravitational constant
    associated with the requested atmosphere model.

    Parameters
    ----------
    atmosphere_model : AtmosphereModel
        Atmosphere model enum.

    Returns
    -------
    tuple
        (source_data, planet, radius, gravity, sea_level_density)
    """
    atmosphere_lookup = {
        AtmosphereModel.STANDARD: (USatm1976, 'Earth', RADIUS_EARTH, GRAV_EARTH),
        AtmosphereModel.TROPICAL: (tropical_210A, 'Earth', RADIUS_EARTH, GRAV_EARTH),
        AtmosphereModel.POLAR: (polar_210A, 'Earth', RADIUS_EARTH, GRAV_EARTH),
        AtmosphereModel.HOT: (hot_210A, 'Earth', RADIUS_EARTH, GRAV_EARTH),
        AtmosphereModel.COLD: (cold_210A, 'Earth', RADIUS_EARTH, GRAV_EARTH),
        AtmosphereModel.MARS_REFERENCE: (MarsReference2024, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.MARS_HELLAS_HOT: (MarsHellasHot, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.MARS_HELLAS_COLD: (MarsHellasCold, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.MARS_EQUATOR_HOT: (MarsEquatorHot, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.MARS_EQUATOR_COLD: (MarsEquatorCold, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.MARS_POLAR_HOT: (MarsPolarHot, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.MARS_POLAR_COLD: (MarsPolarCold, 'Mars', RADIUS_MARS, GRAV_MARS),
        AtmosphereModel.VENUS_REFERENCE: (VenusReference2021, 'Venus', RADIUS_VENUS, GRAV_VENUS),
    }

    try:
        data = atmosphere_lookup[atmosphere_model]
    except KeyError:
        raise ValueError(f'Could not find {atmosphere_model} in get_atmosphere_data().')

    # Lookup sea level density in the atmosphere model.
    source_data = data[0]
    altitudes = source_data.alt
    idx = np.argwhere(altitudes == 0.0)[0][0] + 1
    sea_level_density = (float(source_data.akima_rho[idx][0]), 'kg/m**3')

    return tuple([*data, sea_level_density])
