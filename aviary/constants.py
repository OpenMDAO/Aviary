from openmdao.utils import units

units.add_unit('distance_units', '1*m')

GRAV_EARTH = (
    9.80665,
    'm/s**2',
)  # NIST https://physics.nist.gov/cgi-bin/cuu/Value?gn|search_for=gravity

GRAV_MARS = (
    3.712,
    'm/s**2',
)  # Mars Global Reference Atmospheric Model (Mars-GRAM) 2024: User Guide, NASA/TM-20240012934

GRAV_VENUS = (
    8.870,
    'm/s**2',
)  # Venus Global Reference Atmospheric Model (Venus-GRAM): User Guide, NASA/TM-20210022168

# NOTE convert_geopotential_altitude() is a python utility function for engine decs that require RADIUS_EARTH to be specified in meters!
RADIUS_EARTH = (
    6371009.0,
    'm',
)  # Source: GRS80, mean earth radius (rounded to nearest meter)

RADIUS_MARS = (
    3386200.0,
    'm',
)  # Mars Global Reference Atmospheric Model (Mars-GRAM) 2024: User Guide, NASA/TM-20240012934, avg of equatorial and polar radius

RADIUS_VENUS = (
    6051800.0,
    'm',
)  # Venus Global Reference Atmospheric Model (Venus-GRAM): User Guide, NASA/TM-20210022168, avg of equatorial and polar radius

# Earth sea level standard pressure in psf
# TODO get this value from loaded atmosphere
PSLS_PSF = 2116.22

# Earth sea level standard temperature in deg R
# TODO get this value from loaded atmosphere
TSLS_DEGR = 518.67

# fuel properties
DENSITY_JET_A = (6.7, 'lbm/galUS')

# mph to knots conversion
# From NIST, 1 mph = 0.44704 m/s, 1 knot = 1852/3600 m/s. So Knots = MPH * 0.44704 / (1852/3600) =  0.8689762419
MPH_TO_KTS = (0.868976, 'unitless')
