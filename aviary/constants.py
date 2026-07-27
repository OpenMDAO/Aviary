import openmdao.utils.units as units

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

RADIUS_EARTH = (6371009, 'm')  # Source: GRS80, mean earth radius (rounded to nearest meter)
# convert_geopotential_altitude() is a python utility function that require RADIUS_EARTH to be specified in meters!

RADIUS_MARS = (
    3386200,
    'm',
)  # Mars Global Reference Atmospheric Model (Mars-GRAM) 2024: User Guide, NASA/TM-20240012934, avg of equatorial and polar radius
RADIUS_VENUS = (
    6051800,
    'm',
)  # Venus Global Reference Atmospheric Model (Venus-GRAM): User Guide, NASA/TM-20210022168, avg of equatorial and polar radius

# GNS = 9.8236930  # grav_accel_at_surface_earth # TODO: Remove this from other parts of Aviary
GRAV_METRIC_GASP = 9.81  # m/s^2
GRAV_ENGLISH_GASP = 32.2  # ft/s^2
GRAV_ENGLISH_LBM = 1.0  # lbf/lbm

# sea level standard pressure in psf
PSLS_PSF = 2116.22

# sea level standard temperature in deg R
TSLS_DEGR = 518.67
