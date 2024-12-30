import openmdao.utils.units as units

units.add_unit('distance_units', '1*m')

GRAV_METRIC_GASP = 9.81  # m/s^2
GRAV_ENGLISH_GASP = 32.2  # ft/s^2
GRAV_METRIC_FLOPS = 9.80665  # m/s^2
GRAV_ENGLISH_FLOPS = 32.17399  # ft/s^2
GRAV_ENGLISH_LBM = 1.0  # lbf/lbm
# TODO this does not match what dymos atmosphere comp predicts, which leads to subtle
#      problems such as density ratio not being 1 at sea level
RHO_SEA_LEVEL_ENGLISH = 0.0023769  # slug/ft^3
RHO_SEA_LEVEL_METRIC = 1.225  # kg/m^3
MU_TAKEOFF = 0.02  # TODO: fill in coefficient of friction for takeoff
MU_LANDING = 0.4  # from gasp output
# sea level standard pressure in psf
PSLS_PSF = 2116.22
# sea level standard temperature in deg R
TSLS_DEGR = 518.67
RADIUS_EARTH_METRIC = 6367533.0  # m (meridional)
