"""
The following test data comes from a file which was LEAPS2/problems/full_mission_test.py
results. The file does not exist anymore, but the data is used for regression testing.
The analysis converged using SNOPT with 0/1 results and good plots.
One point was taken from each phase: climb, cruise, and descent.

Notes
-----
FLOPS/LEAPS1 mission analyses data cannot be used to unit test Aviary mission
analysis data, because the methodologies are too different. FLOPS/LEAPS1
mission analyses use implementations that do not rely on some Aviary variables,
such as altitude rate and velocity rate. These missing values cannot be derived
from the available FLOPS/LEAPS1 data.
"""

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

data = AviaryValues()

data.set_val(
    Mission.Design.GROSS_MASS,
    val=181200.0,
    units='lbm',
)

data.set_val(
    Aircraft.Wing.AREA,
    val=1341,
    units='ft**2',
)

data.set_val(
    Mission.Takeoff.FUEL_SIMPLE,
    val=577,
    units='lbm',
)

data.set_val(
    Mission.Takeoff.LIFT_COEFFICIENT_MAX,
    val=2,
    units='unitless',
)

data.set_val(
    Mission.Takeoff.LIFT_OVER_DRAG,
    val=17.35,
    units='unitless',
)

data.set_val(
    Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
    val=0.0175,
    units='unitless',
)

data.set_val(
    Mission.Design.THRUST_TAKEOFF_PER_ENG,
    val=24555.5,
    units='lbf',
)

data.set_val(
    # states:altitude
    Dynamic.Mission.ALTITUDE,
    val=[
        29.3112920637369,
        10668,
        26.3564405194251,
    ],
    units='m',
)

data.set_val(
    # outputs
    Dynamic.Mission.ALTITUDE_RATE,
    val=[
        29.8463233754212,
        -5.69941245767868e-09,
        -4.32644785970493,
    ],
    units='ft/s',
)

data.set_val(
    # outputs
    Dynamic.Mission.ALTITUDE_RATE_MAX,
    val=[
        3679.0525544843,
        3.86361517135375,
        6557.07891846677,
    ],
    units='ft/min',
)

data.set_val(
    # outputs
    Dynamic.Vehicle.DRAG,
    val=[
        9978.32211087097,
        8769.90342254821,
        7235.03338269778,
    ],
    units='lbf',
)

data.set_val(
    # outputs
    Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE,
    val=[
        16602.302762413,
        5551.61304633633,
        1286,
    ],
    units='lbm/h',
)

data.set_val(
    Dynamic.Atmosphere.MACH,
    val=[
        0.482191004489294,
        0.785,
        0.345807620281699,
    ],
    units='unitless',
)

data.set_val(
    # states:mass
    Dynamic.Vehicle.MASS,
    val=[
        81796.1389890711,
        74616.9849763798,
        65193.7423491884,
    ],
    units='kg',
)

# TODO: double check values
data.set_val(
    Dynamic.Vehicle.Propulsion.THROTTLE,
    val=[
        0.5,
        0.5,
        0.0,
    ],
    units='unitless',
)

# TODO: double check values
data.set_val(
    'throttle_max',
    val=[
        1.0,
        1.0,
        1.0,
    ],
    units='unitless',
)

data.set_val(
    # state_rates:range
    Dynamic.Mission.DISTANCE_RATE,
    val=[
        163.776550884386,
        232.775306059091,
        117.631414542995,
    ],
    units='m/s',
)

data.set_val(
    # outputs
    Dynamic.Mission.SPECIFIC_ENERGY_RATE,
    val=[
        18.4428113202544191,
        -1.7371801250963e-9,
        -5.9217623736010768,
    ],
    units='m/s',
)

data.set_val(
    # outputs
    Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
    val=[
        28.03523893220630,
        3.8636151713537548,
        28.706899839848,
    ],
    units='m/s',
)

data.set_val(
    # outputs
    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
    val=[
        30253.9128379374,
        8769.90342132054,
        0,
    ],
    units='lbf',
)

data.set_val(
    # outputs
    Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
    val=[
        40799.6009633346,
        11500.32,
        42308.2709683461,
    ],
    units='lbf',
)

data.set_val(
    # outputs
    'time',
    val=[
        85.8098321364695,
        8852.1530657591,
        23851.1097964271,
    ],
    units='s',
)

data.set_val(
    # states:velocity
    Dynamic.Mission.VELOCITY,
    val=[
        164.029012458452,
        232.775306059091,
        117.638805929526,
    ],
    units='m/s',
)

data.set_val(
    # state_rates:velocity
    Dynamic.Mission.VELOCITY_RATE,
    val=[
        0.558739800813549,
        3.33665416459715e-17,
        -0.38372209277242,
    ],
    units='m/s**2',
)
