"""Extended variable metadata for the RC electric propulsion subsystem."""



from copy import deepcopy

import aviary.api as av
from aviary.variable_info.dbf_variables import Aircraft, Dynamic

ExtendedMetaData = deepcopy(av.CoreMetaData)


##### RC Variables #####

# --- Battery ---
av.add_meta_data(
    Aircraft.Battery.VOLTAGE,
    meta_data=ExtendedMetaData,
    units='V',
    desc='Total voltage input from the battery pack',
    default_value=22.2,
)

av.add_meta_data(
    Aircraft.Battery.RESISTANCE,
    meta_data=ExtendedMetaData,
    units='ohm',
    desc='Internal resistance of the battery pack',
    default_value=0.05,
)

# --- Motor ---
av.add_meta_data(
    Aircraft.Engine.Motor.IDLE_CURRENT,
    meta_data=ExtendedMetaData,
    units='A',
    desc='The idle or no-load current from a single motor.',
    multivalue=True,
)

av.add_meta_data(
    Aircraft.Engine.Motor.MAX_CONT_CURRENT,
    meta_data=ExtendedMetaData,
    units='A',
    desc='Maximum continuous current that flows through a single motor.',
    multivalue=True,
)

av.add_meta_data(
    Aircraft.Engine.Motor.RESISTANCE,
    meta_data=ExtendedMetaData,
    units='ohm',
    desc='Resistance from windings of a single motor.',
    multivalue=True,
)

av.add_meta_data(
    Aircraft.Engine.Motor.KV,
    meta_data=ExtendedMetaData,
    units='rpm/V',
    desc='Speed constant of a single motor.',
    multivalue=True,
)

av.add_meta_data(
    Aircraft.Engine.Motor.KV_EQ_SLOPE,
    meta_data=ExtendedMetaData,
    units='unitless',
    desc='Slope of the linear fit relating motor KV to mass.',
    default_value=2105.53674,
    option=True,
    multivalue=True,
)

av.add_meta_data(
    Aircraft.Engine.Motor.KV_EQ_INT,
    meta_data=ExtendedMetaData,
    units='unitless',
    desc='Intercept of the linear fit relating motor KV to mass.',
    default_value=-80.83469,
    option=True,
    multivalue=True,
)

# --- Propeller ---
av.add_meta_data(
    Aircraft.Engine.Propeller.PITCH,
    meta_data=ExtendedMetaData,
    units='inch',
    desc='Forward distance a propeller advances through one revolution.',
    multivalue=True,
)

# --- Dynamic propulsion ---
av.add_meta_data(
    Dynamic.Vehicle.Propulsion.CURRENT,
    meta_data=ExtendedMetaData,
    units='A',
    desc='Electrical current flow through an engine',
    multivalue=True,
)

av.add_meta_data(
    Dynamic.Vehicle.Propulsion.CURRENT_MAX,
    meta_data=ExtendedMetaData,
    units='A',
    desc='Electrical current flow through an engine at full throttle',
    multivalue=True,
)

av.add_meta_data(
    Dynamic.Vehicle.Propulsion.RPM_MAX,
    meta_data=ExtendedMetaData,
    units='rpm',
    desc='Rotational rate of shaft, per engine, at max throttle condition.',
    multivalue=True,
)

av.add_meta_data(
    Dynamic.Vehicle.Propulsion.PROP_POWER,
    meta_data=ExtendedMetaData,
    units='W',
    desc='Power output from a propeller',
    default_value=0.0,
    multivalue=True,
)

av.add_meta_data(
    Dynamic.Vehicle.Propulsion.PROP_POWER_MAX,
    meta_data=ExtendedMetaData,
    units='W',
    desc='Power output from a propeller at full throttle',
    default_value=0.0,
    multivalue=True,
)
