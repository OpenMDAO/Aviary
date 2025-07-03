import aviary.api as av
from aviary.examples.external_subsystems.engine_NPSS.NPSS_variables import Aircraft, Dynamic

ExtendedMetaData = av.CoreMetaData


##### ENGINE VALUES #####

av.add_meta_data(
    Aircraft.Engine.DESIGN_MACH,
    units='unitless',
    desc='Mach Number at Design Point (Cruise)',
    default_value=0.8,
    meta_data=ExtendedMetaData,
    historical_name={'NPSS': 'ambient.MN_in'},
)

av.add_meta_data(
    Aircraft.Engine.DESIGN_ALTITUDE,
    units='ft',
    desc='Altitude at Design Point (Cruise)',
    default_value=35000.0,
    meta_data=ExtendedMetaData,
    historical_name={'NPSS': 'ambient.alt_in'},
)

av.add_meta_data(
    Aircraft.Engine.DESIGN_MASS_FLOW,
    units='lbm/s',
    desc='Mass Flow at Design Point (Cruise) for a single engine',
    default_value=350.0,
    meta_data=ExtendedMetaData,
    historical_name={'NPSS': 'start.W_in'},
)

av.add_meta_data(
    Aircraft.Engine.DESIGN_NET_THRUST,
    units='lbf',
    desc='Net Thrust at Design Point (Cruise)',
    default_value=3888.1,
    meta_data=ExtendedMetaData,
    historical_name={'NPSS': 'PERF.Fn'},
)

av.add_meta_data(
    Dynamic.Engine.SHAFT_MECH_SPEED,
    units='rpm',
    desc='Mechanical Speed of Shaft for Mission From the Part Power Case',
    default_value=5000,
    meta_data=ExtendedMetaData,
    historical_name={'NPSS': 'shaft.Nmech'},
)
