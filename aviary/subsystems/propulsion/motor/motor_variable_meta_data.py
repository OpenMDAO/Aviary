import aviary.api as av
from aviary.subsystems.propulsion.motor.motor_variables import Aircraft, Dynamic


ExtendedMetaData = av.CoreMetaData

##### MOTOR VALUES #####

av.add_meta_data(
    Aircraft.Motor.MASS,
    units="kg",
    desc="Total motor mass (considers number of motors)",
    default_value=1.0,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Motor.RPM,
    units="rpm",
    desc="Motor RPM",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Motor.TORQUE_MAX,
    units="N*m",
    desc="Max torque value that can be output from a single motor. Used to determine motor mass in pre-mission",
    meta_data=ExtendedMetaData
)

##### MOTOR MISSION VALUES #####

av.add_meta_data(
    Dynamic.Mission.Motor.EFFICIENCY,
    units=None,
    desc="Motor efficiency",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Dynamic.Mission.TORQUE,
    units="N*m",
    desc="Motor torque",
    default_value=None,
    meta_data=ExtendedMetaData
)

# av.add_meta_data(
#     Dynamic.Mission.Motor.TORQUE_CON,
#     units="N*m",
#     desc="Motor torque constraint to ensure torque in mission is less than torque_max. Only use if you don't know your torque_max a-priori",
#     default_value=None,
#     meta_data=ExtendedMetaData
# )

##### PROP VALUES #####

av.add_meta_data(
    Aircraft.Prop.RPM,
    units="rpm",
    desc="Prop RPM",
    default_value=None,
    meta_data=ExtendedMetaData
)

# av.add_meta_data(
#     Dynamic.Mission.Prop.TORQUE,
#     units="N*m",
#     desc="Torque output to a single propellar shaft from a motor/turbine/gearbox",
#     default_value=None,
#     meta_data=ExtendedMetaData
# )

####### GEARBOX VALUES ######
av.add_meta_data(
    Aircraft.Gearbox.GEAR_RATIO,
    units=None,
    desc="The ratio of the RPM_out divided by the RPM_in for the gearbox.",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Aircraft.Gearbox.MASS,
    units='kg',
    desc="The mass of the gearbox.",
    default_value=None,
    meta_data=ExtendedMetaData
)

av.add_meta_data(
    Dynamic.Mission.Gearbox.EFFICIENCY,
    units=None,
    desc="The efficiency of the gearbox during the mission",
    default_value=1.0,
    meta_data=ExtendedMetaData
)
