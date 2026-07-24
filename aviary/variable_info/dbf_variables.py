"""Extended variable hierarchy for the RC electric propulsion subsystem."""
 
from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Dynamic as av_Dynamic
 
AviaryAircraft = av_Aircraft
AviaryDynamic = av_Dynamic
 
 
class Aircraft(AviaryAircraft):
    """Aircraft data hierarchy extended for the RC electric subsystem."""
 
    class Battery(AviaryAircraft.Battery):
        VOLTAGE = 'aircraft:battery:voltage'
        RESISTANCE = 'aircraft:battery:resistance'
 
    class Engine(AviaryAircraft.Engine):
        class Motor(AviaryAircraft.Engine.Motor):
            IDLE_CURRENT = 'aircraft:engine:motor:idle_current'
            MAX_CONT_CURRENT = 'aircraft:engine:motor:max_cont_current'
            RESISTANCE = 'aircraft:engine:motor:resistance'
            KV = 'aircraft:engine:motor:kv'
            KV_EQ_SLOPE = 'aircraft:engine:motor:kv_eq_slope'
            KV_EQ_INT = 'aircraft:engine:motor:kv_eq_int'
 
        class Propeller(AviaryAircraft.Engine.Propeller):
            PITCH = 'aircraft:engine:propeller:pitch'
 
 
class Dynamic(AviaryDynamic):
    """Dynamic data hierarchy extended for the RC electric subsystem."""
 
    class Vehicle(AviaryDynamic.Vehicle):
        class Propulsion(AviaryDynamic.Vehicle.Propulsion):
            CURRENT = 'current_flow'
            CURRENT_MAX = 'current_flow_max'
            RPM_MAX = 'rotations_per_minute_max'
            PROP_POWER = 'prop_power'
            PROP_POWER_MAX = 'prop_power_max'
 