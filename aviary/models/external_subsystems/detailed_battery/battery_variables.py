from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Dynamic as av_Dynamic

AviaryAircraft = av_Aircraft
AviaryDynamic = av_Dynamic


class Aircraft(AviaryAircraft):
    """Aircraft data hierarchy for battery subsystem."""

    # cell = single cell, battery = one case plus multiple cells

    class Battery(AviaryAircraft.Battery):
        CURRENT_MAX = 'aircraft:battery:current_max'
        ENERGY_REQUIRED = 'aircraft:battery:energy_required'
        HEAT_CAPACITY = 'aircraft:battery:heat_capacity'
        N_PARALLEL = 'aircraft:battery:n_parallel'
        N_SERIES = 'aircraft:battery:n_series'
        VOLTAGE = 'aircraft:battery:voltage'

        class Case:
            HEAT_CAPACITY = 'aircraft:battery:case:heat_capacity'
            WEIGHT_FRAC = 'aircraft:battery:case:weight_frac'

        class Cell:
            DISCHARGE_RATE = 'aircraft:battery:cell:discharge_rate'
            ENERGY_CAPACITY_MAX = 'aircraft:battery:cell:energy_capacity_max'
            HEAT_CAPACITY = 'aircraft:battery:cell:heat_capacity'
            MASS = 'aircraft:battery:cell:mass'
            VOLTAGE_LOW = 'aircraft:battery:cell:voltage_low'
            VOLUME = 'aircraft:battery:cell:volume'
            TYPE = 'aircraft:battery:cell:type'


class Dynamic(AviaryDynamic):
    """Dynamic data hierarchy for battery subsystem."""

    class Battery:
        CURRENT = 'dynamic:battery:current'
        EFFICIENCY = 'dynamic:battery:efficiency'
        HEAT_OUT = 'dynamic:battery:heat_out'
        STATE_OF_CHARGE = 'dynamic:battery:state_of_charge'
        STATE_OF_CHARGE_RATE = 'dynamic:battery:state_of_charge_rate'
        TEMPERATURE = 'dynamic:battery:temperature'
        VOLTAGE = 'dynamic:battery:voltage'
        VOLTAGE_THEVENIN = 'dynamic:battery:voltage_thevenin'
        VOLTAGE_THEVENIN_RATE = 'dynamic:battery:voltage_thevenin_rate'
