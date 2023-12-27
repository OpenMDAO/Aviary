from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Mission as av_Mission

AviaryAircraft = av_Aircraft
AviaryMission = av_Mission


# ---------------------------
# Aircraft data hierarchy
# ---------------------------

class Aircraft(AviaryAircraft):

    # cell = single cell, battery = one case plus multiple cells

    class Battery:
        CURRENT_MAX = "aircraft:battery:current_max"
        EFFICIENCY = "aircraft:battery:efficiency"
        ENERGY_REQUIRED = "aircraft:battery:energy_required"
        HEAT_CAPACITY = "aircraft:battery:heat_capacity"
        MASS = "aircraft:battery:mass"
        N_PARALLEL = "aircraft:battery:n_parallel"
        N_SERIES = "aircraft:battery:n_series"
        VOLTAGE = "aircraft:battery:voltage"
        VOLUME = "aircraft:battery:volume"

        class Case:
            HEAT_CAPACITY = "aircraft:battery:case:heat_capacity"
            WEIGHT_FRAC = "aircraft:battery:case:weight_frac"

        class Cell:
            DISCHARGE_RATE = "aircraft:battery:cell:discharge_rate"
            ENERGY_CAPACITY_MAX = "aircraft:battery:cell:energy_capacity_max"
            HEAT_CAPACITY = "aircraft:battery:cell:heat_capacity"
            MASS = "aircraft:battery:cell:mass"
            VOLTAGE_LOW = "aircraft:battery:cell:voltage_low"
            VOLUME = "aircraft:battery:cell:volume"
            TYPE = "aircraft:battery:cell:type"

# ---------------------------
# Mission data hierarchy
# ---------------------------


class Mission(AviaryMission):

    class Battery:
        CURRENT = "mission:battery:current"
        HEAT_OUT = "mission:battery:heat_out"
        STATE_OF_CHARGE = "mission:battery:state_of_charge"
        STATE_OF_CHARGE_RATE = "mission:battery:state_of_charge_rate"
        TEMPERATURE = "mission:battery:temperature"
        VOLTAGE = "mission:battery:voltage"
        VOLTAGE_THEVENIN = "mission:battery:voltage_thevenin"
        VOLTAGE_THEVENIN_RATE = "mission:battery:voltage_thevenin_rate"
