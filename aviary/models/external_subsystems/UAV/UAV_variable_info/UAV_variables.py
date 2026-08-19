"""Extended variable hierarchy for UAV propulsion and mass subsystems."""

from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Dynamic as av_Dynamic
from aviary.variable_info.variables import Settings as av_Settings
from aviary.variable_info.variables import Mission as av_Mission

AviaryAircraft = av_Aircraft
AviaryDynamic = av_Dynamic
AviarySettings = av_Settings
AviaryMission = av_Mission

class Aircraft(AviaryAircraft):
    """Aircraft data hierarchy extended for UAV propulsion and mass."""

    class Fuselage(AviaryAircraft.Fuselage):
        AVG_HEIGHT = 'aircraft:fuselage:average_height'
        AVG_WIDTH = 'aircraft:fuselage:average_width'
        BULKHEAD_DENSITY = 'aircraft:fuselage:bulkhead_density'
        BULKHEAD_LIGHTENING_FACTOR = 'aircraft:fuselage:bulkhead_lightening_factor'
        BULKHEAD_MATERIALS = 'aircraft:fuselage:bulkhead_materials'
        BULKHEAD_THICKNESS = 'aircraft:fuselage:bulkhead_thickness'
        FLOOR_DENSITY = 'aircraft:fuselage:floor_density'
        FLOOR_LENGTH = 'aircraft:fuselage:floor_length'
        FLOOR_THICKNESS = 'aircraft:fuselage:floor_thickness'
        GLUE_FACTOR = 'aircraft:fuselage:glue_factor'
        MISC_MASS = 'aircraft:fuselage:misc_mass'
        NUM_BULKHEADS = 'aircraft:fuselage:number_of_bulkheads'
        NUM_SPARS = 'aircraft:fuselage:number_of_spars'
        SHEETING_COVERAGE = 'aircraft:fuselage:sheeting_coverage'
        SHEETING_DENSITY = 'aircraft:fuselage:sheeting_density'
        SHEETING_LIGHTENING_FACTOR = 'aircraft:fuselage:sheeting_lightening_factor'
        SHEETING_THICKNESS = 'aircraft:fuselage:sheeting_thickness'
        AREAL_SKIN_DENSITY = 'aircraft:fuselage:areal_skin_density'
        SPAR_DENSITY = 'aircraft:fuselage:spar_density'
        SPAR_OUTER_DIAMETER = 'aircraft:fuselage:spar_outer_diameter'
        SPAR_WALL_THICKNESS = 'aircraft:fuselage:spar_wall_thickness'
        STRINGER_DENSITY = 'aircraft:fuselage:stringer_density'
        STRINGER_THICKNESS = 'aircraft:fuselage:stringer_thickness'

    class HorizontalTail(AviaryAircraft.HorizontalTail):
        AIRFOIL_PATH = 'aircraft:horizontal_tail:airfoil_path'
        GLUE_FACTOR = 'aircraft:horizontal_tail:glue_factor'
        MISC_MASS = 'aircraft:horizontal_tail:misc_mass'
        NUM_RIBS = 'aircraft:horizontal_tail:number_of_ribs'
        NUM_SPARS = 'aircraft:horizontal_tail:number_of_spars'
        NUM_STRINGERS = 'aircraft:horizontal_tail:number_of_stringers'
        RIB_DENSITY = 'aircraft:horizontal_tail:rib_density'
        RIB_LIGHTENING_FACTOR = 'aircraft:horizontal_tail:rib_lightening_factor'
        RIB_MATERIALS = 'aircraft:horizontal_tail:rib_materials'
        RIB_THICKNESS = 'aircraft:horizontal_tail:rib_thickness'
        SHEETING_COVERAGE = 'aircraft:horizontal_tail:sheeting_coverage'
        SHEETING_DENSITY = 'aircraft:horizontal_tail:sheeting_density'
        SHEETING_LIGHTENING_FACTOR = 'aircraft:horizontal_tail:sheeting_lightening_factor'
        SHEETING_THICKNESS = 'aircraft:horizontal_tail:sheeting_thickness'
        AREAL_SKIN_DENSITY = 'aircraft:horizontal_tail:areal_skin_density'
        SPAR_DENSITY = 'aircraft:horizontal_tail:spar_density'
        SPAR_OUTER_DIAMETER = 'aircraft:horizontal_tail:spar_outer_diameter'
        SPAR_WALL_THICKNESS = 'aircraft:horizontal_tail:spar_wall_thickness'
        STRINGER_DENSITY = 'aircraft:horizontal_tail:stringer_density'
        STRINGER_THICKNESS = 'aircraft:horizontal_tail:stringer_thickness'

    class VerticalTail(AviaryAircraft.VerticalTail):
        AIRFOIL_PATH = 'aircraft:vertical_tail:airfoil_path'
        GLUE_FACTOR = 'aircraft:vertical_tail:glue_factor'
        MISC_MASS = 'aircraft:vertical_tail:misc_mass'
        NUM_RIBS = 'aircraft:vertical_tail:number_of_ribs'
        NUM_SPARS = 'aircraft:vertical_tail:number_of_spars'
        NUM_STRINGERS = 'aircraft:vertical_tail:number_of_stringers'
        RIB_DENSITY = 'aircraft:vertical_tail:rib_density'
        RIB_LIGHTENING_FACTOR = 'aircraft:vertical_tail:rib_lightening_factor'
        RIB_MATERIALS = 'aircraft:vertical_tail:rib_materials'
        RIB_THICKNESS = 'aircraft:vertical_tail:rib_thickness'
        SHEETING_COVERAGE = 'aircraft:vertical_tail:sheeting_coverage'
        SHEETING_DENSITY = 'aircraft:vertical_tail:sheeting_density'
        SHEETING_LIGHTENING_FACTOR = 'aircraft:vertical_tail:sheeting_lightening_factor'
        SHEETING_THICKNESS = 'aircraft:vertical_tail:sheeting_thickness'
        AREAL_SKIN_DENSITY = 'aircraft:vertical_tail:areal_skin_density'
        SPAR_DENSITY = 'aircraft:vertical_tail:spar_density'
        SPAR_OUTER_DIAMETER = 'aircraft:vertical_tail:spar_outer_diameter'
        SPAR_WALL_THICKNESS = 'aircraft:vertical_tail:spar_wall_thickness'
        STRINGER_DENSITY = 'aircraft:vertical_tail:stringer_density'
        STRINGER_THICKNESS = 'aircraft:vertical_tail:stringer_thickness'

    class Wing(AviaryAircraft.Wing):
        AIRFOIL_PATH = 'aircraft:wing:airfoil_path'
        FOAM_DENSITY = 'aircraft:wing:foam_density'
        GLUE_FACTOR = 'aircraft:wing:glue_factor'
        NUM_RIBS = 'aircraft:wing:number_of_ribs'
        NUM_SPARS = 'aircraft:wing:number_of_spars'
        NUM_STRINGERS = 'aircraft:wing:number_of_stringers'
        RIB_DENSITY = 'aircraft:wing:rib_density'
        RIB_LIGHTENING_FACTOR = 'aircraft:wing:rib_lightening_factor'
        RIB_MATERIALS = 'aircraft:wing:rib_materials'
        RIB_THICKNESS = 'aircraft:wing:rib_thickness'
        ROD_DENSITY = 'aircraft:wing:rod_density'
        ROD_RADIUS = 'aircraft:wing:rod_radius'
        ROD_THICKNESS = 'aircraft:wing:rod_thickness'
        SHEETING_COVERAGE = 'aircraft:wing:sheeting_coverage'
        SHEETING_DENSITY = 'aircraft:wing:sheeting_density'
        SHEETING_LIGHTENING_FACTOR = 'aircraft:wing:sheeting_lightening_factor'
        SHEETING_THICKNESS = 'aircraft:wing:sheeting_thickness'
        AREAL_SKIN_DENSITY = 'aircraft:wing:areal_skin_density'
        SPAR_DENSITY = 'aircraft:wing:spar_density'
        SPAR_OUTER_DIAMETER = 'aircraft:wing:spar_outer_diameter'
        SPAR_WALL_THICKNESS = 'aircraft:wing:spar_wall_thickness'
        STRINGER_DENSITY = 'aircraft:wing:stringer_density'
        STRINGER_THICKNESS = 'aircraft:wing:stringer_thickness'
        TYPE = 'aircraft:wing:type'

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

class Settings(AviarySettings):
    "needed for imports to recognize settings variables"
    pass

class Mission(AviaryMission):
    "needed for imports to recognize mission variables"
    pass