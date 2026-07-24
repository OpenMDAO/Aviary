from aviary.variable_info.variables import Aircraft as av_Aircraft

"""
This is an extension of the variable hierarchy that adds variables for UAV aircraft mass calculations.
"""

AviaryAircraft = av_Aircraft

class Aircraft(AviaryAircraft):

    class Fuselage(AviaryAircraft.Fuselage):
        #Fuselage Mass Inputs
        AVG_HEIGHT = 'aircraft:fuselage:average_height'
        AVG_WIDTH = 'aircraft:fuselage:average_width'

        #Fuselage Mass Options
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
        #Horizontal Tail Mass Options
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
        #Vertical Tail Mass Options
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
        #Wing Mass Options
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