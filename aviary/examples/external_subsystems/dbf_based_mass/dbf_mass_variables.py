from aviary.variable_info.variables import Aircraft as av_Aircraft

AviaryAircraft = av_Aircraft


class Aircraft(AviaryAircraft):
    """Aircraft data hierarchy for  mass subsystem."""

    class Wing(AviaryAircraft.Wing):
        # Custom added for  -- Make sure to sort later
        NUM_SPARS = 'aircraft:wing:number_of_spars'
        SPAR_THICKNESS = 'aircraft:wing:spar_thickness'
        NUM_RIBS = 'aircraft:wing:number_of_ribs'
        RIB_THICKNESS = 'aircraft:wing:rib_thickness'
        SPAR_DENSITY = 'aircraft:wing:spar_density'
        RIB_DENSITY = 'aircraft:wing:rib_density'
        SKIN_DENSITY = 'aircraft:wing:skin_density'
        AIRFOIL_X_COORDS = 'aircraft:wing:airfoil_x_coords'
        AIRFOIL_Y_COORDS = 'aircraft:wing:airfoil_y_coords'

    class HorizontalTail(AviaryAircraft.HorizontalTail):
        NUM_SPARS = 'aircraft:horizontaltail:number_of_spars'
        SPAR_THICKNESS = 'aircraft:horizontaltail:spar_thickness'
        NUM_RIBS = 'aircraft:horizontaltail:number_of_ribs'
        RIB_THICKNESS = 'aircraft:horizontaltail:rib_thickness'
        SPAR_DENSITY = 'aircraft:horizontaltail:spar_density'
        RIB_DENSITY = 'aircraft:horizontaltail:rib_density'
        SKIN_DENSITY = 'aircraft:horizontaltail:skin_density'
        AIRFOIL_X_COORDS = 'aircraft:horizontaltail:airfoil_x_coords'
        AIRFOIL_Y_COORDS = 'aircraft:horizontaltail:airfoil_y_coords'

    class VerticalTail(AviaryAircraft.VerticalTail):
        NUM_SPARS = 'aircraft:verticaltail:number_of_spars'
        SPAR_THICKNESS = 'aircraft:verticaltail:spar_thickness'
        NUM_RIBS = 'aircraft:verticaltail:number_of_ribs'
        RIB_THICKNESS = 'aircraft:verticaltail:rib_thickness'
        SPAR_DENSITY = 'aircraft:verticaltail:spar_density'
        RIB_DENSITY = 'aircraft:verticaltail:rib_density'
        SKIN_DENSITY = 'aircraft:verticaltail:skin_density'
        AIRFOIL_X_COORDS = 'aircraft:verticaltail:airfoil_x_coords'
        AIRFOIL_Y_COORDS = 'aircraft:verticaltail:airfoil_y_coords'

    class Fuselage(AviaryAircraft.Fuselage):
        NUM_SPARS = 'aircraft:fuselage:number_of_spars'
        SPAR_THICKNESS = 'aircraft:fuselage:spar_thickness'
        NUM_RIBS = 'aircraft:fuselage:number_of_ribs'
        RIB_THICKNESS = 'aircraft:fuselage:rib_thickness'
        SPAR_DENSITY = 'aircraft:fuselage:spar_density'
        RIB_DENSITY = 'aircraft:fuselage:rib_density'
        SKIN_DENSITY = 'aircraft:fuselage:skin_density'
        FUSELAGE_HEIGHT = 'aircraft:fuselage:fuselage_height'
        FUSELAGE_WIDTH = 'aircraft:fuselage:fuselage_width'
