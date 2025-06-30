from aviary.variable_info.variables import Aircraft as av_Aircraft

AviaryAircraft = av_Aircraft


class Aircraft(AviaryAircraft):
    """Aircraft data hierarchy for battery subsystem."""

    # cell = single cell, battery = one case plus multiple cells

    class DBFWing(AviaryAircraft.Wing):
        # Custom added for DBF -- Make sure to sort later
        NUM_SPARS = 'aircraft:DBFwing:number_of_spars'
        SPAR_THICKNESS = 'aircraft:DBFwing:spar_thickness'
        NUM_RIBS = 'aircraft:DBFwing:number_of_ribs'
        RIB_THICKNESS = 'aircraft:DBFwing:rib_thickness'
        SPAR_DENSITY = 'aircraft:DBFwing:spar_density'
        RIB_DENSITY = 'aircraft:DBFwing:rib_density'
        SKIN_DENSITY = 'aircraft:DBFwing:skin_density'
        AIRFOIL_X_COORDS = 'aircraft:DBFwing:airfoil_x_coords'
        AIRFOIL_Y_COORDS = 'aircraft:DBFwing:airfoil_y_coords'

    class DBFHorizontalTail(AviaryAircraft.HorizontalTail):
        NUM_SPARS = 'aircraft:DBFHorizontalTail:number_of_spars'
        SPAR_THICKNESS = 'aircraft:DBFHorizontalTail:spar_thickness'
        NUM_RIBS = 'aircraft:DBFHorizontalTail:number_of_ribs'
        RIB_THICKNESS = 'aircraft:DBFHorizontalTail:rib_thickness'
        SPAR_DENSITY = 'aircraft:DBFHorizontalTail:spar_density'
        RIB_DENSITY = 'aircraft:DBFHorizontalTail:rib_density'
        SKIN_DENSITY = 'aircraft:DBFHorizontalTail:skin_density'
        AIRFOIL_X_COORDS = 'aircraft:DBFHorizontalTail:airfoil_x_coords'
        AIRFOIL_Y_COORDS = 'aircraft:DBFHorizontalTail:airfoil_y_coords'

    class DBFVerticalTail(AviaryAircraft.VerticalTail):
        NUM_SPARS = 'aircraft:DBFVerticalTail:number_of_spars'
        SPAR_THICKNESS = 'aircraft:DBFVerticalTail:spar_thickness'
        NUM_RIBS = 'aircraft:DBFVerticalTail:number_of_ribs'
        RIB_THICKNESS = 'aircraft:DBFVerticalTail:rib_thickness'
        SPAR_DENSITY = 'aircraft:DBFVerticalTail:spar_density'
        RIB_DENSITY = 'aircraft:DBFVerticalTail:rib_density'
        SKIN_DENSITY = 'aircraft:DBFVerticalTail:skin_density'
        AIRFOIL_X_COORDS = 'aircraft:DBFVerticalTail:airfoil_x_coords'
        AIRFOIL_Y_COORDS = 'aircraft:DBFVerticalTail:airfoil_y_coords'
