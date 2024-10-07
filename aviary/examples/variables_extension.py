import aviary.api as av

AviaryAircraft = av.Aircraft
AviaryMission = av.Mission


class Aircraft(AviaryAircraft):
    """Extended Aircraft data hierarchy"""

    CG = "aircraft:center_of_gravity"
    MASS = "aircraft:mass"

    class HorizontalTail(AviaryAircraft.HorizontalTail):
        MEAN_AERO_CHORD = "aircraft:horizontal_tail:mean_aerodynamic_chord"

        class Elevator:
            AREA = "aircraft:horizontal_tail:elevator:area_dist"
            ROOT_CHORD = "aircraft:horizontal_tail:elevator:root_chord_dist"
            SPAN = "aircraft:horizontal_tail:elevator:span_dist"

    class Jury:
        MASS = "aircraft:jury:mass"

    class LandingGear(AviaryAircraft.LandingGear):
        MAIN_GEAR_OLEO_DIAMETER = "aircraft:landing_gear:main_gear_oleo_diameter"

    class VerticalTail(AviaryAircraft.VerticalTail):
        MEAN_AERO_CHORD = "aircraft:vertical_tail:mean_aerodynamic_chord"

        class Rudder:
            AREA = "aircraft:vertical_tail:rudder:area_dist"
            ROOT_CHORD = "aircraft:vertical_tail:rudder:root_chord_dist"
            SPAN = "aircraft:vertical_tail:rudder:span_dist"

    class Wing(AviaryAircraft.Wing):

        AERO_CENTER = "aircraft:wing:aerodynamic_center"
        CHORD = "aircraft:wing:chord"

        class Flap:
            AREA = "aircraft:wing:flap:area_dist"
            ROOT_CHORD = "aircraft:wing:flap:root_chord_dist"
            SPAN = "aircraft:wing:flap:span_dist"

        class Krueger:
            AREA = "aircraft:wing:krueger:area_dist"
            ROOT_CHORD = "aircraft:wing:krueger:root_chord_dist"
            SPAN = "aircraft:wing:krueger:span_dist"


class Mission(AviaryMission):
    """Extended Mission data hierarchy"""

    class Cruise:
        FUEL_MASS = "mission:cruise:fuel_mass"
        MACH = "mission:cruise:mach"
        MASS = "mission:cruise:mass"

    # note that we do not include a Mission.Taxi object, despite the fact that we will edit
    # the Mission.Taxi.DURATION variable in the meta_data_extension example. We do not need
    # to add that variable here because it already exists in the AviaryMission variable
    # hierarchy that is provided in Aviary core.
