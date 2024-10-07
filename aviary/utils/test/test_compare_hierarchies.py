import unittest

import aviary.api as av
from aviary.utils.compare_hierarchies import compare_hierarchies_to_merge

AviaryAircraft = av.Aircraft
AviaryMission = av.Mission


class Aircraft1(AviaryAircraft):
    ''' Simple variable hierarchy extension for testing. '''

    CG = "aircraft:center_of_gravity"

    class LandingGear(AviaryAircraft.LandingGear):
        SHAPE = 'aircraft:landing_gear:shape'

    class Wing(AviaryAircraft.Wing):
        CHARACTER = 'aircraft:wing:character'

    class Jury:
        MASS = 'aircraft:jury:mass'

    class Engine(AviaryAircraft.Engine):
        COLOR = 'aircraft:engine:color'

    class Fuselage(AviaryAircraft.Fuselage):

        class Nose:
            ECCENTRICITY = 'aircraft:fuselage:nose:eccentricity'


class Aircraft2(AviaryAircraft):
    ''' Simple variable hierarchy extension for testing. '''

    MASS = "aircraft:mass"
    CG = 'aircraft:center_of_gravity'

    class HorizontalTail(AviaryAircraft.HorizontalTail):
        # add variable that is in the core hierarchy and give it the right value
        AREA = 'aircraft:horizontal_tail:area'
        MEAN_AERO_CHORD = "aircraft:horizontal_tail:mean_aerodynamic_chord"

        class Elevator:
            AREA = "aircraft:horizontal_tail:elevator_area_dist"

    class Wing(AviaryAircraft.Wing):

        AERO_CENTER = "aircraft:wing:aerodynamic_center"
        CHORD = "aircraft:wing:chord"
        AIRFOIL_TECHNOLOGY = 'aircraft:wing:airfoil_technology'

        class Flap:
            AREA = "aircraft:wing:flap_area_dist"
            ROOT_CHORD = "aircraft:wing:flap_root_chord_dist"
            SPAN = "aircraft:wing:flap_span_dist"

    class Jury:
        MASS = 'aircraft:jury:mass'

    class Engine(AviaryAircraft.Engine):
        COLOR = 'aircraft:engine:color'

    class Fuselage(AviaryAircraft.Fuselage):

        class Nose:
            ECCENTRICITY = 'aircraft:fuselage:nose:eccentricity'


class Aircraft3(AviaryAircraft):
    ''' Simple variable hierarchy extension for testing. '''

    class HorizontalTail:
        AREA = 'the_wrong_value'  # add variable that is in the core hierarchy and give it the wrong value


class Aircraft_a:
    ''' Simple variable hierarchy for testing. '''

    A1 = 'a1'

    class Subclass:
        A2 = 'a2'

        class Subsubclass:
            A3 = 'a3'

            class Subsubsubclass:
                A4 = 'a4'

                class Subsubsubsubclass:
                    A5 = 'a5'

                    class Sub5:
                        A6 = 'a6'

                        class Sub6:
                            A7 = 'a7'

                            class Sub7:
                                A8 = 'a8'

                                class Sub8:
                                    A9 = 'a9'

                                    class Sub9:
                                        A10 = 'a10'

                                        class Sub10:
                                            A11 = 'a11'


class Aircraft_b:
    ''' Simple variable hierarchy for testing. '''

    A1 = 'a1'

    class Subclass:
        A2 = 'a2'

        class Subsubclass:
            A3 = 'a3'

            class Subsubsubclass:
                A4 = 'a4'

                class Subsubsubsubclass:
                    A5 = 'a5'

                    class Sub5:
                        A6 = 'a6'

                        class Sub6:
                            A7 = 'a7'

                            class Sub7:
                                A8 = 'a8'

                                class Sub8:
                                    A9 = 'a9'

                                    class Sub9:
                                        A10 = 'a10'

                                        class Sub10:
                                            A11 = 'a11'


class Aircraft_c:
    ''' Simple variable hierarchy for testing. '''

    A1 = 'a1'

    class Subclass:
        A2 = 'a2'

        class Subsubclass:
            A3 = 'a3'

            class Subsubsubclass:
                A4 = 'a4'

                class Subsubsubsubclass:
                    A5 = 'a5'

                    class Sub5:
                        A6 = 'a6'

                        class Sub6:
                            A7 = 'a7'

                            class Sub7:
                                A8 = 'a8'

                                class Sub8:
                                    A9 = 'a9'

                                    class Sub9:
                                        A10 = 'a10'

                                        class Sub10:
                                            A11 = 'a1'


class CompareHierarchiesTest(unittest.TestCase):
    """
    Test the functionality of compare_hierarchies_to_merge function.
    """

    def test_compare_successful(self):
        # this shouldn't throw an error
        compare_hierarchies_to_merge([Aircraft1, Aircraft2, Aircraft_a, Aircraft_b])

    def test_compare_error_in_extended(self):
        self.maxDiff = None
        extension_error_msg = "You have attempted to merge two variable hierarchies together that have the same variable with a different string name associated to it. The offending variable is 'A11'. In 'Aircraft_a.Subclass.Subsubclass.Subsubsubclass.Subsubsubsubclass.Sub5.Sub6.Sub7.Sub8.Sub9.Sub10' it has a value of 'a11' and in 'Aircraft_c.Subclass.Subsubclass.Subsubsubclass.Subsubsubsubclass.Sub5.Sub6.Sub7.Sub8.Sub9.Sub10' it has a value of 'a1'."
        with self.assertRaises(ValueError) as cm:
            compare_hierarchies_to_merge([Aircraft_a, Aircraft_c])
        self.assertEqual(str(cm.exception), extension_error_msg)

    def test_compare_error_in_updated(self):
        self.maxDiff = None
        updated_error_msg = "You have attempted to merge two variable hierarchies together that have the same variable with a different string name associated to it. The offending variable is 'AREA'. In 'Aircraft3.HorizontalTail' it has a value of 'the_wrong_value' and in 'Aircraft.HorizontalTail' it has a value of 'aircraft:horizontal_tail:area'."
        with self.assertRaises(ValueError) as cm:
            compare_hierarchies_to_merge([Aircraft3, Aircraft_c])
        self.assertEqual(str(cm.exception), updated_error_msg)


if __name__ == '__main__':
    unittest.main()
