import unittest

import aviary.api as av
from aviary.utils.merge_hierarchies import merge_hierarchies

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

    class Test:
        var1 = '1'

        class SubTest:
            var2 = '2'

            class SubSubTest:
                var3 = '3'

                class SubSubSubTest:
                    var4 = '4'


class Aircraft2(AviaryAircraft):
    ''' Simple variable hierarchy extension for testing. '''

    MASS = "aircraft:mass"
    CG = 'aircraft:center_of_gravity'

    class HorizontalTail(AviaryAircraft.HorizontalTail):
        # add variable that is in the core hierarchy and give it the right value
        AREA = 'aircraft:horizontal_tail:area'
        MEAN_AERO_CHORD = "aircraft:horizontal_tail:mean_aerodynamic_chord"
        TEST = 'test'

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

    class Engine(AviaryAircraft.Engine):
        COLOR = 'aircraft:engine:color'


class Aircraft3(AviaryAircraft):
    ''' Simple variable hierarchy extension for testing. '''

    class HorizontalTail(AviaryAircraft.HorizontalTail):

        class Elevator:
            SPAN = "aircraft:horizontal_tail:elevator_span"

    class Wing(AviaryAircraft.Wing):

        class Flap:
            CHORD = "aircraft:wing:flap_chord"


class Aircraft4(AviaryAircraft):
    ''' Simple variable hierarchy extension for testing. '''

    class HorizontalTail(AviaryAircraft.HorizontalTail):

        class Elevator:
            SPAN = "wrong_value"


class SimpleHierarchy:
    class InnerClass:
        var1 = 'dummy'


class SimpleHierarchy2:
    class SubOne:
        var_sub_one = 'dummy1'

    class SubTwo:
        var_sub_two = 'dummy2'

    dummy_var = 'dummy3'


class Mission1(AviaryMission):
    dummy = 'dummy'


class DeepSub:
    class Test:
        class SubTest:
            class Sub2Test:
                class Sub3Test:
                    class Sub4Test:
                        class Sub5Test:
                            class Sub6Test:
                                class Sub7Test:
                                    class Sub8Test:
                                        class Sub9Test:
                                            class Sub10Test:
                                                var10 = 'dummy10'


merge_combo1 = [SimpleHierarchy, Aircraft1, Aircraft2]
merge_combo2 = [Aircraft1, Aircraft2, SimpleHierarchy]
merge_combo3 = [Aircraft1, Aircraft2, SimpleHierarchy, Mission1]
merge_combo4 = [SimpleHierarchy2, SimpleHierarchy]
merge_combo5 = [SimpleHierarchy, DeepSub]
merge_combo6 = [DeepSub, SimpleHierarchy]
merge_combo7 = [Aircraft2, Aircraft3]
merge_combo8 = [Aircraft3, Aircraft4]


class MergeHierarchiesTest(unittest.TestCase):
    """
    Test functionality of merge_hierarchies function.
    """

    def test_merge1(self):
        self.maxDiff = None

        merged = merge_hierarchies(merge_combo1)

        # check for presence of all variables from input hierarchies
        self.assertEqual(merged.InnerClass.var1, 'dummy')
        self.assertEqual(merged.CG, 'aircraft:center_of_gravity')
        self.assertEqual(merged.LandingGear.SHAPE, 'aircraft:landing_gear:shape')
        self.assertEqual(merged.Wing.CHARACTER, 'aircraft:wing:character')
        self.assertEqual(merged.Jury.MASS, 'aircraft:jury:mass')
        self.assertEqual(merged.Engine.COLOR, 'aircraft:engine:color')
        self.assertEqual(merged.Fuselage.Nose.ECCENTRICITY,
                         'aircraft:fuselage:nose:eccentricity')
        self.assertEqual(merged.Test.var1, '1')
        self.assertEqual(merged.Test.SubTest.var2, '2')
        self.assertEqual(merged.Test.SubTest.SubSubTest.var3, '3')
        self.assertEqual(merged.Test.SubTest.SubSubTest.SubSubSubTest.var4, '4')
        self.assertEqual(merged.MASS, 'aircraft:mass')
        self.assertEqual(merged.HorizontalTail.AREA, 'aircraft:horizontal_tail:area')
        self.assertEqual(merged.HorizontalTail.MEAN_AERO_CHORD,
                         'aircraft:horizontal_tail:mean_aerodynamic_chord')
        self.assertEqual(merged.HorizontalTail.TEST, 'test')
        self.assertEqual(merged.HorizontalTail.Elevator.AREA,
                         'aircraft:horizontal_tail:elevator_area_dist')
        self.assertEqual(merged.Wing.AERO_CENTER, 'aircraft:wing:aerodynamic_center')
        self.assertEqual(merged.Wing.CHORD, 'aircraft:wing:chord')
        self.assertEqual(merged.Wing.AIRFOIL_TECHNOLOGY,
                         'aircraft:wing:airfoil_technology')
        self.assertEqual(merged.Wing.Flap.AREA, 'aircraft:wing:flap_area_dist')
        self.assertEqual(merged.Wing.Flap.ROOT_CHORD,
                         'aircraft:wing:flap_root_chord_dist')
        self.assertEqual(merged.Wing.Flap.SPAN, 'aircraft:wing:flap_span_dist')
        self.assertEqual(merged.Engine.COLOR, 'aircraft:engine:color')
        self.assertEqual(merged.MASS, 'aircraft:mass')

        # check for presence of variables from original Aviary core hierarchy
        self.assertEqual(merged.AirConditioning.MASS, 'aircraft:air_conditioning:mass')
        self.assertEqual(merged.AntiIcing.MASS_SCALER, 'aircraft:anti_icing:mass_scaler')
        self.assertEqual(merged.APU.MASS, 'aircraft:apu:mass')
        self.assertEqual(merged.Canard.AREA, 'aircraft:canard:area')
        self.assertEqual(merged.CrewPayload.MISC_CARGO,
                         'aircraft:crew_and_payload:misc_cargo')
        self.assertEqual(merged.Design.EMPTY_MASS, 'aircraft:design:empty_mass')
        self.assertEqual(merged.Wing.BWB_AFTBODY_MASS, 'aircraft:wing:bwb_aft_body_mass')
        self.assertEqual(merged.Hydraulics.MASS, 'aircraft:hydraulics:mass')

    def test_merge2(self):
        self.maxDiff = None

        merged = merge_hierarchies(merge_combo2)

        # check for presence of all variables from input hierarchies
        self.assertEqual(merged.InnerClass.var1, 'dummy')
        self.assertEqual(merged.CG, 'aircraft:center_of_gravity')
        self.assertEqual(merged.LandingGear.SHAPE, 'aircraft:landing_gear:shape')
        self.assertEqual(merged.Wing.CHARACTER, 'aircraft:wing:character')
        self.assertEqual(merged.Jury.MASS, 'aircraft:jury:mass')
        self.assertEqual(merged.Engine.COLOR, 'aircraft:engine:color')
        self.assertEqual(merged.Fuselage.Nose.ECCENTRICITY,
                         'aircraft:fuselage:nose:eccentricity')
        self.assertEqual(merged.Test.var1, '1')
        self.assertEqual(merged.Test.SubTest.var2, '2')
        self.assertEqual(merged.Test.SubTest.SubSubTest.var3, '3')
        self.assertEqual(merged.Test.SubTest.SubSubTest.SubSubSubTest.var4, '4')
        self.assertEqual(merged.MASS, 'aircraft:mass')
        self.assertEqual(merged.HorizontalTail.AREA, 'aircraft:horizontal_tail:area')
        self.assertEqual(merged.HorizontalTail.MEAN_AERO_CHORD,
                         'aircraft:horizontal_tail:mean_aerodynamic_chord')
        self.assertEqual(merged.HorizontalTail.TEST, 'test')
        self.assertEqual(merged.HorizontalTail.Elevator.AREA,
                         'aircraft:horizontal_tail:elevator_area_dist')
        self.assertEqual(merged.Wing.AERO_CENTER, 'aircraft:wing:aerodynamic_center')
        self.assertEqual(merged.Wing.CHORD, 'aircraft:wing:chord')
        self.assertEqual(merged.Wing.AIRFOIL_TECHNOLOGY,
                         'aircraft:wing:airfoil_technology')
        self.assertEqual(merged.Wing.Flap.AREA, 'aircraft:wing:flap_area_dist')
        self.assertEqual(merged.Wing.Flap.ROOT_CHORD,
                         'aircraft:wing:flap_root_chord_dist')
        self.assertEqual(merged.Wing.Flap.SPAN, 'aircraft:wing:flap_span_dist')
        self.assertEqual(merged.Engine.COLOR, 'aircraft:engine:color')
        self.assertEqual(merged.MASS, 'aircraft:mass')

        # check for presence of variables from original Aviary core hierarchy
        self.assertEqual(merged.AirConditioning.MASS, 'aircraft:air_conditioning:mass')
        self.assertEqual(merged.AntiIcing.MASS_SCALER, 'aircraft:anti_icing:mass_scaler')
        self.assertEqual(merged.APU.MASS, 'aircraft:apu:mass')
        self.assertEqual(merged.Canard.AREA, 'aircraft:canard:area')
        self.assertEqual(merged.CrewPayload.MISC_CARGO,
                         'aircraft:crew_and_payload:misc_cargo')
        self.assertEqual(merged.Design.EMPTY_MASS, 'aircraft:design:empty_mass')
        self.assertEqual(merged.Wing.BWB_AFTBODY_MASS, 'aircraft:wing:bwb_aft_body_mass')
        self.assertEqual(merged.Hydraulics.MASS, 'aircraft:hydraulics:mass')

    def test_merge3(self):
        self.maxDiff = None
        combo3_error_msg = "You have attempted to merge together variable hierarchies that subclass from different superclasses. 'Aircraft1' is a subclass of '<class 'aviary.variable_info.variables.Aircraft'>' and 'Mission1' is a subclass of '<class 'aviary.variable_info.variables.Mission'>'."

        with self.assertRaises(ValueError) as cm:
            merged = merge_hierarchies(merge_combo3)
        self.assertEqual(str(cm.exception), combo3_error_msg)

    def test_merge4(self):
        self.maxDiff = None

        merged = merge_hierarchies(merge_combo4)

        self.assertEqual(merged.SubOne.var_sub_one, 'dummy1')
        self.assertEqual(merged.SubTwo.var_sub_two, 'dummy2')
        self.assertEqual(merged.dummy_var, 'dummy3')
        self.assertEqual(merged.InnerClass.var1, 'dummy')

    def test_merge5(self):
        self.maxDiff = None

        merged = merge_hierarchies(merge_combo5)

        self.assertEqual(merged.InnerClass.var1, 'dummy')
        self.assertEqual(
            merged.Test.SubTest.Sub2Test.Sub3Test.Sub4Test.Sub5Test.Sub6Test.Sub7Test.Sub8Test.Sub9Test.Sub10Test.var10, 'dummy10')

    def test_merge6(self):
        self.maxDiff = None

        merged = merge_hierarchies(merge_combo6)

        self.assertEqual(merged.InnerClass.var1, 'dummy')
        self.assertEqual(
            merged.Test.SubTest.Sub2Test.Sub3Test.Sub4Test.Sub5Test.Sub6Test.Sub7Test.Sub8Test.Sub9Test.Sub10Test.var10, 'dummy10')

    def test_merge7(self):
        self.maxDiff = None

        merged = merge_hierarchies(merge_combo7)

        self.assertEqual(merged.MASS, 'aircraft:mass')
        self.assertEqual(merged.CG, 'aircraft:center_of_gravity')
        self.assertEqual(merged.HorizontalTail.AREA, 'aircraft:horizontal_tail:area')
        self.assertEqual(merged.HorizontalTail.MEAN_AERO_CHORD,
                         'aircraft:horizontal_tail:mean_aerodynamic_chord')
        self.assertEqual(merged.HorizontalTail.TEST, 'test')
        self.assertEqual(merged.HorizontalTail.Elevator.AREA,
                         'aircraft:horizontal_tail:elevator_area_dist')
        self.assertEqual(merged.HorizontalTail.Elevator.SPAN,
                         'aircraft:horizontal_tail:elevator_span')
        self.assertEqual(merged.Wing.AERO_CENTER, 'aircraft:wing:aerodynamic_center')
        self.assertEqual(merged.Wing.CHORD, 'aircraft:wing:chord')
        self.assertEqual(merged.Wing.AIRFOIL_TECHNOLOGY,
                         'aircraft:wing:airfoil_technology')
        self.assertEqual(merged.Wing.Flap.CHORD, 'aircraft:wing:flap_chord')
        self.assertEqual(merged.Wing.Flap.ROOT_CHORD,
                         'aircraft:wing:flap_root_chord_dist')
        self.assertEqual(merged.Wing.Flap.AREA, 'aircraft:wing:flap_area_dist')
        self.assertEqual(merged.Wing.Flap.SPAN, 'aircraft:wing:flap_span_dist')
        self.assertEqual(merged.Engine.COLOR, 'aircraft:engine:color')

    def test_merge8(self):
        self.maxDiff = None
        combo8_error_msg = "You have attempted to merge two variable hierarchies together that have the same variable with a different string name associated to it. The offending variable is 'SPAN'. In 'Aircraft3.HorizontalTail.Elevator' it has a value of 'aircraft:horizontal_tail:elevator_span' and in 'Aircraft4.HorizontalTail.Elevator' it has a value of 'wrong_value'."

        with self.assertRaises(ValueError) as cm:
            merged = merge_hierarchies(merge_combo8)
        self.assertEqual(str(cm.exception), combo8_error_msg)


if __name__ == '__main__':
    unittest.main()
