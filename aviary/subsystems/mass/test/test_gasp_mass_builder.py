import unittest

import aviary.api as av
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft

GASP = LegacyCode.GASP


class TestGASPMassBuilderHybrid(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreMassBuilder(
            'test_core_mass', meta_data=BaseMetaData, code_origin=GASP
        )
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, 3, units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        self.aviary_values.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_FOLD, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_STRUT, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 1)


class TestGASPMassBuilder(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreMassBuilder(
            'test_core_mass', meta_data=BaseMetaData, code_origin=GASP
        )
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, 3, units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        self.aviary_values.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, False, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_FOLD, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_STRUT, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 1)


if __name__ == '__main__':
    unittest.main()
