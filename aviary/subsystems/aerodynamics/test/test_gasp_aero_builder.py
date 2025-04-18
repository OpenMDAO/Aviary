import unittest

from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
import aviary.api as av

GASP = LegacyCode.GASP


class TestAeroBuilder(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, GASP)
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')


if __name__ == '__main__':
    unittest.main()
