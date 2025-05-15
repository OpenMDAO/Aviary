import unittest

import aviary.api as av
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft

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

    def test_build_mission(self):
        kwargs = {'method': 'cruise'}
        return super().test_build_mission(**kwargs)

    def test_get_parameters(self):
        kwargs = {'method': 'cruise'}
        return super().test_get_parameters(**kwargs)

    def test_check_parameters(self):
        kwargs = {'method': 'cruise'}
        return super().test_check_parameters(**kwargs)


if __name__ == '__main__':
    unittest.main()
