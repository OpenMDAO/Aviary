import unittest

import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
import aviary.api as av

FLOPS = LegacyCode.FLOPS


class TestFLOPSMassBuilder(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreMassBuilder(
            'test_core_mass', meta_data=BaseMetaData, code_origin=FLOPS)
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Design.USE_ALT_MASS, False, units='unitless')
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(
            Aircraft.Engine.NUM_WING_ENGINES, [2], units='unitless')


class TestFLOPSMassBuilderAltMass(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreMassBuilder(
            'test_core_mass', meta_data=BaseMetaData, code_origin=FLOPS)
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Design.USE_ALT_MASS, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(
            Aircraft.Engine.NUM_WING_ENGINES, [2], units='unitless')


if __name__ == '__main__':
    unittest.main()
