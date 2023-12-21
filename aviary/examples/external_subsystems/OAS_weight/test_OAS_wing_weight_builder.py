import unittest
import aviary.api as av

from aviary.examples.external_subsystems.OAS_weight.OAS_wing_weight_builder import OASWingWeightBuilder


class TestStructures(av.TestSubsystemBuilderBase):

    def setUp(self):
        self.subsystem_builder = OASWingWeightBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
