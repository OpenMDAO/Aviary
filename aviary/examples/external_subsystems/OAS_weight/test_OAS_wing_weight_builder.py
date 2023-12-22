import unittest
import aviary.api as av

try:
    from aviary.examples.external_subsystems.OAS_weight.OAS_wing_weight_builder import OASWingWeightBuilder
    missing_dependecies = False
except ImportError:
    missing_dependecies = True

@unittest.skipIf(missing_dependecies, "Skipping due to missing dependencies")
class TestStructures(av.TestSubsystemBuilderBase):

    def setUp(self):
        self.subsystem_builder = OASWingWeightBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
