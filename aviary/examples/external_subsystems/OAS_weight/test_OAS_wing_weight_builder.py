import importlib.util
import unittest

import aviary.api as av

path_to_builder = 'OAS_weight.OAS_wing_weight_builder.OASWingWeightBuilder'
OASWingWeightBuilder = av.TestSubsystemBuilderBase.import_builder(path_to_builder)


@av.skipIfMissingDependencies(OASWingWeightBuilder)
@unittest.skipUnless(importlib.util.find_spec('ambiance'), "'ambiance' is not installed")
class TestStructures(av.TestSubsystemBuilderBase):
    """Test OAS structure builder."""

    def setUp(self):
        self.subsystem_builder = OASWingWeightBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
