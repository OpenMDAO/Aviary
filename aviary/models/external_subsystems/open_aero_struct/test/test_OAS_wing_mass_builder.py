import unittest

import aviary.api as av
from aviary.subsystems.test.subsystem_tester import skipIfMissingDependencies

path_to_builder = 'open_aero_struct.OAS_wing_mass_builder.OASWingMassBuilder'
OASWingWeightBuilder = av.TestSubsystemBuilder.import_builder(path_to_builder)


@skipIfMissingDependencies(OASWingWeightBuilder)
class TestStructures(av.TestSubsystemBuilder):
    """Test OAS structure builder."""

    def setUp(self):
        self.subsystem_builder = OASWingWeightBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
