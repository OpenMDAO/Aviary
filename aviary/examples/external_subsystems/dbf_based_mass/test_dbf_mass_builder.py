import unittest

import aviary.api as av

DBFMassBuilder = av.TestSubsystemBuilderBase.import_builder(
    'dbf_based_mass.dbf_mass_builder.DBFMassBuilder'
)


@av.skipIfMissingDependencies(DBFMassBuilder)
class TestDBFMass(av.TestSubsystemBuilderBase):
    """Test mass builder."""

    def setUp(self):
        self.subsystem_builder = DBFMassBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
