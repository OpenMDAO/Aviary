import unittest
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variable_meta_data import (
    ExtendedMetaData,
)
import aviary.api as av

DBFMassBuilder = av.TestSubsystemBuilderBase.import_builder(
    'dbf_based_mass.dbf_mass_builder.DBFMassBuilder'
)


@av.skipIfMissingDependencies(DBFMassBuilder)
class TestDBFMass(av.TestSubsystemBuilderBase):
    """Test mass builder."""

    def setUp(self):
        self.subsystem_builder = DBFMassBuilder(meta_data=ExtendedMetaData)
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
