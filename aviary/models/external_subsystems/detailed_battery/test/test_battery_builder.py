import unittest

import aviary.api as av
from aviary.subsystems.test.subsystem_tester import skipIfMissingDependencies

BatteryBuilder = av.TestSubsystemBuilder.import_builder('battery.battery_builder.BatteryBuilder')


@skipIfMissingDependencies(BatteryBuilder)
class TestBattery(av.TestSubsystemBuilder):
    """Test battery builder."""

    def setUp(self):
        self.subsystem_builder = BatteryBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
