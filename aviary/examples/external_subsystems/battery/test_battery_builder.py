import unittest
import aviary.api as av


BatteryBuilder = av.TestSubsystemBuilderBase.import_builder(
    'battery.battery_builder.BatteryBuilder')


@av.skipIfMissingDependencies(BatteryBuilder)
class TestBattery(av.TestSubsystemBuilderBase):
    """
    Test battery builder
    """

    def setUp(self):
        self.subsystem_builder = BatteryBuilder()
        self.aviary_values = av.AviaryValues()


if __name__ == '__main__':
    unittest.main()
