import unittest
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilderBase
from aviary.utils.aviary_values import AviaryValues

try:
    from aviary.examples.external_subsystems.battery.battery_builder import BatteryBuilder
    from aviary.examples.external_subsystems.battery.battery_variables import Aircraft
    missing_dependecies = False
except ImportError:
    missing_dependecies = True


@unittest.skipIf(missing_dependecies, "Skipping due to missing dependencies")
class TestBattery(TestSubsystemBuilderBase):

    def setUp(self):
        self.subsystem_builder = BatteryBuilder()
        self.aviary_values = AviaryValues()


if __name__ == '__main__':
    unittest.main()
