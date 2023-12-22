import unittest
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilderBase
from aviary.examples.external_subsystems.battery.battery_builder import BatteryBuilder
from aviary.examples.external_subsystems.battery.battery_variables import Aircraft
from aviary.utils.aviary_values import AviaryValues


class TestBattery(TestSubsystemBuilderBase):

    def setUp(self):
        self.subsystem_builder = BatteryBuilder()
        self.aviary_values = AviaryValues()


if __name__ == '__main__':
    unittest.main()
