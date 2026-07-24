import unittest
import aviary.api as av
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

class TestMassBuilder(av.TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = MassBuilder()
        self.aviary_values = av.AviaryValues()

        self.aviary_values.set_val(Aircraft.Wing.SPAN, 2.0, units="m")
        self.aviary_values.set_val(Aircraft.Wing.ROOT_CHORD, 0.2, units="m")
        self.aviary_values.set_val(Aircraft.Wing.WETTED_AREA, 1.0, units="m**2")

        self.aviary_values.set_val(Aircraft.Fuselage.LENGTH, 1.2, units="m")
        self.aviary_values.set_val(Aircraft.Fuselage.AVG_HEIGHT, 0.12, units="m")
        self.aviary_values.set_val(Aircraft.Fuselage.AVG_WIDTH, 0.10, units="m")
        self.aviary_values.set_val(Aircraft.Fuselage.WETTED_AREA, 0.58, units="m**2")

        self.aviary_values.set_val(Aircraft.HorizontalTail.SPAN, 1.0, units="m")
        self.aviary_values.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.2, units="m")
        self.aviary_values.set_val(Aircraft.HorizontalTail.WETTED_AREA, 0.5, units="m**2")

        self.aviary_values.set_val(Aircraft.VerticalTail.SPAN, 0.5, units="m")
        self.aviary_values.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.2, units="m")
        self.aviary_values.set_val(Aircraft.VerticalTail.WETTED_AREA, 0.25, units="m**2")
        
if __name__ == "__main__":
    unittest.main()