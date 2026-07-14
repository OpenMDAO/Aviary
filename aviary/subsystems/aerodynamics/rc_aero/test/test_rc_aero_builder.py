import unittest

from aviary.subsystems.aerodynamics.rc_aero.rc_aero_builder import RCAeroBuilder
from aviary.variable_info.variables import Aircraft, Dynamic

class TestRCAeroBuilder(unittest.TestCase):
    def test_mission_interface_exposes_expected_outputs_and_parameters(self):
        builder = RCAeroBuilder()

        mission_inputs = builder.mission_inputs()
        self.assertIn(Dynamic.Mission.ALTITUDE, mission_inputs)
        self.assertIn(Dynamic.Mission.VELOCITY, mission_inputs)
        self.assertIn(Aircraft.Wing.SPAN, mission_inputs)
        self.assertIn(Aircraft.HorizontalTail.SPAN, mission_inputs)
        self.assertIn(Aircraft.Fuselage.LENGTH, mission_inputs)
        self.assertIn(Aircraft.VerticalTail.SPAN, mission_inputs)

        mission_outputs = builder.mission_outputs()
        self.assertIn(Dynamic.Vehicle.DRAG, mission_outputs)
        self.assertIn(Dynamic.Vehicle.LIFT, mission_outputs)
        self.assertIn('alpha', mission_outputs)
        self.assertIn('lifting_surface_CD', mission_outputs)
        self.assertIn('avg_CL', mission_outputs)

    def test_needs_mission_solver_accepts_builder_kwargs(self):
        builder = RCAeroBuilder()

        self.assertFalse(builder.needs_mission_solver())


if __name__ == '__main__':
    unittest.main()
