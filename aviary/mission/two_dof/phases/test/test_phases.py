"""Test for some features when using an external subsystem in the mission."""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.models.missions.two_dof_default import phase_info as two_dof_phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.enums import PhaseType


class DynBuilder(SubsystemBuilder):
    def get_states(self):
        return {
            'x': {
                'rate_source': 'x_dot',
            }
        }

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return om.ExecComp('x_dot = x**2 + x')


@use_tempdirs
class TestTwoDOFPhases(unittest.TestCase):
    def test_breguet_error_message(self):
        local_phase_info = deepcopy(two_dof_phase_info)

        local_phase_info['cruise'] = {
            'subsystem_options': {'aerodynamics': {'method': 'cruise'}},
            'user_options': {
                'phase_builder': PhaseType.BREGUET_RANGE,
                'alt_cruise': (37.5e3, 'ft'),
                'mach_cruise': 10.8,
            },
            'initial_guesses': {
                # [Initial mass, delta mass] for special cruise phase.
                'mass': ([171481.0, -35000], 'lbm'),
                'initial_distance': (200.0e3, 'ft'),
                'initial_time': (1516.0, 's'),
                'altitude': (37.5e3, 'ft'),
                'mach': (0.8, 'unitless'),
            },
        }

        prob = AviaryProblem()

        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
        )
        prob.load_external_subsystems([DynBuilder()])
        prob.check_and_preprocess_inputs()

        with self.assertRaises(AttributeError) as cm:
            prob.build_model()

        err_text = 'The Breguet Cruise phase does not support dynamic variables in its subsystems.'
        self.assertEqual(str(cm.exception), err_text)


if __name__ == '__main__':
    unittest.main()
