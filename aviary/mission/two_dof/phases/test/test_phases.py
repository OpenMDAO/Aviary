"""Test for some features when using an external subsystem in the mission."""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.models.missions.two_dof_default import phase_info as two_dof_phase_info
from aviary.core.aviary_problem import AviaryProblem
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.enums import PhaseType


class DynBuilder(SubsystemBuilder):
    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            'x': {
                'rate_source': 'x_dot',
            }
        }

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        comp = om.ExecComp(
            'x_dot = x**2 + x',
            x_dot={'shape': num_nodes},
            x={'shape': num_nodes},
            has_diag_partials=True,
        )
        return comp


@use_tempdirs
class TestTwoDOFPhases(unittest.TestCase):
    def test_breguet_with_states(self):
        local_phase_info = deepcopy(two_dof_phase_info)

        local_phase_info['cruise'] = {
            'subsystem_options': {'aerodynamics': {'method': 'cruise', 'output_alpha': True}},
            'user_options': {
                'phase_type': PhaseType.BREGUET_RANGE,
                'num_segments': 1,
                'order': 3,
                'alt_cruise': (37.5e3, 'ft'),
                'mach_cruise': 0.8,
                'mass_ref': (171000, 'lbm'),
                'time_duration_ref': (26500, 's'),
            },
            'initial_guesses': {
                # [Initial mass, delta mass] for special cruise phase.
                'mass': ([171481.0, 136000], 'lbm'),
                'initial_distance': (200.0e3, 'ft'),
                'time': ([1504.0, 26500.0], 's'),
                'altitude': (37.5e3, 'ft'),
                'mach': (0.8, 'unitless'),
            }
        }

        prob = AviaryProblem()

        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_GwGm.csv',
            local_phase_info,
        )
        prob.load_external_subsystems([DynBuilder()])
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        # Nonsense component, but make sure it runs.
        prob.run_model()

        self.assertTrue('x' in prob.model.traj.phases.cruise.state_options)


if __name__ == '__main__':
    unittest.main()
