import unittest
from copy import deepcopy

from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

import aviary.api as av
from aviary.interface.methods_for_level2 import reload_aviary_problem
from aviary.models.missions.height_energy_default import phase_info, phase_info_parameterization
from aviary.utils.functions import get_path


@use_tempdirs
class TestSizingResults(unittest.TestCase):
    """
    These tests just check that the json files for the sizing mission results can be saved or loaded
    and used to run an off-design problem without error. These tests don't check that the off-design
    mission ran correctly.
    """

    @require_pyoptsparse(optimizer='SLSQP')
    def test_save_json(self):
        local_phase_info = deepcopy(phase_info)

        prob = av.AviaryProblem()
        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', local_phase_info
        )

        # Preprocess inputs
        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases(phase_info_parameterization=phase_info_parameterization)
        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()
        prob.add_driver('SLSQP', max_iter=0)
        prob.add_design_variables()

        # Load optimization problem formulation
        # Detail which variables the optimizer can control
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()
        prob.run_aviary_problem()
        prob.save_results()

        self.compare_files(
            'sizing_results.json',
            'interface/test/sizing_results_for_test.json',
        )

    @require_pyoptsparse(optimizer='IPOPT')
    def test_alternate(self):
        local_phase_info = deepcopy(phase_info)

        prob = reload_aviary_problem('interface/test/sizing_results_for_test.json')
        prob.run_off_design_mission(problem_type='alternate', phase_info=local_phase_info)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_fallout(self):
        local_phase_info = deepcopy(phase_info)

        prob = reload_aviary_problem('interface/test/sizing_results_for_test.json')
        prob.run_off_design_mission(problem_type='fallout', phase_info=local_phase_info)

    def compare_files(self, test_file, validation_file):
        """
        Compares the specified file with a validation file.

        Use the `skip_list` input to specify strings that are in lines you want to skip. This is
        useful for skipping lines that are expected to differ (such as timestamps)
        """
        test_file = get_path(test_file)

        validation_file = get_path(validation_file)

        # Open the converted and validation files
        with open(test_file, 'r') as f_in, open(validation_file, 'r') as expected:
            for line in f_in:
                # Remove whitespace and compare
                expected_line = ''.join(expected.readline().split())
                line_no_whitespace = ''.join(line.split())

                # Assert that the lines are equal
                try:
                    self.assertEqual(line_no_whitespace.count(expected_line), 1)

                except Exception:
                    exc_string = f'Error: {test_file}\nFound: {line_no_whitespace}\nExpected: {expected_line}'
                    raise Exception(exc_string)


if __name__ == '__main__':
    unittest.main()

    # test = TestSizingResults()
    # test.test_save_json()
    # test.test_fallout()
