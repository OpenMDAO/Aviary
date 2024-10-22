import subprocess
import unittest
from pathlib import Path

from aviary.utils.functions import get_aviary_resource_path
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
import aviary.api as av
from aviary.interface.default_phase_info.height_energy import phase_info, phase_info_parameterization


@use_tempdirs
@require_pyoptsparse
class TestJson(unittest.TestCase):
    def get_file(self, filename):
        filepath = get_aviary_resource_path(filename)
        if not Path(filepath).exists():
            self.skipTest(f"couldn't find {filepath}")
        return filepath

    def setUp(self):
        self.prob = prob = av.AviaryProblem()
        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases(phase_info_parameterization=phase_info_parameterization)
        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()
        prob.add_driver('IPOPT', max_iter=0)
        prob.add_design_variables()

        # Load optimization problem formulation
        # Detail which variables the optimizer can control
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()

    def test_save_json(self):
        self.prob.run_aviary_problem()
        self.prob.save_sizing_to_json()

    def test_alternate(self):
        filepath = self.get_file('interface/test/sizing_problem_for_test.json')
        prob_alternate = self.prob.alternate_mission(
            run_mission=False, json_filename=filepath, phase_info=phase_info)

    def test_fallout(self):
        filepath = self.get_file('interface/test/sizing_problem_for_test.json')
        prob_fallout = self.prob.fallout_mission(
            run_mission=False, json_filename=filepath, phase_info=phase_info)


if __name__ == "__main__":
    unittest.main()
