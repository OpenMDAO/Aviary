
import numpy as np
import unittest

from numpy.testing import assert_almost_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem


@use_tempdirs
class LargeTurbopropFreighterBenchmark(unittest.TestCase):

    def build_and_run_problem(self):
        # Build problem
        prob = AviaryProblem()

        prob.load_inputs(
            "models/large_turboprop_freighter/large_turboprop_freighter.csv", phase_info)

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        prob.add_driver("SLSQP", max_iter=0, verbosity=0)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.set_initial_guesses()
        prob.run_aviary_problem("dymos_solution.db", make_plots=False)
