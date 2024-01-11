from copy import deepcopy
import os
import unittest

import dymos
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from packaging import version

from aviary.api import Mission
from aviary.interface.default_phase_info.solved import phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.variable_info.enums import EquationsOfMotion


@unittest.skipIf(version.parse(dymos.__version__) <= version.parse("1.8.0"),
                 "Older version of Dymos treats non-time integration variables differently.")
@use_tempdirs
@require_pyoptsparse(optimizer="SNOPT")
class ProblemPhaseTestCase(unittest.TestCase):
    def bench_test_solved_full_mission(self):
        # Build problem
        local_phase_info = deepcopy(phase_info)
        prob = AviaryProblem()

        input_file = 'models/test_aircraft/aircraft_for_bench_GwGm_solved.csv'
        prob.load_inputs(input_file, local_phase_info)
        prob.aviary_inputs.set_val(Mission.Design.RANGE, 2000.0, units="NM")


# Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SNOPT", max_iter=50, use_coloring=True)

        prob.add_design_variables()

        # Load optimization problem formulation
        # Detail which variables the optimizer can control
        prob.add_objective(objective_type="hybrid_objective")

        prob.setup()

        prob.set_initial_guesses()

        prob.run_aviary_problem("dymos_solution.db")


if __name__ == '__main__':
    z = ProblemPhaseTestCase()
    z.bench_test_solved_full_mission()
