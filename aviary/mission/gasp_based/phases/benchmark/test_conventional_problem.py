import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse

from aviary.mission.gasp_based.phases.run_phases.run_conventional_problem import \
    run_conventional_problem


@unittest.skip('Currently skipping the new conventional problem setup due to formulation instabilities')
class TestConventionalProblem(unittest.TestCase):

    def assert_result(self, p):
        mass_after_climbs = p.get_val(
            "traj.climb_to_cruise.states:mass", units="lbm")[-1, ...]

        assert_near_equal(mass_after_climbs, 171.329e3, tolerance=1.e-4)

    @require_pyoptsparse(optimizer="IPOPT")
    def test_conventional_problem_ipopt(self):
        p = run_conventional_problem(optimizer="IPOPT")
        self.assert_result(p)

    @require_pyoptsparse(optimizer="SNOPT")
    def test_conventional_problem_snopt(self):
        p = run_conventional_problem(optimizer="SNOPT")
        self.assert_result(p)
