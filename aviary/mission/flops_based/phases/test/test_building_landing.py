import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.variable_info.variables import Mission


class LandingPhaseTest(unittest.TestCase):
    """
    Test landing phase builder
    """

    def test_case1(self):
        landing_options = Landing(
            ref_wing_area=1370.0,  # ft**2
            Cl_max_ldg=3,  # no units
        )

        use_detailed = False
        landing = landing_options.build_phase(use_detailed=use_detailed)

        prob = om.Problem()
        prob.model = landing
        prob.setup(force_alloc_complex=True)
        prob.run_model()
        partial_data = prob.check_partials(
            out_stream=None, method="cs", compact_print=False, excludes=["*atmosphere*"])
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

        tol = 1e-6
        assert_near_equal(
            prob[Mission.Landing.GROUND_DISTANCE], 6331.781, tol
        )
        assert_near_equal(
            prob[Mission.Landing.INITIAL_VELOCITY], 134.9752, tol)


if __name__ == "__main__":
    unittest.main()
