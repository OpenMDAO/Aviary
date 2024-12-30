import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.variable_info.variables import Aircraft, Mission


class TakeoffPhaseTest(unittest.TestCase):
    """
    Test takeoff phase builder
    """

    def test_case1(self):
        takeoff_options = Takeoff(
            airport_altitude=0,  # ft
            ramp_mass=181200.0,  # lbm
            num_engines=2,  # no units
        )

        use_detailed = False
        takeoff = takeoff_options.build_phase(use_detailed=use_detailed)

        prob = om.Problem()
        prob.model = takeoff
        prob.model.set_input_defaults(
            Aircraft.Wing.AREA, 1370.3, units="ft**2"
        )
        prob.setup(force_alloc_complex=True)
        prob.run_model()
        partial_data = prob.check_partials(
            out_stream=None, method="cs", compact_print=False, excludes=["*atmosphere*"])
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

        tol = 1e-6
        assert_near_equal(
            prob[Mission.Takeoff.GROUND_DISTANCE], 2811.442, tol
        )


if __name__ == "__main__":
    unittest.main()
