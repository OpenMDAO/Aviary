import unittest
import os

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from packaging import version

from aviary.mission.gasp_based.phases.landing_group import LandingSegment
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.IO_test_util import (assert_match_spec,
                                                  check_prob_outputs,
                                                  skipIfMissingXDSM)
from aviary.variable_info.variables import Dynamic, Mission


class DLandTestCase(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model = LandingSegment(aviary_options=get_option_defaults())

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"), "Skipping due to OpenMDAO version being too low (<3.26)")
    def test_dland(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val(Mission.Landing.AIRPORT_ALTITUDE, 0, units="ft")
        self.prob.set_val(Mission.Landing.INITIAL_MACH, 0.1, units="unitless")
        self.prob.set_val("alpha", 0, units="deg")  # doesn't matter
        self.prob.set_val(Mission.Landing.MAXIMUM_SINK_RATE, 900, units="ft/min")
        self.prob.set_val(Mission.Landing.GLIDE_TO_STALL_RATIO, 1.3, units="unitless")
        self.prob.set_val(Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
                          1.15, units="unitless")
        self.prob.set_val(Mission.Landing.TOUCHDOWN_SINK_RATE, 5, units="ft/s")
        self.prob.set_val(Mission.Landing.BRAKING_DELAY, 1, units="s")
        self.prob.set_val("mass", 165279, units="lbm")
        self.prob.set_val(Dynamic.Mission.THROTTLE, 0.0, units='unitless')

        self.prob.run_model()

        testvals = {
            Mission.Landing.INITIAL_VELOCITY: 142.74 * 1.68781,
            "TAS_touchdown": 126.27 * 1.68781,
            "theta": np.deg2rad(3.57),
            "flare_alt": 20.8,
            "ground_roll_distance": 1798,
            Mission.Landing.GROUND_DISTANCE: 2980,
            "CL_max": 2.9533,
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-2)

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*USatm*", "*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)

    @skipIfMissingXDSM('statics_specs/landing.json')
    def test_dland_spec(self):
        subsystem = self.prob.model
        assert_match_spec(subsystem, "statics_specs/landing.json")


if __name__ == "__main__":
    unittest.main()
