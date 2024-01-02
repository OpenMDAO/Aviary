import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.L_and_D_increments import \
    LiftAndDragIncrements
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft

"""
All data is from validation files using standalone flaps model
"""


class LiftAndDragIncrementsTestCase(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem(model=om.Group())

        self.prob.model.add_subsystem('LaDIs', LiftAndDragIncrements(), promotes=['*'])

        self.prob.setup()

        # initial conditions
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.5)
        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 1)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.25725)
        self.prob.set_val("VLAM5", 1.0)
        self.prob.set_val("VLAM6", 1.0)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444)
        self.prob.set_val("VLAM13", 1.03512)
        self.prob.set_val("VLAM14", 0.99124)

    def test_case(self):

        self.prob.run_model()
        tol = 5e-4
        print()

        reg_data = 0.0650
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.0293
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method="fd")
        assert_check_partials(data, atol=1e-4, rtol=1e-4)

    @skipIfMissingXDSM('flaps_specs/increments.json')
    def test_increment_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "flaps_specs/increments.json")


if __name__ == "__main__":
    unittest.main()
