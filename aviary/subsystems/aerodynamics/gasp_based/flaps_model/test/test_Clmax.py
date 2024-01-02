import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.Cl_max import \
    CLmaxCalculation
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.variables import Aircraft, Dynamic

"""
All data is from validation files using standalone flaps model
"""


class CLmaxCalculationTestCase(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem(model=om.Group())

        self.prob.model.add_subsystem('CLmC', CLmaxCalculation(), promotes=['*'])

        self.prob.setup()

        # initial conditions
        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.25725)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 1.0000)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03512)
        self.prob.set_val("VLAM14", 0.99124)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")  #
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.7, units="degR")

    def test_case(self):

        self.prob.run_model()
        tol = 6e-4
        print()

        reg_data = 2.8155
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.17522
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 157.19864
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(out_stream=None, method="fd")
        assert_check_partials(
            data, atol=6400, rtol=0.007
        )  # large to account for large magnitude value, discrepancies are acceptable

    @skipIfMissingXDSM('flaps_specs/CL_max.json')
    def test_CLmax_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "flaps_specs/CL_max.json")


if __name__ == "__main__":
    unittest.main()
