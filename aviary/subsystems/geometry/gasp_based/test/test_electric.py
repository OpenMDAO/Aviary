import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.electric import CableSize
from aviary.variable_info.variables import Aircraft


class ElectricTestCase(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem("cable", CableSize(), promotes=["*"])

        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, 0.35, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, 128, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 10, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Electrical.HYBRID_CABLE_LENGTH], 64.8, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
