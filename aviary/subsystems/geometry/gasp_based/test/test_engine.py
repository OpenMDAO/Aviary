import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.engine import EngineSize
from aviary.variable_info.variables import Aircraft


class TestEngine(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem("engsz", EngineSize(), promotes=["*"])

        self.prob.model.set_input_defaults(
            Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALE_FACTOR, val=1.028233
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.FINENESS, 2, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_large_sinle_aisle_1_defaults(self):
        self.prob.run_model()
        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_DIAMETER], 7.35163, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.70326, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.SURFACE_AREA], 339.58389, tol)

    def test_partials(self):
        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
