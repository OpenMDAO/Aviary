import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.strut import StrutGeom
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class SizeGroupTestCase1(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem("strut", StrutGeom(
            aviary_options=get_option_defaults()), promotes=["*"])

        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=.2, units=None
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=150, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=1.0
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=10.0, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob["strut_y"], 0.5, tol)  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 10.9658561, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.CHORD], 1.36788226, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
