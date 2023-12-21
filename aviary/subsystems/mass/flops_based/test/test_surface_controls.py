import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.surface_controls import (
    AltSurfaceControlMass, SurfaceControlMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class SurfaceCtrlMassTest(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "surf_ctrl",
            SurfaceControlMass(aviary_options=get_flops_inputs(case_name)),
            promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
                        Mission.Design.GROSS_MASS,
                        Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO,
                        Aircraft.Wing.AREA],
            output_keys=[Aircraft.Wing.SURFACE_CONTROL_MASS,
                         Aircraft.Wing.CONTROL_SURFACE_AREA],
            version=Version.TRANSPORT,
            tol=2e-4,
            atol=1e-11,
            rtol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltSurfaceCtrlMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "surf_ctrl",
            AltSurfaceControlMass(aviary_options=get_flops_inputs(case_name)),
            promotes=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
                        Aircraft.Wing.AREA,
                        Aircraft.HorizontalTail.WETTED_AREA,
                        Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                        Aircraft.VerticalTail.AREA],
            output_keys=Aircraft.Wing.SURFACE_CONTROL_MASS,
            version=Version.ALTERNATE,
            atol=1e-11,
            rtol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
