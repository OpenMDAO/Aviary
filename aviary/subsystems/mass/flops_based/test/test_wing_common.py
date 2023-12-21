import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_common import (
    WingBendingMass, WingMiscMass, WingShearControlMass)
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class WingShearControlMassTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            "wing",
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.COMPOSITE_FRACTION,
                        Aircraft.Wing.CONTROL_SURFACE_AREA,
                        Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER,
                        Mission.Design.GROSS_MASS],
            output_keys=Aircraft.Wing.SHEAR_CONTROL_MASS,
            atol=1e-11,
            rtol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class WingMiscMassTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            "wing",
            WingMiscMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.COMPOSITE_FRACTION,
                        Aircraft.Wing.AREA,
                        Aircraft.Wing.MISC_MASS_SCALER],
            output_keys=Aircraft.Wing.MISC_MASS,
            atol=1e-11,
            rtol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class WingBendingMassTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            "wing",
            WingBendingMass(aviary_options=get_option_defaults()),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                        Aircraft.Wing.BENDING_FACTOR,
                        Aircraft.Wing.BENDING_MASS_SCALER,
                        Aircraft.Wing.COMPOSITE_FRACTION,
                        Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
                        Mission.Design.GROSS_MASS,
                        Aircraft.Wing.LOAD_FRACTION,
                        Aircraft.Wing.MISC_MASS,
                        Aircraft.Wing.MISC_MASS_SCALER,
                        Aircraft.Wing.SHEAR_CONTROL_MASS,
                        Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER,
                        Aircraft.Wing.SPAN,
                        Aircraft.Wing.SWEEP,
                        Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                        Aircraft.Wing.VAR_SWEEP_MASS_PENALTY],
            output_keys=Aircraft.Wing.BENDING_MASS,
            atol=1e-11,
            rtol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
