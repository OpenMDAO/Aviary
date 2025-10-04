import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.surface_controls import (
    AltSurfaceControlMass,
    SurfaceControlMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings


class SurfaceCtrlMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem('surf_ctrl', SurfaceControlMass(), promotes=['*'])

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
                Mission.Design.GROSS_MASS,
                Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO,
                Aircraft.Wing.AREA,
            ],
            output_keys=[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.CONTROL_SURFACE_AREA],
            version=Version.TRANSPORT,
            tol=2e-4,
            atol=1e-11,
            rtol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class SurfaceCtrlMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.surface_controls as surface

        surface.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.surface_controls as surface

        surface.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem('surf_ctrl', SurfaceControlMass(), promotes=['*'])

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Mission.Design.GROSS_MASS, 130000, 'lbm')
        prob.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 1, 'unitless')
        prob.set_val(Aircraft.Wing.AREA, 1000, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AltSurfaceCtrlMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem('surf_ctrl', AltSurfaceControlMass(), promotes=['*'])

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
                Aircraft.Wing.AREA,
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.VerticalTail.AREA,
            ],
            output_keys=Aircraft.Wing.SURFACE_CONTROL_MASS,
            version=Version.ALTERNATE,
            atol=1e-11,
            rtol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltSurfaceCtrlMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.surface_controls as surface

        surface.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.surface_controls as surface

        surface.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem('surf_ctrl', AltSurfaceControlMass(), promotes=['*'])

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.AREA, 1000, 'ft**2')
        prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, 100, 'ft**2')
        prob.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1, 'unitless')
        prob.set_val(Aircraft.VerticalTail.AREA, 100, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBSurfaceCtrlMassTest(unittest.TestCase):
    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        aviary_options.set_val(Mission.Constraints.MAX_MACH, val=0.85, units='unitless')
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('surf_ctrl', SurfaceControlMass(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 874099, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.CONTROL_SURFACE_AREA_RATIO, 0.333, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 16555.972297926455, units='ft**2')

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        prob = self.prob
        prob.run_model()

        tol = 1e-9
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 14152.3734702, tol)
        assert_near_equal(prob[Aircraft.Wing.CONTROL_SURFACE_AREA], 5513.13877521, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
