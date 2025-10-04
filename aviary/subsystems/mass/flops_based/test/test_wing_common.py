import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_common import (
    BWBWingMiscMass,
    WingBendingMass,
    WingMiscMass,
    WingShearControlMass,
)

from aviary.variable_info.functions import setup_model_options
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission, Settings


class WingShearControlMassTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'wing',
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.COMPOSITE_FRACTION,
                Aircraft.Wing.CONTROL_SURFACE_AREA,
                Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER,
                Mission.Design.GROSS_MASS,
            ],
            output_keys=Aircraft.Wing.SHEAR_CONTROL_MASS,
            atol=1e-11,
            rtol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class WingShearControlMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.wing_common as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.wing_common as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'wing',
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 0.333, 'unitless')
        prob.set_val(Aircraft.Wing.CONTROL_SURFACE_AREA, 400, 'ft**2')
        prob.set_val(Mission.Design.GROSS_MASS, 100000, 'lbm')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class WingMiscMassTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'wing',
            WingMiscMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.COMPOSITE_FRACTION,
                Aircraft.Wing.AREA,
                Aircraft.Wing.MISC_MASS_SCALER,
            ],
            output_keys=Aircraft.Wing.MISC_MASS,
            atol=1e-11,
            rtol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class WingMiscMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.wing_common as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.wing_common as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'wing',
            WingMiscMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 0.333, 'unitless')
        prob.set_val(Aircraft.Wing.AREA, 1000, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingBendingMassTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        opts = {
            Aircraft.Fuselage.NUM_FUSELAGES: 1,
        }

        prob.model.add_subsystem(
            'wing',
            WingBendingMass(**opts),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                Aircraft.Wing.BENDING_MATERIAL_FACTOR,
                Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER,
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
                Aircraft.Wing.VAR_SWEEP_MASS_PENALTY,
            ],
            output_keys=Aircraft.Wing.BENDING_MATERIAL_MASS,
            atol=1e-11,
            rtol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class WingBendingMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.wing_common as wing

        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.wing_common as wing

        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'wing',
            WingBendingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.333, 'unitless')
        prob.set_val(Aircraft.Wing.BENDING_MATERIAL_FACTOR, 10, 'unitless')
        prob.set_val(Aircraft.Wing.COMPOSITE_FRACTION, 0.333, 'unitless')
        prob.set_val(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 1, 'unitless')
        prob.set_val(Mission.Design.GROSS_MASS, 100000, 'lbm')
        prob.set_val(Aircraft.Wing.LOAD_FRACTION, 1, 'unitless')
        prob.set_val(Aircraft.Wing.MISC_MASS, 2000, 'lbm')
        prob.set_val(Aircraft.Wing.SHEAR_CONTROL_MASS, 4000, 'lbm')
        prob.set_val(Aircraft.Wing.SPAN, 100, 'ft')
        prob.set_val(Aircraft.Wing.SWEEP, 20, 'deg')
        prob.set_val(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75, 'unitless')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBWingMiscMassTest(unittest.TestCase):
    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        aviary_options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'wing_misc',
            BWBWingMiscMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.COMPOSITE_FRACTION, 1.0, units='unitless')
        prob.model.set_input_defaults('calculated_wing_area', 9165.7048657769119, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.MISC_MASS_SCALER, 1.0, units='unitless')

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        prob = self.prob
        prob.run_model()
        # In FLOPS, W3 = 21498.833077784657
        assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 21498.83307778, 1e-9)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBShearControlMassTest(unittest.TestCase):
    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        aviary_options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'wing_sc',
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 874099, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.COMPOSITE_FRACTION, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.CONTROL_SURFACE_AREA, 5513.13877521, units='ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0, units='unitless'
        )

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        prob = self.prob
        prob.run_model()
        # FLOPS W2 = 38779.214997388881
        assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 38779.21499739, 1e-9)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBWingBendingMassTest(unittest.TestCase):
    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        aviary_options.set_val(Aircraft.Fuselage.NUM_FUSELAGES, val=1, units='unitless')
        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'wing_bending',
            WingBendingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 874099, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.COMPOSITE_FRACTION, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.BENDING_MATERIAL_FACTOR, 2.68745091, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.LOAD_FRACTION, 0.5311, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MISC_MASS, 21498.83307778, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.MISC_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SHEAR_CONTROL_MASS, 38779.2149974, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 238.080049, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 35.7, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0.0, units='unitless')

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        prob = self.prob
        prob.run_model()
        tol = 1e-9
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 6313.44762977, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
