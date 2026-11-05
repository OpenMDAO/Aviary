import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.wing_common import (
    BWBWingMiscMass,
    WingBendingMass,
    WingMiscMass,
    WingShearControlMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS', 'BWB300FLOPS']


@use_tempdirs
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

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.COMPOSITE_FRACTION,
                Aircraft.Wing.CONTROL_SURFACE_AREA,
                Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER,
                Aircraft.Design.GROSS_MASS,
            ],
            output_keys=Aircraft.Wing.SHEAR_CONTROL_MASS,
            atol=1e-11,
            rtol=1e-11,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)

    def test_bwb(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        aviary_options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        prob = om.Problem()
        prob.model.add_subsystem(
            'wing_sc',
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 874099.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.COMPOSITE_FRACTION, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.CONTROL_SURFACE_AREA, 5513.13877521, units='ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0, units='unitless'
        )

        setup_model_options(prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        with self.subTest(check=Aircraft.Wing.SHEAR_CONTROL_MASS):
            # FLOPS W2 = 38779.214997388881
            assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 38779.21499739, 1e-9)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_alt_gravity(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 0, units='unitless')
        aviary_options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        aviary_options.set_val(Mission.GRAVITY, 36, 'ft/s**2')

        prob = om.Problem()
        prob.model.add_subsystem(
            'wing_sc',
            WingShearControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 874099.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.COMPOSITE_FRACTION, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.CONTROL_SURFACE_AREA, 5513.13877521, units='ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0, units='unitless'
        )

        setup_model_options(prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        with self.subTest(check=Aircraft.Wing.SHEAR_CONTROL_MASS):
            assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 41483.66191532, 1e-9)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
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

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        flops_validation_test(
            self,
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


@use_tempdirs
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

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                Aircraft.Wing.BENDING_MATERIAL_FACTOR,
                Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER,
                Aircraft.Wing.COMPOSITE_FRACTION,
                Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
                Aircraft.Design.GROSS_MASS,
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

    def test_bwb(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 0, units='unitless')
        aviary_options.set_val(Aircraft.Fuselage.NUM_FUSELAGES, val=1, units='unitless')
        prob = om.Problem()
        prob.model.add_subsystem(
            'wing_bending',
            WingBendingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 874099.0, units='lbm')
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

        setup_model_options(prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        with self.subTest(check=Aircraft.Wing.BENDING_MATERIAL_MASS):
            assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 6313.44762977, 1e-9)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_alt_gravity(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Mission.GRAVITY, 35, units='ft/s**2')
        aviary_options.set_val(Aircraft.Fuselage.NUM_FUSELAGES, val=1, units='unitless')

        prob = om.Problem()
        prob.model.add_subsystem(
            'wing_bending',
            WingBendingMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 874099.0, units='lbm')
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

        setup_model_options(prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        with self.subTest(check=Aircraft.Wing.BENDING_MATERIAL_MASS):
            assert_near_equal(
                prob.get_val(Aircraft.Wing.BENDING_MATERIAL_MASS, 'lbm'), 6867.97829171, 1e-9
            )

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class BWBWingMiscMassTest(unittest.TestCase):
    """Tests wing misc mass calculation for BWB."""

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

        with self.subTest(check=Aircraft.Wing.MISC_MASS):
            # In FLOPS, W3 = 21498.833077784657
            assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 21498.83307778, 1e-9)

        with self.subTest(check='partials'):
            partial_data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
