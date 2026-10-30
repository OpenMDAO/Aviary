import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.control import (
    ControlMassGroup,
    MiscControlMass,
    SumControlMass,
    SurfaceControlMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class MiscControlMassTestCase(unittest.TestCase):
    """Tests for the MiscControlMass component."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('misc_control', MiscControlMass(), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

    def _set_inputs(self, gross_mass):
        self.prob.set_val(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.set_val(Aircraft.Design.GROSS_MASS, val=gross_mass, units='lbm')
        self.prob.set_val(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )
        self.prob.set_val(Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless')
        self.prob.set_val(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless')

    def test_case_1(self):
        # this is the large single aisle 1 V3 test case
        self._set_inputs(gross_mass=175400)
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.COCKPIT_CONTROL_MASS: (137.25749725, 'lbm'),
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: (0.0, 'lbm'),
        }
        tol = 5e-4

        with self.subTest(check='values'):
            for var_name, (expected, units) in expected_values.items():
                with self.subTest(var=var_name):
                    actual = self.prob.get_val(var_name, units=units)
                    assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-11, rtol=1e-12)

    def test_BWB(self):
        # GASP BWB model
        self._set_inputs(gross_mass=150000)
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.COCKPIT_CONTROL_MASS: (128.73047164, 'lbm'),
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: (0.0, 'lbm'),
        }
        tol = 1e-7

        with self.subTest(check='values'):
            for var_name, (expected, units) in expected_values.items():
                with self.subTest(var=var_name):
                    actual = self.prob.get_val(var_name, units=units)
                    assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-11, rtol=1e-12)


@use_tempdirs
class SurfaceControlMassTestCase(unittest.TestCase):
    """Tests for the SurfaceControlMass component."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('surface_control', SurfaceControlMass(), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

    def _set_inputs(
        self, coefficient, area, gross_mass, ultimate_load_factor, min_dive_vel, cockpit_mass
    ):
        self.prob.set_val(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=coefficient, units='unitless'
        )
        self.prob.set_val(Aircraft.Wing.AREA, val=area, units='ft**2')
        self.prob.set_val(Aircraft.Design.GROSS_MASS, val=gross_mass, units='lbm')
        self.prob.set_val(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=ultimate_load_factor, units='unitless'
        )
        self.prob.set_val('min_dive_vel', val=min_dive_vel, units='kn')
        self.prob.set_val(Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless')
        self.prob.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless')
        self.prob.set_val(Aircraft.Controls.COCKPIT_CONTROL_MASS, val=cockpit_mass, units='lbm')

    def test_case_1(self):
        # this is the large single aisle 1 V3 test case
        self._set_inputs(
            coefficient=0.95,
            area=1392.1,
            gross_mass=175400,
            ultimate_load_factor=3.951,
            min_dive_vel=420,
            cockpit_mass=137.25749725,
        )
        self.prob.run_model()

        with self.subTest(check='value'):
            assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 3807.92115815, 5e-4)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-11, rtol=1e-12)

    def test_BWB(self):
        # GASP BWB model
        self._set_inputs(
            coefficient=0.5,
            area=2142.85714286,
            gross_mass=150000,
            ultimate_load_factor=3.97744787,
            min_dive_vel=420,
            cockpit_mass=128.73047164,
        )
        self.prob.run_model()

        assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 2045.5556421, 1e-7)

    def test_alt_gravity(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'surface_control',
            SurfaceControlMass(**{Mission.GRAVITY: (25, 'ft/s**2')}),
            promotes=['*'],
        )

        self.prob.setup(check=False, force_alloc_complex=True)

        self._set_inputs(
            coefficient=0.95,
            area=1392.1,
            gross_mass=175400,
            ultimate_load_factor=3.951,
            min_dive_vel=420,
            cockpit_mass=137.25749725,
        )
        self.prob.run_model()

        with self.subTest(check='value'):
            assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 3252.0277469, 1e-10)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-11, rtol=1e-12)


@use_tempdirs
class SumControlMassTestCase(unittest.TestCase):
    """Tests for the SumControlMass component."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('sum_control', SumControlMass(), promotes=['*'])

        self.prob.setup(check=False, force_alloc_complex=True)

    def _set_inputs(self, cockpit_mass, stab_mass, surface_mass):
        self.prob.set_val(Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm')
        self.prob.set_val(Aircraft.Controls.COCKPIT_CONTROL_MASS, val=cockpit_mass, units='lbm')
        self.prob.set_val(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=stab_mass, units='lbm'
        )
        self.prob.set_val(Aircraft.Wing.SURFACE_CONTROL_MASS, val=surface_mass, units='lbm')

    def test_case_1(self):
        # this is the large single aisle 1 V3 test case
        self._set_inputs(cockpit_mass=137.25749725, stab_mass=0.0, surface_mass=3807.92115815)
        self.prob.run_model()

        with self.subTest(check='value'):
            assert_near_equal(self.prob[Aircraft.Controls.MASS], 3945.0, 5e-4)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-11, rtol=1e-12)

    def test_BWB(self):
        # GASP BWB model
        self._set_inputs(
            cockpit_mass=128.73047164,
            stab_mass=0.0,
            surface_mass=2045.5556421,
        )
        self.prob.run_model()

        assert_near_equal(self.prob[Aircraft.Controls.MASS], 2174.28611375, 1e-7)


@use_tempdirs
class ControlMassGroupTestCase(unittest.TestCase):
    """Tests for the ControlMassGroup group."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('control_group', ControlMassGroup(), promotes=['*'])

    def _set_inputs(self, coefficient, area, gross_mass, ultimate_load_factor, min_dive_vel):
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=coefficient, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=area, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=gross_mass, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=ultimate_load_factor, units='unitless'
        )
        self.prob.model.set_input_defaults('min_dive_vel', val=min_dive_vel, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )

        options = AviaryValues()
        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case_1(self):
        # this is the large single aisle 1 V3 test case
        self._set_inputs(
            coefficient=0.95,
            area=1392.1,
            gross_mass=175400,
            ultimate_load_factor=3.951,
            min_dive_vel=420,
        )
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.COCKPIT_CONTROL_MASS: (137.25749725, 'lbm'),
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: (0.0, 'lbm'),
            Aircraft.Wing.SURFACE_CONTROL_MASS: (3807.92115815, 'lbm'),
            Aircraft.Controls.MASS: (3945.0, 'lbm'),
        }
        tol = 5e-4

        with self.subTest(check='values'):
            for var_name, (expected, units) in expected_values.items():
                with self.subTest(var=var_name):
                    actual = self.prob.get_val(var_name, units=units)
                    assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=3e-11, rtol=1e-12)

    def test_BWB(self):
        # GASP BWB model
        self._set_inputs(
            coefficient=0.5,
            area=2142.85714286,
            gross_mass=150000,
            ultimate_load_factor=3.97744787,
            min_dive_vel=420,
        )
        self.prob.run_model()

        expected_values = {
            Aircraft.Controls.COCKPIT_CONTROL_MASS: (128.73047164, 'lbm'),
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: (0.0, 'lbm'),
            Aircraft.Wing.SURFACE_CONTROL_MASS: (2045.5556421, 'lbm'),
            Aircraft.Controls.MASS: (2174.28611375, 'lbm'),
        }
        tol = 1e-7

        with self.subTest(check='values'):
            for var_name, (expected, units) in expected_values.items():
                with self.subTest(var=var_name):
                    actual = self.prob.get_val(var_name, units=units)
                    assert_near_equal(actual, expected, tol)

        with self.subTest(check='partials'):
            data = self.prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-11, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
