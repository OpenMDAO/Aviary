import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.engine import (
    AdditionalEngineMass,
    EngineMassGroup,
    EnginePodMass,
    TotalEngineMass,
    WingMountEngineMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TotalEngineMassTestCase(unittest.TestCase):
    """Tests for the TotalEngineMass component."""

    def _make_prob(self, num_engines):
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, num_engines)

        prob = om.Problem()
        prob.model.add_subsystem('total_engine', TotalEngineMass(), promotes=['*'])

        setup_model_options(prob, options)
        return prob

    def test_case1(self):
        # large single aisle 1 V3
        prob = self._make_prob(num_engines=[2])

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=0, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606.0,
            Aircraft.Nacelle.MASS: 1018.74,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_bwb(self):
        # GASP BWB model
        prob = self._make_prob(num_engines=[2])

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, 2.5, units='lbm/ft**2')
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, 194.957186763, units='ft**2'
        )  # 6.76*3.14159265*9.18
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, 1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, 0, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 7005.15475443,
            Aircraft.Nacelle.MASS: 487.39296691,
            'pylon_mass': 558.757916785,
        }
        tol = 1e-7

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-10, rtol=1e-12)

    def test_multi_engine(self):
        # arbitrary test case with multiple engine types
        prob = self._make_prob(num_engines=np.array([2, 4]))

        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=[3, 2.45], units='lbm/ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=[339.58, 235.66], units='ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SCALER,
            val=[1.0, 1.0],
            units='unitless',
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=[1.25, 1.28], units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=[0, 0], units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 23405.94,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class EnginePodMassTestCase(unittest.TestCase):
    """Tests for the EnginePodMass component."""

    def _make_prob(self, num_engines):
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, num_engines)

        prob = om.Problem()
        prob.model.add_subsystem('engine_pod', EnginePodMass(), promotes=['*'])

        setup_model_options(prob, options)
        return prob

    def test_case1(self):
        # large single aisle 1 V3
        prob = self._make_prob(num_engines=[2])

        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS, val=1018.74, units='lbm')
        prob.model.set_input_defaults(
            'pylon_mass', val=873.50386333, units='lbm'
        )  # TODO: fill in expected value, not set anywhere in original test file

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Engine.POD_MASS: 1892.24386333,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_bwb(self):
        # GASP BWB model
        prob = self._make_prob(num_engines=[2])

        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS, val=487.39296691, units='lbm')
        prob.model.set_input_defaults('pylon_mass', val=558.757916785, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Engine.POD_MASS: 1046.15088237,
        }
        tol = 1e-7

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-10, rtol=1e-12)

    def test_multi_engine(self):
        # arbitrary test case with multiple engine types
        prob = self._make_prob(num_engines=np.array([2, 4]))

        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=[1, 1], units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS, val=[1018.74, 577.367], units='lbm')
        prob.model.set_input_defaults(
            'pylon_mass', val=[873.50386333, 495.03559317], units='lbm'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Engine.POD_MASS: [1892.24386333, 1072.40259317],
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class AdditionalEngineMassTestCase(unittest.TestCase):
    """Tests for the AdditionalEngineMass component."""

    def _make_prob(self, num_engines, additional_mass_fraction):
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, num_engines)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, additional_mass_fraction)

        prob = om.Problem()
        prob.model.add_subsystem('additional_engine', AdditionalEngineMass(), promotes=['*'])

        setup_model_options(prob, options)
        return prob

    def test_case1(self):
        # large single aisle 1 V3
        prob = self._make_prob(num_engines=[2], additional_mass_fraction=0.14)

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Engine.ADDITIONAL_MASS: 1765.0 / 2,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_bwb(self):
        # GASP BWB model
        prob = self._make_prob(num_engines=[2], additional_mass_fraction=0.04373)

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Engine.ADDITIONAL_MASS: 153.16770871,
        }
        tol = 1e-7

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-10, rtol=1e-12)

    def test_multi_engine(self):
        # arbitrary test case with multiple engine types
        prob = self._make_prob(
            num_engines=np.array([2, 4]), additional_mass_fraction=np.array([0.14, 0.19])
        )

        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Engine.ADDITIONAL_MASS: [882.4158, 513.0],
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class WingMountEngineMassTestCase(unittest.TestCase):
    """Tests for the WingMountEngineMass component."""

    def _make_prob(
        self, num_engines, additional_mass_fraction, has_hybrid_system, total_num_wing_engines
    ):
        options = AviaryValues()
        options.set_val(
            Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=has_hybrid_system, units='unitless'
        )
        options.set_val(Aircraft.Engine.NUM_ENGINES, num_engines)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, additional_mass_fraction)
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, total_num_wing_engines)

        prob = om.Problem()
        prob.model.add_subsystem('wing_engine', WingMountEngineMass(), promotes=['*'])

        setup_model_options(prob, options)
        return prob

    def test_case1(self):
        # large single aisle 1 V3, HAS_HYBRID_SYSTEM = False
        prob = self._make_prob(
            num_engines=[2],
            additional_mass_fraction=0.14,
            has_hybrid_system=False,
            total_num_wing_engines=2,
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS, val=1892.24386333, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.ADDITIONAL_MASS, val=1765.0 / 2, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            'eng_comb_mass': 14370.8,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_hybrid_system(self):
        # HAS_HYBRID_SYSTEM = True
        prob = self._make_prob(
            num_engines=[2],
            additional_mass_fraction=0.14,
            has_hybrid_system=True,
            total_num_wing_engines=2,
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=0, units='lbm')
        prob.model.set_input_defaults('aug_mass', val=0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS, val=1892.24386333, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.ADDITIONAL_MASS, val=1765.0 / 2, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            'eng_comb_mass': 14370.8,
            'prop_mass_sum': 0,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_multi_engine(self):
        # arbitrary test case with multiple engine types
        prob = self._make_prob(
            num_engines=np.array([2, 4]),
            additional_mass_fraction=np.array([0.14, 0.19]),
            has_hybrid_system=False,
            total_num_wing_engines=6,
        )

        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=[1, 0.9], units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=[0.35, 0.0, 0.1], units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=[0, 0], units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Engine.POD_MASS, val=[1892.24386333, 1072.40259317], units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.ADDITIONAL_MASS, val=[882.4158, 513.0], units='lbm'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            'eng_comb_mass': 26142.7716,
            'wing_mounted_mass': 41417.49593562,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_bwb(self):
        # GASP BWB model
        prob = self._make_prob(
            num_engines=[2],
            additional_mass_fraction=0.04373,
            has_hybrid_system=False,
            total_num_wing_engines=0,
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, 6630.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS, 1046.15088237, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.ADDITIONAL_MASS, 153.16770871, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            'eng_comb_mass': 7311.49017184,
            'wing_mounted_mass': 0,
        }
        tol = 1e-7

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-10, rtol=1e-12)


@use_tempdirs
class EngineMassGroupTestCase(unittest.TestCase):
    """Tests for the EngineMassGroup."""

    def _make_prob(
        self, num_engines, additional_mass_fraction, has_hybrid_system, total_num_wing_engines=None
    ):
        options = AviaryValues()
        options.set_val(
            Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=has_hybrid_system, units='unitless'
        )
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, additional_mass_fraction)
        options.set_val(Aircraft.Engine.NUM_ENGINES, num_engines)
        if total_num_wing_engines is not None:
            options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, total_num_wing_engines)

        prob = om.Problem()
        prob.model.add_subsystem('engine', EngineMassGroup(), promotes=['*'])

        setup_model_options(prob, options)
        return prob

    def test_case1(self):
        # large single aisle 1 V3, HAS_HYBRID_SYSTEM = False
        prob = self._make_prob(
            num_engines=[2], additional_mass_fraction=0.14, has_hybrid_system=False
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=0, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606.0,
            Aircraft.Engine.ADDITIONAL_MASS: 1765.0 / 2,
            Aircraft.Engine.POD_MASS: 1892.24386333,
            Aircraft.Nacelle.MASS: 1018.74,
            'eng_comb_mass': 14370.8,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_hybrid_system(self):
        # HAS_HYBRID_SYSTEM = True
        prob = self._make_prob(
            num_engines=[2], additional_mass_fraction=0.14, has_hybrid_system=True
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=0, units='lbm')
        prob.model.set_input_defaults('aug_mass', val=0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606.0,
            Aircraft.Engine.ADDITIONAL_MASS: 1765.0 / 2,
            Aircraft.Engine.POD_MASS: 1892.24386333,
            'eng_comb_mass': 14370.8,
            'prop_mass_sum': 0,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=2e-11, rtol=1e-12)

    def test_multi_engine(self):
        # arbitrary test case with multiple engine types
        prob = self._make_prob(
            num_engines=np.array([2, 4]),
            additional_mass_fraction=np.array([0.14, 0.19]),
            has_hybrid_system=False,
            total_num_wing_engines=6,
        )

        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=[3, 2.45], units='lbm/ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=[339.58, 235.66], units='ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SCALER,
            val=[1.0, 1.0],
            units='unitless',
        )
        prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=[1.25, 1.28], units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=[1, 0.9], units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=[0.35, 0.0, 0.1], units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=[1, 1], units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, val=[0, 0], units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 23405.94,
            Aircraft.Engine.ADDITIONAL_MASS: [882.4158, 513.0],
            Aircraft.Engine.POD_MASS: [1892.24386333, 1072.40259317],
            'eng_comb_mass': 26142.7716,
            'wing_mounted_mass': 41417.49593562,
        }

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-8, rtol=1e-8)

    def test_bwb(self):
        # GASP BWB model
        prob = self._make_prob(
            num_engines=[2], additional_mass_fraction=0.04373, has_hybrid_system=False
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, 2.5, units='lbm/ft**2')
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, 194.957186763, units='ft**2'
        )  # 6.76*3.14159265*9.18
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, 1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, 6630.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.Propeller.MASS, 0, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 7005.15475443,
            Aircraft.Nacelle.MASS: 487.39296691,
            'pylon_mass': 558.757916785,
            Aircraft.Engine.ADDITIONAL_MASS: 153.16770871,
            Aircraft.Engine.POD_MASS: 1046.15088237,
            'eng_comb_mass': 7311.49017184,
            'wing_mounted_mass': 0,
        }
        tol = 1e-7

        with self.subTest(check='value'):
            for var_name, expected_val in expected_values.items():
                with self.subTest(var=var_name):
                    assert_near_equal(prob[var_name], expected_val, tol)

        with self.subTest(check='partials'):
            data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(data, atol=1e-10, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
