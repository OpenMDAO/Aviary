import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary import constants
from aviary.subsystems.mass.gasp_based.engine import EngineMassGroup
from aviary.variable_info.functions import extract_options, setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class EngineTestCase1(unittest.TestCase):  # this is the large single aisle 1 V3 test case
    """HAS_HYBRID_SYSTEM = False"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606.0,
            # Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 3785.0,
            Aircraft.Engine.ADDITIONAL_MASS: 1765.0 / 2,
            Aircraft.Engine.POD_MASS: 1892.24386333,
            Aircraft.Nacelle.MASS: 1018.74,
            'eng_comb_mass': 14370.8,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=2e-11, rtol=1e-12)


@use_tempdirs
class EngineTestCase2(unittest.TestCase):
    """HAS_HYBRID_SYSTEM = True"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.Propeller.MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'aug_mass', val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606.0,
            # Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 3785.0,
            Aircraft.Engine.ADDITIONAL_MASS: 1765.0 / 2,
            Aircraft.Engine.POD_MASS: 1892.24386333,
            'eng_comb_mass': 14370.8,
            'prop_mass_sum': 0,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=2e-11, rtol=1e-12)


@use_tempdirs
class EngineGroupTestCase1(unittest.TestCase):
    """HAS_HYBRID_SYSTEM = False (large single aisle 1 V3)"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine_group',
            EngineMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.ADDITIONAL_MASS, val=882.4158, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            'eng_comb_mass': 14370.8,
            'wing_mounted_mass': 24446.343040697346,
            'prop_mass_sum': 0.0,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=2e-11, rtol=1e-12)


@use_tempdirs
class EngineGroupTestCase2(unittest.TestCase):
    """
    HAS_HYBRID_SYSTEM = False (large single aisle 1 V3).
    Check between weight and mass conversion
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine_group',
            EngineMassGroup(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.engine as engine

        constants.GRAV_ENGLISH_LBM = 1.1
        engine.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.ADDITIONAL_MASS, val=882.4158, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.engine as engine

        constants.GRAV_ENGLISH_LBM = 1.0
        engine.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            'eng_comb_mass': 14370.8,
            'wing_mounted_mass': 24402.3185636,
            'prop_mass_sum': 0.0,
        }
        tol = 5e-4

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=2e-11, rtol=1e-12)


# arbitrary test case with multiple engine types
@use_tempdirs
class EngineTestCaseMultiEngine(unittest.TestCase):
    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')

        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 6)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, np.array([0.14, 0.19]))

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=[3, 2.45], units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=[339.58, 235.66], units='ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=[1.25, 1.28], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=[1, 0.9], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=[0.35, 0.0, 0.1], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model_options['*'] = extract_options(options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 23405.94,
            # Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 8074.09809932,
            Aircraft.Engine.ADDITIONAL_MASS: [882.4158, 513.0],
            Aircraft.Engine.POD_MASS: [1892.24386333, 1072.40259317],
            'eng_comb_mass': 26142.7716,
            'wing_mounted_mass': 41417.49593562,
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


# arbitrary test case with multiple engine types
@use_tempdirs
class EngineTestCaseMultiEngine2(unittest.TestCase):
    """
    Check between weight and mass conversion
    """

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.engine as engine

        constants.GRAV_ENGLISH_LBM = 1.0
        engine.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')

        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 6)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, np.array([0.14, 0.19]))

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMassGroup(),
            promotes=['*'],
        )

        import aviary.subsystems.mass.gasp_based.engine as engine

        constants.GRAV_ENGLISH_LBM = 1.1
        engine.GRAV_ENGLISH_LBM = 1.1

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=[3, 2.45], units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=[339.58, 235.66], units='ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=[1.25, 1.28], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=[1, 0.9], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=[0.35, 0.0, 0.1], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model_options['*'] = extract_options(options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        tol = 5e-4
        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 23405.94,
            # Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 8074.09809932,
            Aircraft.Engine.ADDITIONAL_MASS: [882.4158, 513.0],
            Aircraft.Engine.POD_MASS: [1870.53906934, 1060.10196575],
            'eng_comb_mass': 26142.7716,
            'wing_mounted_mass': 41325.4946656,
        }

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class BWBEngineTestCase(unittest.TestCase):
    "GASP BWB model"

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.04373)

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, 2.5, units='lbm/ft**2')
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, 194.957186763, units='ft**2'
        )  # 6.76*3.14159265*9.18
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, 1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, 6630.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 7005.15475443,
            Aircraft.Nacelle.MASS: 487.39296691,
            'pylon_mass': 558.757916785,
            # Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 2092.30176475,
            Aircraft.Engine.ADDITIONAL_MASS: 153.16770871,
            Aircraft.Engine.POD_MASS: 1046.15088237,
            'eng_comb_mass': 7311.49017184,
            'wing_mounted_mass': 0,
        }
        tol = 1e-7

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-10, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
