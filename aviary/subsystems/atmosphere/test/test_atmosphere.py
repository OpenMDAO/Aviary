import unittest

import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging.version import Version

from aviary.subsystems.atmosphere.atmosphere import AtmosphereComp
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AtmosphereModel
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Dynamic, Settings

# USATM1976 test values
# Reference values based on altitudes of [-1000, 0, 10950, 11000, 11100, 20000, 32000] #meters
expected_temp = [294.65, 288.15, 216.975, 216.65, 216.65, 216.65, 228.65]  # (deg K)
expected_pressure = [
    113929.1,
    101325,
    22811.08,
    22632.06,
    22277.98,
    5474.889,
    868.0187,
]  # (Pa)
expected_density = [
    1.346995,
    1.224999,
    0.3662468,
    0.3639178,
    0.3582242,
    0.0880348,
    0.013225,
]  # (kg/m**3)
expected_sos = [
    344.07756866,
    340.26121619,
    295.26229189,
    295.04107699,
    295.04107699,
    295.04107699,
    303.1019573,
]  # (m/s)
expected_viscosity = [
    1.82057492e-05,
    1.78938028e-05,
    1.42339868e-05,
    1.42161308e-05,
    1.42161308e-05,
    1.42161308e-05,
    1.48679326e-05,
]  # (Pa*s)


class USatm1976TestCase1(unittest.TestCase):
    def test_geopotential(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=7),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.STANDARD)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-1000, 0, 10950, 11000, 11100, 20000, 32000], units='m'
        )

        tol = 1e-4
        self.prob.run_model()

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)

    def test_geopotential_delta_T(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=18, num_nodes=7),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.STANDARD)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-1000, 0, 10950, 11000, 11100, 20000, 32000], units='m'
        )

        tol = 1e-4
        self.prob.run_model()

        # USATM1976 test values
        # Reference values based on deltaT of 18deg K and altitudes of [-1000, 0, 10950, 11000, 11100, 20000, 32000] #meters
        expected_temp = [312.65, 306.15, 234.975, 234.65, 234.65, 234.65, 246.65]  # (deg K)
        expected_pressure = [
            113929.1,
            101325,
            22811.08,
            22632.06,
            22277.98,
            5474.889,
            868.0187,
        ]  # (Pa)
        expected_density = [
            1.26945945,
            1.15298865,
            0.33819588,
            0.33600664,
            0.33074973,
            0.08128286,
            0.01226004,
        ]  # (kg/m**3)
        expected_sos = [
            354.4315341,
            350.72786367,
            307.26561819,
            307.05305116,
            307.05305116,
            307.05305116,
            314.80650506,
        ]  # (m/s)
        expected_viscosity = [
            1.90525660e-05,
            1.87495902e-05,
            1.52054443e-05,
            1.51882008e-05,
            1.51882008e-05,
            1.51882008e-05,
            1.58179488e-05,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)

    def test_geometric(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=7, h_def='geometric'),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.STANDARD)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-1000, 0, 10969, 11019, 11119, 20063, 32162], units='m'
        )

        tol = 1e-4
        self.prob.run_model()

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)

    def test_geometric_delta_T(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=15, num_nodes=7, h_def='geometric'),
            promotes=['*'],
        )

        # creating an empty aviary value and setting the atmosphere model option
        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.STANDARD)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-1000, 0, 10969, 11019, 11119, 20063, 32162], units='m'
        )

        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MILSPEC210AColdTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=6),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.COLD)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [0, 10000, 35000, 55000, 70000, 100000], units='ft'
        )

    def test_case1(self):
        tol = 1e-4
        self.prob.run_model()

        # MILSPEC210A Cold test values
        # Reference values based on altitudes of [0, 10000, 35000, 55000, 70000, 100000] ft
        expected_temp = [-60.0, -15.0, -85.0, -125.0, -100.5, -103.9]  # (degF)
        expected_pressure = [29.9, 20.6, 7.0, 2.7, 1.3, 0.32]  # (inHg60)
        expected_density = [
            0.09941773,
            0.06145239,
            0.02509574,
            0.01061743,
            0.0048261,
            0.00128696,
        ]  # (lbm/ft**3)
        expected_sos = [
            298.68792148,
            315.05458237,
            289.19537664,
            273.32243258,
            283.15022612,
            281.80685121,
        ]  # (m/s)
        expected_viscosity = [
            1.45107303e-05,
            1.58381230e-05,
            1.37449935e-05,
            1.24738805e-05,
            1.32593665e-05,
            1.31516893e-05,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'degF'): expected_temp,
            (Dynamic.Atmosphere.DENSITY, 'lbm/ft**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        # inHg60 is a newer unit in OpenMDAO so we'll do this check only of that newer version is installed
        if Version(openmdao.__version__) >= Version('3.42.0'):
            with self.subTest(var=Dynamic.Atmosphere.STATIC_PRESSURE):
                assert_near_equal(
                    self.prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, units='inHg60'),
                    expected_pressure,
                    tol,
                )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MILSPEC210ATropicalTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=6),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.TROPICAL)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [0, 10000, 35000, 55000, 70000, 100000], units='ft'
        )

    def test_case1(self):
        tol = 1e-4
        self.prob.run_model()

        # MILSPEC210A Tropical test values
        # Reference values based on altitudes of [0, 10000, 35000, 55000, 70000, 100000] ft
        expected_temp = [89.8, 51.0, -45.6, -109.0, -75.5, -33.5]  # (degF)
        expected_pressure = [29.92, 20.58, 7.04, 2.69, 1.31, 0.32]  # (inHg60)
        expected_density = [
            0.07226285,
            0.05347322,
            0.02256043,
            0.01018951,
            0.00452367,
            0.00099482,
        ]  # (lbm/ft**3)
        expected_sos = [
            350.21833448,
            337.62691318,
            304.02112982,
            279.77969483,
            292.83879375,
            308.43121361,
        ]  # (m/s)
        expected_viscosity = [
            1.87079134e-05,
            1.76785535e-05,
            1.49423963e-05,
            1.29893806e-05,
            1.40384719e-05,
            1.53000203e-05,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'degF'): expected_temp,
            (Dynamic.Atmosphere.DENSITY, 'lbm/ft**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        if Version(openmdao.__version__) >= Version('3.42.0'):
            with self.subTest(var=Dynamic.Atmosphere.STATIC_PRESSURE):
                assert_near_equal(
                    self.prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, units='inHg60'),
                    expected_pressure,
                    tol,
                )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MILSPEC210AHotTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=6),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.HOT)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [0, 10000, 35000, 55000, 70000, 100000], units='ft'
        )

    def test_case1(self):
        tol = 1e-4
        self.prob.run_model()

        # MILSPEC210A Hot test values
        # Reference values based on altitudes of [0, 10000, 35000, 55000, 70000, 100000] ft
        expected_temp = [103.0, 63.9, -30.1, -39.1, -34.7, -11.6]  # (degF)
        expected_pressure = [29.9, 20.6, 7.0, 2.7, 1.3, 0.32]  # (inHg60)
        expected_density = [
            0.07046111,
            0.05212192,
            0.02187834,
            0.00836525,
            0.00418262,
            0.00096522,
        ]  # (lbm/ft**3)
        expected_sos = [
            354.40004279,
            341.86470266,
            309.65910723,
            306.39807376,
            307.99667062,
            316.25676125,
        ]  # (m/s)
        expected_viscosity = [
            1.90499896e-05,
            1.80248559e-05,
            1.53996922e-05,
            1.51350772e-05,
            1.52647572e-05,
            1.59359066e-05,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'degF'): expected_temp,
            (Dynamic.Atmosphere.DENSITY, 'lbm/ft**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        if Version(openmdao.__version__) >= Version('3.42.0'):
            with self.subTest(var=Dynamic.Atmosphere.STATIC_PRESSURE):
                assert_near_equal(
                    self.prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, units='inHg60'),
                    expected_pressure,
                    tol,
                )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MILSPEC210APolarTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=6),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.POLAR)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [0, 10000, 35000, 55000, 70000, 100000], units='ft'
        )

    def test_case1(self):
        tol = 1e-4
        self.prob.run_model()

        # MILSPEC210A Polar test values
        # Reference values based on altitudes of [0, 10000, 35000, 55000, 70000, 100000] ft
        expected_temp = [-15.7, -9.7, -68.3, -73.5, -77.3, -81.4]  # (degF)
        expected_pressure = [29.92, 20.58, 7.04, 2.69, 1.31, 0.32]  # (inHg60)
        expected_density = [
            0.08941162,
            0.06068021,
            0.02386669,
            0.00925325,
            0.00454619,
            0.00112062,
        ]  # (lbm/ft**3)
        expected_sos = [
            314.80650506,
            316.92658091,
            295.57020417,
            293.6000678,
            292.15195106,
            290.58141562,
        ]  # (m/s)
        expected_viscosity = [
            1.58179488e-05,
            1.59904030e-05,
            1.42588439e-05,
            1.40998624e-05,
            1.39831040e-05,
            1.38565730e-05,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'degF'): expected_temp,
            (Dynamic.Atmosphere.DENSITY, 'lbm/ft**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        if Version(openmdao.__version__) >= Version('3.42.0'):
            with self.subTest(var=Dynamic.Atmosphere.STATIC_PRESSURE):
                assert_near_equal(
                    self.prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, units='inHg60'),
                    expected_pressure,
                    tol,
                )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MarsReference2024TestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=7),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.MARS_REFERENCE)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-5000, 0, 5000, 10000, 30000, 60000, 80000], units='m'
        )

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        # # MarsReference2024 test values
        # # Reference values based on altitudes of [-5000, 0, 5000, 10000, 30000, 60000, 80000] m
        expected_temp = [214.0, 214.0, 212.9, 205.0, 175.0, 144.2, 139.0]  # (K)
        expected_pressure = [
            1.00e03,
            6.36e02,
            4.03e02,
            2.54e02,
            3.28e01,
            8.78e-01,
            6.08e-02,
        ]  # (Pa)
        expected_density = [
            2.45e-02,
            1.55e-02,
            9.90e-03,
            6.47e-03,
            9.80e-04,
            3.18e-05,
            2.29e-06,
        ]  # (kg/m**3)
        expected_sos = [
            236.28995258,
            236.28995258,
            235.68188291,
            231.26786919,
            213.67681067,
            193.96394057,
            190.43456078,
        ]  # (m/s)
        expected_viscosity = [
            1.07917817e-05,
            1.07917817e-05,
            1.07357667e-05,
            1.03314647e-05,
            8.76446774e-06,
            7.10703626e-06,
            6.82297813e-06,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MarsReference2024TempOffset(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=15, num_nodes=7),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.MARS_REFERENCE)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-5000, 0, 5000, 10000, 30000, 60000, 80000], units='m'
        )

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        # # MarsReference2024 test values
        # # Reference values based on altitudes of [-5000, 0, 5000, 10000, 30000, 60000, 80000] m
        expected_temp = [229.0, 229.0, 227.9, 220.0, 190.0, 159.2, 154.0]  # (K)
        expected_pressure = [
            1.00e03,
            6.36e02,
            4.03e02,
            2.54e02,
            3.28e01,
            8.78e-01,
            6.08e-02,
        ]  # (Pa)
        expected_density = [
            2.28864806e-02,
            1.44842219e-02,
            9.24637207e-03,
            6.02814162e-03,
            9.02413378e-04,
            2.87985425e-05,
            2.06607291e-06,
        ]  # (kg/m**3)
        expected_sos = [
            244.43090156,
            244.43090156,
            243.84313376,
            239.57953015,
            222.64613935,
            203.80268445,
            200.44661606,
        ]  # (m/s)
        expected_viscosity = [
            1.15487681e-05,
            1.15487681e-05,
            1.14936899e-05,
            1.10961095e-05,
            9.55415243e-06,
            7.91991798e-06,
            7.63927628e-06,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)


class MarsReference2024Geometric(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=7, h_def='geometric'),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.MARS_REFERENCE)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(
            Dynamic.Mission.ALTITUDE, [-5000, 0, 5000, 10000, 30000, 60000, 80000], units='m'
        )

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        # # MarsReference2024 test values
        # # Reference values based on altitudes of [-5000, 0, 5000, 10000, 30000, 60000, 80000] m
        expected_temp = [
            214.0,
            214.0,
            212.90367517,
            205.06165228,
            175.32748498,
            144.92608122,
            138.76963819,
        ]  # (K)
        expected_pressure = [
            1.00063894e03,
            6.36000000e02,
            4.03263659e02,
            2.54700751e02,
            3.36257797e01,
            9.85028628e-01,
            6.05868760e-02,
        ]  # (Pa)
        expected_density = [
            2.45157229e-02,
            1.55000000e-02,
            9.90639679e-03,
            6.48607867e-03,
            1.00353097e-03,
            3.55496703e-05,
            2.34696016e-06,
        ]  # (kg/m**3)
        expected_sos = [
            236.28995258,
            236.28995258,
            235.68391712,
            231.30264265,
            213.8766485,
            194.45165464,
            190.27669372,
        ]  # (m/s)
        expected_viscosity = [
            1.07917817e-05,
            1.07917817e-05,
            1.07359540e-05,
            1.03346336e-05,
            8.78183710e-06,
            7.14661229e-06,
            6.81036963e-06,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class VenusReference2021TestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=5),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.VENUS_REFERENCE)
        setup_model_options(self.prob, options)

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val(Dynamic.Mission.ALTITUDE, [0, 20000, 60000, 100_000, 140_000], units='m')

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        # # MarsReference2024 test values
        # # Reference values based on altitudes of [0, 20000, 60000, 100_000, 140_000] m
        expected_temp = [735.3, 580.15373493, 260.78698987, 171.79217223, 180.78311551]  # (K)
        expected_pressure = [
            9.21000000e06,
            2.23851198e06,
            2.20277689e04,
            2.17264730e00,
            1.91702893e-05,
        ]  # (Pa)
        expected_density = [
            6.48000000e01,
            2.03136169e01,
            4.38717066e-01,
            6.50102839e-05,
            2.75367723e-10,
        ]  # (kg/m**3)
        expected_sos = [
            426.03533001,
            378.42940837,
            253.72092923,
            205.92781816,
            211.24783454,
        ]  # (m/s) * This estimate is not very accurate.
        expected_viscosity = [
            3.13045720e-05,
            2.61827952e-05,
            1.31108940e-05,
            8.59403184e-06,
            9.07035998e-06,
        ]  # (Pa*s)

        expected_values = {
            (Dynamic.Atmosphere.TEMPERATURE, 'K'): expected_temp,
            (Dynamic.Atmosphere.STATIC_PRESSURE, 'Pa'): expected_pressure,
            (Dynamic.Atmosphere.DENSITY, 'kg/m**3'): expected_density,
            (Dynamic.Atmosphere.SPEED_OF_SOUND, 'm/s'): expected_sos,
            (Dynamic.Atmosphere.DYNAMIC_VISCOSITY, 'Pa*s'): expected_viscosity,
        }

        for (var_name, units), expected in expected_values.items():
            with self.subTest(var=var_name):
                # print(self.prob.get_val(var_name, units=units))
                assert_near_equal(self.prob.get_val(var_name, units=units), expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


if __name__ == '__main__':
    unittest.main()
