import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging.version import Version
import openmdao

from aviary.subsystems.atmosphere.atmosphere import AtmosphereComp

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
    def test_geocentric(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='USatm1976', delta_T_Kelvin=0, num_nodes=7),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [-1000, 0, 10950, 11000, 11100, 20000, 32000], units='m')

        tol = 1e-4
        self.prob.run_model()

        assert_near_equal(self.prob.get_val('temp', units='K'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('pres', units='Pa'), expected_pressure, tol)
        assert_near_equal(self.prob.get_val('rho', units='kg/m**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_geocentric_delta_T(self):

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='USatm1976', delta_T_Kelvin=18, num_nodes=7),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [-1000, 0, 10950, 11000, 11100, 20000, 32000], units='m')

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

        assert_near_equal(self.prob.get_val('temp', units='K'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('pres', units='Pa'), expected_pressure, tol)
        assert_near_equal(self.prob.get_val('rho', units='kg/m**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

    def test_geodetic(self):

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='USatm1976', delta_T_Kelvin=0, num_nodes=7, h_def='geodetic'),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [-1000, 0, 10969, 11019, 11119, 20063, 32162], units='m')

        tol = 1e-4
        self.prob.run_model()

        assert_near_equal(self.prob.get_val('temp', units='K'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('pres', units='Pa'), expected_pressure, tol)
        assert_near_equal(self.prob.get_val('rho', units='kg/m**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)
 
    def test_geodetic_delta_T(self):

        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='USatm1976', delta_T_Kelvin=15, num_nodes=7, h_def='geodetic'),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [-1000, 0, 10969, 11019, 11119, 20063, 32162], units='m')

        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data)


class MILSPEC210AColdTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='cold', delta_T_Kelvin=0, num_nodes=6),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [0, 10000, 35000, 55000, 70000, 100000], units='ft')

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

        assert_near_equal(self.prob.get_val('temp', units='degF'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('rho', units='lbm/ft**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)

        # inHg60 is a newer unit in OpenMDAO so we'll do this check only of that newer version is installed
        if Version(openmdao.__version__) >= Version("3.42.0"):
            assert_near_equal(self.prob.get_val('pres', units='inHg60'), expected_pressure, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class MILSPEC210ATropicalTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='tropical', delta_T_Kelvin=0, num_nodes=6),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [0, 10000, 35000, 55000, 70000, 100000], units='ft')

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

        assert_near_equal(self.prob.get_val('temp', units='degF'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('rho', units='lbm/ft**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)
        if Version(openmdao.__version__) >= Version("3.42.0"):
            assert_near_equal(self.prob.get_val('pres', units='inHg60'), expected_pressure, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class MILSPEC210AHotTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo', AtmosphereComp(data_source='hot', delta_T_Kelvin=0, num_nodes=6), promotes=['*']
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [0, 10000, 35000, 55000, 70000, 100000], units='ft')

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

        assert_near_equal(self.prob.get_val('temp', units='degF'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('rho', units='lbm/ft**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)
        if Version(openmdao.__version__) >= Version("3.42.0"):
            assert_near_equal(self.prob.get_val('pres', units='inHg60'), expected_pressure, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class MILSPEC210APolarTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        self.prob.model.add_subsystem(
            'atmo',
            AtmosphereComp(data_source='polar', delta_T_Kelvin=0, num_nodes=6),
            promotes=['*'],
        )

        self.prob.set_solver_print(level=0)

        self.prob.setup(
            force_alloc_complex=True,
            check=False,
        )
        self.prob.set_val('h', [0, 10000, 35000, 55000, 70000, 100000], units='ft')

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

        assert_near_equal(self.prob.get_val('temp', units='degF'), expected_temp, tol)
        assert_near_equal(self.prob.get_val('rho', units='lbm/ft**3'), expected_density, tol)
        assert_near_equal(self.prob.get_val('sos', units='m/s'), expected_sos, tol)
        assert_near_equal(self.prob.get_val('viscosity', units='Pa*s'), expected_viscosity, tol)
        if Version(openmdao.__version__) >= Version("3.42.0"):
            assert_near_equal(self.prob.get_val('pres', units='inHg60'), expected_pressure, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')

        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
