import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.constants import GRAV_EARTH
from aviary.mission.two_dof.ode.breguet_cruise_eom import ElectricRangeComp, RangeComp
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Dynamic, Mission


@use_tempdirs
class TestRangeComp(unittest.TestCase):
    """Test cruise range and time in the RangeComp component."""

    def setUp(self):
        self.nn = nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem('range_comp', RangeComp(num_nodes=nn), promotes=['*'])

        aviary_options = AviaryValues()
        aviary_options.set_val(Mission.GRAVITY, val=GRAV_EARTH[0], units=GRAV_EARTH[1])
        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('cruise_time_initial', 0.0, units='s')
        self.prob.set_val('cruise_distance_initial', 0.0, units='NM')
        self.prob.set_val('TAS_cruise', 458.8, units='kn')
        self.prob.set_val('mass', np.linspace(171481, 171481 - 10000, nn), units='lbm')
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
            -5870 * np.ones(nn),
            units='lbm/h',
        )

    def test_results(self):
        self.prob.run_model()

        expected_values = {
            'cruise_range': (781.6042667487062, 'NM'),
            'cruise_time': (6132.901831506848, 's'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)[-1, ...]
                assert_near_equal(actual, expected, tolerance=1e-6)

    def test_partials(self):
        tol = 1e-6
        self.prob.run_model()

        partial_data = self.prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partial_data, atol=tol, rtol=tol)

    def test_range_time_consistency(self):
        """Range and time increments must satisfy the underlying kinematic/mass-flow identities."""
        self.prob.run_model()

        gravity = GRAV_EARTH[0]

        W = self.prob.get_val('mass', units='kg') * gravity
        V = self.prob.get_val('TAS_cruise', units='m/s')
        r = self.prob.get_val('cruise_range', units='m')
        t = self.prob.get_val('cruise_time', units='s')
        fuel_flow = -self.prob.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL, units='kg/s'
        )

        v_avg = (V[:-1] + V[1:]) / 2
        fuel_flow_avg = (fuel_flow[:-1] + fuel_flow[1:]) / 2

        with self.subTest(check='range rate matches average speed'):
            # Range increment should equal the average speed in the segment times the change in time
            assert_near_equal(np.diff(r), v_avg * np.diff(t), tolerance=1.0e-5)

        with self.subTest(check='time increment matches mass flow'):
            # Time increment should satisfy: dt = -dW / (gravity * fuel_flow)
            assert_near_equal(np.diff(t), -np.diff(W) / (gravity * fuel_flow_avg), tolerance=1.0e-5)


@use_tempdirs
class TestElectricRangeComp(unittest.TestCase):
    """Test cruise range and time in the ElectricRangeComp component."""

    def setUp(self):
        self.nn = nn = 10

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'electric_range_comp', ElectricRangeComp(num_nodes=nn), promotes=['*']
        )

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_val('cruise_time_initial', 0.0, units='s')
        self.prob.set_val('cruise_distance_initial', 0.0, units='m')
        self.prob.set_val('TAS_cruise', 458.8, units='kn')
        self.prob.set_val(
            Dynamic.Vehicle.CUMULATIVE_ELECTRIC_ENERGY_USED,
            np.linspace(5843, 19390, nn),
            units='kW*h',
        )
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
            7531.64 * np.ones(nn),
            units='kW',
        )  # 10100.1 hp in GASP

    def test_results(self):
        self.prob.run_model()

        expected_values = {
            'cruise_range': (825.2337, 'NM'),
            'cruise_time': (6475.243, 's'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)[-1, ...]
                assert_near_equal(actual, expected, tolerance=1e-6)

    def test_partials(self):
        tol = 1e-10
        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=tol, rtol=tol)


if __name__ == '__main__':
    unittest.main()
