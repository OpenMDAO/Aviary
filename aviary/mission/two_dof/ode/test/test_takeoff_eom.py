import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.two_dof.ode.takeoff_eom import TakeoffEOM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class TakeoffEOMTestCase(unittest.TestCase):
    """Tests for the TakeoffEOM component covering the groundroll, rotation, and ascent phases."""

    def _make_prob(self, ground_roll=False, rotation=False, alpha=None):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            TakeoffEOM(num_nodes=2, ground_roll=ground_roll, rotation=rotation),
            promotes=['*'],
        )
        prob.model.set_input_defaults(Dynamic.Vehicle.MASS, val=175400 * np.ones(2), units='lbm')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=22000 * np.ones(2), units='lbf'
        )
        prob.model.set_input_defaults(Dynamic.Vehicle.LIFT, val=200 * np.ones(2), units='lbf')
        prob.model.set_input_defaults(Dynamic.Vehicle.DRAG, val=10000 * np.ones(2), units='lbf')
        prob.model.set_input_defaults(Dynamic.Mission.VELOCITY, val=10 * np.ones(2), units='ft/s')
        prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(2), units='rad'
        )
        prob.model.set_input_defaults(Aircraft.Wing.INCIDENCE, val=0, units='deg')
        prob.model.set_input_defaults(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.02)
        if not ground_roll:
            prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=alpha, units='deg')

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_ground_roll(self):
        tol = 1e-6
        prob = self._make_prob(ground_roll=True)
        prob.run_model()

        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: (np.array([1.55844194, 1.55844194]), 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: (np.array([0.0, 0.0]), 'rad/s'),
            Dynamic.Mission.ALTITUDE_RATE: (np.array([0.0, 0.0]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([10.0, 10.0]), 'ft/s'),
            'normal_force': (np.array([175200.0, 175200.0]), 'lbf'),
            'fuselage_pitch': (np.array([0.0, 0.0]), 'rad'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_rotation(self):
        tol = 1e-6
        prob = self._make_prob(rotation=True, alpha=np.zeros(2))
        prob.run_model()

        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: (np.array([1.55844194, 1.55844194]), 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: (np.array([0.0, 0.0]), 'rad/s'),
            Dynamic.Mission.ALTITUDE_RATE: (np.array([0.0, 0.0]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([10.0, 10.0]), 'ft/s'),
            'normal_force': (np.array([175200.0, 175200.0]), 'lbf'),
            'fuselage_pitch': (np.array([0.0, 0.0]), 'rad'),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_ascent(self):
        tol = 1e-6
        prob = self._make_prob(alpha=np.zeros(2))
        prob.run_model()

        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: (np.array([2.20118919, 2.20118919]), 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: (
                np.array([-3.21373621, -3.21373621]),
                'rad/s',
            ),
        }
        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
