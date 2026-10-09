import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.solved_two_dof.ode.groundroll_eom import GroundrollEOM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class GroundrollEOMTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            GroundrollEOM(num_nodes=2),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.MASS, val=np.array([175400, 174950]), units='lbm'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=np.array([22000, 23000]), units='lbf'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.LIFT, val=np.array([200, 25000]), units='lbf'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, val=np.array([10000, 11000]), units='lbf'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, val=np.array([10, 130]), units='ft/s'
        )
        # flight path angle should usually be zero - flight path angle rate is hardcoded to always
        # be zero regardless of provided path angle, so we test that here
        self.prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.array([0, 1]), units='deg'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.INCIDENCE, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.array([0, 2]), units='deg'
        )
        self.prob.model.set_input_defaults(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.02)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case_1(self):
        tol = 1e-6
        self.prob.run_model()

        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: (np.array([1.55844194, 1.09418326]), 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: (np.array([0.0, 0.0]), 'deg/s'),  # always zero
            Dynamic.Mission.ALTITUDE_RATE: (np.array([0.0, 2.26881284]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([10.0, 129.98020037]), 'ft/s'),
            'normal_force': (np.array([175200.0, 149147.31138944]), 'lbf'),
            'fuselage_pitch': (np.array([0.0, 3]), 'deg'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_case_alt_gravity(self):
        self.prob.model_options['*'] = {Mission.GRAVITY: (10, 'm/s**2')}

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        expected_values = {
            Dynamic.Mission.VELOCITY_RATE: (np.array([1.54575494, 1.07042531]), 'ft/s**2'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE: (np.array([0.0, 0.0]), 'deg/s'),
            Dynamic.Mission.ALTITUDE_RATE: (np.array([0.0, 2.26881284]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([10.0, 129.98020037]), 'ft/s'),
            'normal_force': (np.array([178658.22356576, 152596.66284625]), 'lbf'),
            'fuselage_pitch': (np.array([0.0, 3]), 'deg'),
        }

        tol = 1e-6

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = self.prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
