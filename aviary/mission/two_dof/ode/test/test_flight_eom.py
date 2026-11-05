import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.two_dof.ode.flight_eom import EOMRates
from aviary.variable_info.variables import Dynamic, Mission


@use_tempdirs
class EOMRatesTestCase(unittest.TestCase):
    """
    These tests compare the output of the EOM to the output from GASP. There are some discrepancies.
    These discrepancies were considered to be small enough, given that the difference in calculation methods between the two codes is significant.
    """

    def _make_prob(self, thrust, drag, mass, alpha, gravity=(9.80665, 'm/s**2')):
        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            EOMRates(num_nodes=2, **{Mission.GRAVITY: gravity}),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Mission.VELOCITY, np.array([459, 459]), units='kn')
        prob.model.set_input_defaults(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, thrust, units='lbf')
        prob.model.set_input_defaults(Dynamic.Vehicle.DRAG, drag, units='lbf')
        prob.model.set_input_defaults(Dynamic.Vehicle.MASS, mass, units='lbm')
        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, alpha, units='deg')

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_descent(self):
        tol = 1e-6
        prob = self._make_prob(
            thrust=np.array([452, 452]),
            drag=np.array([7966.927, 7966.927]),  # estimated from GASP values
            mass=np.array([147661, 147661]),
            alpha=np.array([3.2, 3.2]),
        )
        prob.run_model()

        # note: some values from GASP differ slightly due to calculation method differences
        # GASP values: ALTITUDE_RATE=[-39.75, -39.75], DISTANCE_RATE=[964.4634921, 964.4634921] (fd),
        #              required_lift=[146288.8, 146288.8], FLIGHT_PATH_ANGLE=[-.0513127, -.0513127]
        expected_values = {
            Dynamic.Mission.ALTITUDE_RATE: (np.array([-39.42713005, -39.42713005]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([773.70078935, 773.70078935]), 'ft/s'),
            'required_lift': (np.array([147444.41570307, 147444.41570307]), 'lbf'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: (np.array([-0.0509151, -0.0509151]), 'rad'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_climb(self):
        tol = 1e-6
        prob = self._make_prob(
            thrust=np.array([10473, 10473]),
            drag=np.array([9091.517, 9091.517]),
            mass=np.array([171481, 171481]),
            alpha=np.degrees(np.ones(2)),
        )
        prob.run_model()

        # note: some values from GASP differ slightly due to calculation method differences
        # GASP values: ALTITUDE_RATE=[5.9667, 5.9667], DISTANCE_RATE=[799.489, 799.489] (fd),
        #              required_lift=[170316.2, 170316.2], FLIGHT_PATH_ANGLE=[.0076794487, .0076794487]
        expected_values = {
            Dynamic.Mission.ALTITUDE_RATE: (np.array([6.24116612, 6.24116612]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([774.679584, 774.679584]), 'ft/s'),
            'required_lift': (np.array([162662.70954313, 162662.70954313]), 'lbf'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: (np.array([0.00805627, 0.00805627]), 'rad'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_alt_gravity(self):
        tol = 1e-6
        prob = self._make_prob(
            thrust=np.array([452, 452]),
            drag=np.array([7966.927, 7966.927]),
            mass=np.array([147661, 147661]),
            alpha=np.array([3.2, 3.2]),
            gravity=(10, 'm/s**2'),
        )
        prob.run_model()

        expected_values = {
            Dynamic.Mission.ALTITUDE_RATE: (np.array([-38.66480653, -38.66480653]), 'ft/s'),
            Dynamic.Mission.DISTANCE_RATE: (np.array([773.73926019, 773.73926019]), 'ft/s'),
            'required_lift': (np.array([150359.43573899, 150359.43573899]), 'lbf'),
            Dynamic.Mission.FLIGHT_PATH_ANGLE: (np.array([-0.04992983, -0.04992983]), 'rad'),
        }

        for var_name, (expected, units) in expected_values.items():
            with self.subTest(var=var_name):
                actual = prob.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
