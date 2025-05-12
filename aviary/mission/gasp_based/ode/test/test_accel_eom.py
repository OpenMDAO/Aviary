import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.accel_eom import AccelerationRates
from aviary.variable_info.variables import Dynamic


class AccelerationTestCase(unittest.TestCase):
    """
    These tests compare the output of the accel EOM to the output from GASP. There are some discrepancies.
    These discrepancies were considered to be small enough, given that the difference in calculation methods between the two codes is significant.
    Note that the discrepancies occur in values that had to be finite differenced.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', AccelerationRates(num_nodes=2), promotes=['*'])

        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.MASS, np.array([174878, 174878]), units='lbm'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, np.array([2635.225, 2635.225]), units='lbf'
        )  # note: this input value is not provided in the GASP data, so an estimation was made based on another similar data point
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            np.array([32589, 32589]),
            units='lbf',
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, np.array([252, 252]), units='kn'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE],
            np.array([5.51533958, 5.51533958]),
            tol,
            # note: this was finite differenced from GASP. The fd value is: np.array([5.2353365, 5.2353365])
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE_RATE],
            np.array([425.32808399, 425.32808399]),
            tol,
            # note: this was finite differenced from GASP. The fd value is: np.array([441.6439, 441.6439])
        )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AccelerationTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.gasp_based.ode.accel_eom as accel

        accel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.accel_eom as accel

        accel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('group', AccelerationRates(num_nodes=2), promotes=['*'])
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
