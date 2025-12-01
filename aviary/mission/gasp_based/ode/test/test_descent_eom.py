import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.descent_eom import DescentRates
from aviary.variable_info.variables import Dynamic


class DescentTestCase(unittest.TestCase):
    """
    These tests compare the output of the climb EOM to the output from GASP. There are some discrepancies.
    These discrepancies were considered to be small enough, given that the difference in calculation methods between the two codes is significant.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', DescentRates(num_nodes=2), promotes=['*'])

        self.prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, np.array([459, 459]), units='kn'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, np.array([452, 452]), units='lbf'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, np.array([7966.927, 7966.927]), units='lbf'
        )  # estimated from GASP values
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.MASS, np.array([147661, 147661]), units='lbm'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.ANGLE_OF_ATTACK, np.array([3.2, 3.2]), units='deg'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.ALTITUDE_RATE],
            np.array([-39.41011217, -39.41011217]),
            tol,
        )  # note: values from GASP are: np.array([-39.75, -39.75])
        assert_near_equal(
            self.prob[Dynamic.Mission.DISTANCE_RATE],
            np.array([773.70165638, 773.70165638]),
            tol,
            # note: these values are finite differenced and lose accuracy. Fd values are:np.array([964.4634921, 964.4634921])
        )
        assert_near_equal(
            self.prob['required_lift'],
            np.array([147444.58096139, 147444.58096139]),
            tol,
            # note: values from GASP are: np.array([146288.8, 146288.8]) (estimated based on GASP values)
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.FLIGHT_PATH_ANGLE],
            np.array([-0.05089311, -0.05089311]),
            tol,
        )  # note: values from GASP are: np.array([-.0513127, -.0513127])

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class DescentTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.gasp_based.ode.descent_eom as descent

        descent.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.descent_eom as descent

        descent.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('group', DescentRates(num_nodes=2), promotes=['*'])
        prob.model.set_input_defaults(Dynamic.Mission.VELOCITY, np.array([459, 459]), units='kn')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, np.array([452, 452]), units='lbf'
        )
        prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, np.array([7966.927, 7966.927]), units='lbf'
        )
        prob.model.set_input_defaults(Dynamic.Vehicle.MASS, np.array([147661, 147661]), units='lbm')
        prob.model.set_input_defaults(
            Dynamic.Vehicle.ANGLE_OF_ATTACK, np.array([3.2, 3.2]), units='deg'
        )
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
