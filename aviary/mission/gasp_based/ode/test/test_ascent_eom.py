import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.ascent_eom import AscentEOM
from aviary.variable_info.variables import Aircraft, Dynamic


class AscentEOMTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', AscentEOM(num_nodes=2), promotes=['*'])
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.MASS, val=175400 * np.ones(2), units='lbm'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=22000 * np.ones(2), units='lbf'
        )
        self.prob.model.set_input_defaults(Dynamic.Vehicle.LIFT, val=200 * np.ones(2), units='lbf')
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.DRAG, val=10000 * np.ones(2), units='lbf'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.VELOCITY, val=10 * np.ones(2), units='ft/s'
        )
        self.prob.model.set_input_defaults(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(2), units='rad'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.INCIDENCE, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(2), units='deg'
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        tol = 1e-6
        self.prob.run_model()

        assert_near_equal(
            self.prob[Dynamic.Mission.VELOCITY_RATE],
            np.array([2.202965, 2.202965]),
            tol,
        )
        assert_near_equal(
            self.prob[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE],
            np.array([-3.216328, -3.216328]),
            tol,
        )

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AscentEOMTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.gasp_based.ode.ascent_eom as ascent

        ascent.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.ascent_eom as ascent

        ascent.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('group', AscentEOM(num_nodes=2), promotes=['*'])
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
        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(2), units='deg')
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
