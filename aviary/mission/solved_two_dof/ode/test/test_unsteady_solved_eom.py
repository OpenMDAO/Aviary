import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.solved_two_dof.ode.unsteady_solved_eom import UnsteadySolvedEOM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class TestUnsteadySolvedEOM(unittest.TestCase):
    """unit test for UnsteadySolvedEOM."""

    def _test_unsteady_solved_eom(self, ground_roll=False):
        nn = 5

        p = om.Problem()
        p.model.add_subsystem(
            'eom',
            UnsteadySolvedEOM(num_nodes=nn, ground_roll=ground_roll),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        p.setup(force_alloc_complex=True)

        p.set_val(Dynamic.Mission.VELOCITY, 250, units='kn')
        p.set_val('mass', 175_000, units='lbm')
        p.set_val(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, 20_000, units='lbf')
        p.set_val(Dynamic.Vehicle.LIFT, 175_000, units='lbf')
        p.set_val(Dynamic.Vehicle.DRAG, 20_000, units='lbf')
        p.set_val(Aircraft.Wing.INCIDENCE, 0.0, units='deg')
        p.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.02, units='unitless')

        if not ground_roll:
            p.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, units='deg')
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 0, units='deg')
            p.set_val('dh_dr', 0, units=None)
            p.set_val('d2h_dr2', 0, units='1/m')

        p.run_model()

        # True airspeed in level flight is dr_dt.
        # Normal force, fuselage pitch, and dgam_dt are 0.0, and load factor is 1.0 in balanced
        # level flight with zero alpha and wing incidence.

        # normal_force and dgam_dt/dgam_dt_approx come from a near-cancellation of weight and lift
        # (~175,000 lbf each), so their absolute error is limited by the precision of OpenMDAO's
        # lbm/kg unit conversion rather than floating-point precision.
        expected_values = {
            'dt_dr': (1 / 250.0 * np.ones(nn), 'h/NM', 1.0e-12),
            # when comparing against a value of 0 asserts use abs tol, not rel (so this is within 0.001 N)
            'normal_force': (np.zeros(nn), 'lbf', 1e-3),
            'fuselage_pitch': (np.zeros(nn), 'deg', 1.0e-12),
            'load_factor': (np.ones(nn), 'unitless', 1.0e-8),
        }

        if not ground_roll:
            expected_values['dgam_dt'] = (np.zeros(nn), 'deg/s', 1.0e-6)
            expected_values['dgam_dt_approx'] = (np.zeros(nn), 'deg/s', 1.0e-6)

        for var_name, (expected, units, tolerance) in expected_values.items():
            with self.subTest(var=var_name):
                actual = p.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tolerance)

        partial_data = p.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_unsteady_solved_eom_ground_roll(self):
        self._test_unsteady_solved_eom(ground_roll=True)

    def test_unsteady_solved_eom_no_ground_roll(self):
        self._test_unsteady_solved_eom(ground_roll=False)


if __name__ == '__main__':
    unittest.main()
