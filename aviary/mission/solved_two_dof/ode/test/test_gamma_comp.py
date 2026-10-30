import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.mission.solved_two_dof.ode.gamma_comp import GammaComp
from aviary.mission.solved_two_dof.ode.unsteady_solved_eom import UnsteadySolvedEOM
from aviary.variable_info.variables import Aircraft, Dynamic


@use_tempdirs
class UnsteadySolvedEOMTestCase(unittest.TestCase):
    """Test 2-degrees-of-freedom equations of motion for unsteady flight."""

    def _test_unsteady_flight_eom(self, ground_roll=False):
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
        p.set_val(Dynamic.Vehicle.MASS, 175_000, units='lbm')
        p.set_val(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, 20_000, units='lbf')
        p.set_val(Dynamic.Vehicle.LIFT, 175_000, units='lbf')
        p.set_val(Dynamic.Vehicle.DRAG, 20_000, units='lbf')
        p.set_val(Aircraft.Wing.INCIDENCE, 0.0, units='deg')

        if not ground_roll:
            p.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, units='deg')
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 0, units='deg')
            p.set_val('dh_dr', 0, units=None)
            p.set_val('d2h_dr2', 0, units='1/m')

        p.run_model()

        # p.model.list_inputs()
        # p.model.list_outputs(print_arrays=True, units=True)

        with self.subTest(check='true airspeed'):
            # True airspeed in level flight is dr_dt
            dt_dr = p.get_val('dt_dr', units='h/NM')
            assert_near_equal(1 / dt_dr, 250.0 * np.ones(nn), tolerance=1.0e-12)

        # normal_force and dgam_dt/dgam_dt_approx come from a near-cancellation of weight and lift
        # (~175,000 lbf each), so their absolute error is limited by the precision of OpenMDAO's
        # lbm/kg unit conversion rather than floating-point precision.
        expected_values = {
            # Normal force in balanced level flight is 0.0
            # when comparing against a value of 0 asserts use abs tol, not rel (so this is
            # within 0.001 lbf)
            'normal_force': (np.zeros(nn), 'lbf', 1.0e-3),
            # Fuselage pitch balanced level flight with zero alpha and wing incidence is 0.0
            'fuselage_pitch': (np.zeros(nn), 'deg', 1.0e-12),
            # Load factor balanced level flight with zero alpha and wing incidence is 1.0
            'load_factor': (np.ones(nn), 'unitless', 1.0e-8),
        }

        if not ground_roll:
            # Both approximate and computed dgam_dt should be zero.
            expected_values['dgam_dt'] = (np.zeros(nn), 'deg/s', 1.0e-7)
            expected_values['dgam_dt_approx'] = (np.zeros(nn), 'deg/s', 1.0e-12)

        for var_name, (expected, units, tol) in expected_values.items():
            with self.subTest(check=var_name):
                actual = p.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tol)

        p.set_val(Dynamic.Mission.VELOCITY, 250 + 10 * np.random.rand(nn), units='kn')
        p.set_val(Dynamic.Vehicle.MASS, 175_000 + 1000 * np.random.rand(nn), units='lbm')
        p.set_val(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            20_000 + 100 * np.random.rand(nn),
            units='lbf',
        )
        p.set_val(Dynamic.Vehicle.LIFT, 175_000 + 1000 * np.random.rand(nn), units='lbf')
        p.set_val(Dynamic.Vehicle.DRAG, 20_000 + 100 * np.random.rand(nn), units='lbf')
        p.set_val(Aircraft.Wing.INCIDENCE, np.random.rand(1), units='deg')

        if not ground_roll:
            p.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 5 * np.random.rand(nn), units='deg')
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 5 * np.random.rand(nn), units='deg')
            p.set_val('dh_dr', 0.1 * np.random.rand(nn), units=None)
            p.set_val('d2h_dr2', 0.01 * np.random.rand(nn), units='1/m')

        p.run_model()

        with self.subTest(check='partials'):
            cpd = p.check_partials(method='cs', out_stream=None)
            assert_check_partials(cpd)

    def test_unsteady_flight_eom(self):
        for ground_roll in True, False:
            with self.subTest(msg=f'ground_roll={ground_roll}'):
                self._test_unsteady_flight_eom(ground_roll=ground_roll)


@use_tempdirs
class GammaCompTestCase(unittest.TestCase):
    """Test the flight path angle component."""

    def test_gamma_comp(self):
        nn = 2

        p = om.Problem()
        p.model.add_subsystem(
            'gamma',
            GammaComp(num_nodes=nn),
            promotes_inputs=['dh_dr', 'd2h_dr2'],
            promotes_outputs=[Dynamic.Mission.FLIGHT_PATH_ANGLE, 'dgam_dr'],
        )
        p.setup(force_alloc_complex=True)
        p.run_model()

        expected_values = {
            Dynamic.Mission.FLIGHT_PATH_ANGLE: ([0.78539816, 0.78539816], None, 1.0e-6),
            'dgam_dr': ([0.5, 0.5], None, 1.0e-6),
        }

        for var_name, (expected, units, tol) in expected_values.items():
            with self.subTest(var=var_name):
                actual = p[var_name] if units is None else p.get_val(var_name, units=units)
                assert_near_equal(actual, expected, tolerance=tol)

        with self.subTest(check='partials'):
            partial_data = p.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
