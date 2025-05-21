import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_eom import UnsteadySolvedEOM
from aviary.variable_info.variables import Aircraft, Dynamic


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

        if not ground_roll:
            p.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 0.0, units='deg')
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 0, units='deg')
            p.set_val('dh_dr', 0, units=None)
            p.set_val('d2h_dr2', 0, units='1/m')

        p.run_model()

        # p.model.list_inputs()
        # p.model.list_outputs(print_arrays=True, units=True)

        dt_dr = p.get_val('dt_dr', units='h/NM')
        normal_force = p.get_val('normal_force', units='lbf')
        load_factor = p.get_val('load_factor', units='unitless')
        fuselage_pitch = p.get_val('fuselage_pitch', units='deg')

        # True airspeed in level flight is dr_dt
        assert_near_equal(1 / dt_dr, 250.0 * np.ones(nn), tolerance=1.0e-12)

        # Normal force in balanced level flight is 0.0
        assert_near_equal(normal_force, np.zeros(nn), tolerance=1.0e-12)

        # Fuselage pitch balanced level flight with zero alpha and wing incidence is 0.0
        assert_near_equal(fuselage_pitch, np.zeros(nn), tolerance=1.0e-12)

        # Load factor balanced level flight with zero alpha and wing incidence is 1.0
        assert_near_equal(load_factor, np.ones(nn), tolerance=1.0e-12)

        if not ground_roll:
            dgam_dt = p.get_val('dgam_dt', units='deg/s')
            dgam_dt_approx = p.get_val('dgam_dt_approx', units='deg/s')

            # Both approximate and computed dgam_dt should be zero.
            assert_near_equal(dgam_dt, np.zeros(nn), tolerance=1.0e-12)
            assert_near_equal(dgam_dt_approx, np.zeros(nn), tolerance=1.0e-12)

        p.set_val(Dynamic.Mission.VELOCITY, 250 + 10 * np.random.rand(nn), units='kn')
        p.set_val('mass', 175_000 + 1000 * np.random.rand(nn), units='lbm')
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

        with np.printoptions(linewidth=1024):
            cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)

    def test_unsteady_solved_eom(self):
        for ground_roll in True, False:
            with self.subTest(msg=f'ground_roll={ground_roll}'):
                self._test_unsteady_solved_eom(ground_roll=ground_roll)


class TestUnsteadySolvedEOM2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_eom as unsteady

        unsteady.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_eom as unsteady

        unsteady.GRAV_ENGLISH_LBM = 1.0

    def _test_unsteady_solved_eom(self, ground_roll=False):
        nn = 2
        p = om.Problem()
        p.model.add_subsystem(
            'eom',
            UnsteadySolvedEOM(num_nodes=nn, ground_roll=ground_roll),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        p.setup(force_alloc_complex=True)

        p.set_val(Dynamic.Mission.VELOCITY, 250 + 10 * np.random.rand(nn), units='kn')
        p.set_val('mass', 175_000 + 1000 * np.random.rand(nn), units='lbm')
        p.set_val(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            20_000 + 100 * np.random.rand(nn),
            units='lbf',
        )
        p.set_val(Dynamic.Vehicle.LIFT, 175_000 + 1000 * np.random.rand(nn), units='lbf')
        p.set_val(Dynamic.Vehicle.DRAG, 20_000 + 100 * np.random.rand(nn), units='lbf')
        p.set_val(Aircraft.Wing.INCIDENCE, np.random.rand(1), units='deg')

        if not ground_roll:
            p.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, nn * np.random.rand(nn), units='deg')
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, nn * np.random.rand(nn), units='deg')
            p.set_val('dh_dr', 0.1 * np.random.rand(nn), units=None)
            p.set_val('d2h_dr2', 0.01 * np.random.rand(nn), units='1/m')

        partial_data = p.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=5e-11, rtol=1e-11)

    def test_unsteady_solved_eom(self):
        for ground_roll in True, False:
            with self.subTest(msg=f'ground_roll={ground_roll}'):
                self._test_unsteady_solved_eom(ground_roll=ground_roll)


if __name__ == '__main__':
    unittest.main()
