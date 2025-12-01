import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import UnsteadySolvedODE
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class TestUnsteadySolvedODE(unittest.TestCase):
    """Test the unsteady solved ODE in steady level flight."""

    def _test_unsteady_solved_ode(
        self, ground_roll=False, input_speed_type=SpeedType.MACH, clean=True
    ):
        nn = 5

        p = om.Problem()

        aviary_options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', [build_engine_deck(aviary_options)]
        )

        ode = UnsteadySolvedODE(
            num_nodes=nn,
            input_speed_type=input_speed_type,
            clean=clean,
            ground_roll=ground_roll,
            aviary_options=aviary_options,
            core_subsystems=default_mission_subsystems,
        )

        p.model.add_subsystem('ode', ode, promotes=['*'])

        p.model.set_input_defaults(Dynamic.Atmosphere.MACH, 0.8 * np.ones(nn))
        if ground_roll:
            p.model.set_input_defaults(Dynamic.Atmosphere.MACH, 0.1 * np.ones(nn))
            ode.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, np.zeros(nn), units='deg')

        setup_model_options(p, aviary_options)

        p.setup(force_alloc_complex=True)

        set_params_for_unit_tests(p)

        p.final_setup()

        p.set_val(Aircraft.Wing.FORM_FACTOR, 1.25, units='unitless')
        p.set_val(Dynamic.Atmosphere.SPEED_OF_SOUND, 968.076 * np.ones(nn), units='ft/s')
        p.set_val(Dynamic.Atmosphere.DENSITY, 0.000659904 * np.ones(nn), units='slug/ft**3')
        p.set_val('mach', 0.8 * np.ones(nn), units='unitless')
        p.set_val('mass', 170_000 * np.ones(nn), units='lbm')

        if not ground_roll:
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 0.0 * np.ones(nn), units='rad')
            p.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 4 * np.ones(nn), units='deg')
            p.set_val('dh_dr', 0.0 * np.ones(nn), units='ft/NM')
            p.set_val('d2h_dr2', 0.0 * np.ones(nn), units='1/NM')

        p.set_val('thrust_req', 8000 * np.ones(nn), units='lbf')

        p.run_model()

        drag = p.model.get_val(Dynamic.Vehicle.DRAG, units='lbf')
        lift = p.model.get_val(Dynamic.Vehicle.LIFT, units='lbf')
        thrust_req = p.model.get_val('thrust_req', units='lbf')
        gamma = (
            0 if ground_roll else p.model.get_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, units='deg')
        )
        weight = p.model.get_val('mass', units='lbm') * GRAV_ENGLISH_LBM
        fuelflow = p.model.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/s'
        )
        dmass_dr = p.model.get_val('dmass_dr', units='lbm/ft')
        dt_dr = p.model.get_val('dt_dr', units='s/ft')
        tas = p.model.get_val(Dynamic.Mission.VELOCITY, units='ft/s')
        iwing = p.model.get_val(Aircraft.Wing.INCIDENCE, units='deg')
        alpha = p.model.get_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, units='deg')

        c_alphai = np.cos(np.radians(alpha - iwing))
        s_alphai = np.sin(np.radians(alpha - iwing))

        c_gamma = np.cos(np.radians(gamma))
        s_gamma = np.sin(np.radians(gamma))

        # 1. Test that forces balance along the velocity axis
        assert_near_equal(drag + thrust_req * s_gamma, thrust_req * c_alphai, tolerance=1.0e-12)

        # 2. Test that forces balance normal to the velocity axis
        assert_near_equal(lift + thrust_req * s_alphai, weight * c_gamma, tolerance=1.0e-12)

        # 3. Test that dt_dr is the inverse of true airspeed
        assert_near_equal(tas, 1 / dt_dr, tolerance=1.0e-12)

        # 4. Test that the inverse of dt_dr is true airspeed
        assert_near_equal(tas, 1 / dt_dr, tolerance=1.0e-12)

        # 5. Test that fuelflow (lbf/s) * dt_dr (s/ft) is equal to dmass_dr
        assert_near_equal(fuelflow * dt_dr, dmass_dr, tolerance=1.0e-12)

        p.check_partials(out_stream=None, method='cs', excludes=['*params*', '*aero*'])
        # issue #495
        # dTAS_dt_approx wrt flight_path_angle | abs | fwd-fd | 1.8689625335382314
        # dTAS_dt_approx wrt flight_path_angle | rel | fwd-fd | 1.0
        # assert_check_partials(cpd, atol=1e-6, rtol=1e-6)

    def test_steady_level_flight(self):
        # issue #494: why not ground_roll in [True] ?
        for ground_roll in [False]:
            with self.subTest(msg=f'ground_roll={ground_roll}'):
                self._test_unsteady_solved_ode(ground_roll=ground_roll)


if __name__ == '__main__':
    unittest.main()
