import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import \
    UnsteadySolvedODE
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


class TestUnsteadySolvedODE(unittest.TestCase):
    """ Test the unsteady solved ODE in steady level flight. """

    def _test_unsteady_solved_ode(self, ground_roll=False, input_speed_type=SpeedType.TAS, clean=True):
        nn = 5

        p = om.Problem()

        ode = UnsteadySolvedODE(num_nodes=nn,
                                input_speed_type=input_speed_type,
                                clean=clean,
                                ground_roll=ground_roll,
                                aviary_options=get_option_defaults(),
                                core_subsystems=default_mission_subsystems,
                                balance_throttle=True)

        p.model.add_subsystem("ode", ode, promotes=["*"])

        # TODO: paramport
        param_port = ParamPort()
        for key, data in param_port.param_data.items():
            p.model.set_input_defaults(key, **data)

        p.setup(force_alloc_complex=True)

        p.final_setup()

        p.set_val(Dynamic.Mission.SPEED_OF_SOUND, 968.076 * np.ones(nn), units="ft/s")
        p.set_val("rho", 0.000659904 * np.ones(nn), units="slug/ft**3")
        p.set_val("TAS", 487 * np.ones(nn), units="kn")
        p.set_val("mass", 170_000 * np.ones(nn), units="lbm")
        p.set_val("dTAS_dr", 0.0 * np.ones(nn), units="kn/NM")

        if not ground_roll:
            p.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, 0.0 * np.ones(nn), units="rad")
            p.set_val("alpha", 4 * np.ones(nn), units="deg")
            p.set_val("dh_dr", 0.0 * np.ones(nn), units="ft/NM")
            p.set_val("d2h_dr2", 0.0 * np.ones(nn), units="1/NM")

        p.set_val("thrust_req", 8000 * np.ones(nn), units="lbf")

        p.run_model()

        drag = p.model.get_val(Dynamic.Mission.DRAG, units="lbf")
        lift = p.model.get_val(Dynamic.Mission.LIFT, units="lbf")
        thrust_req = p.model.get_val("thrust_req", units="lbf")
        gamma = 0 if ground_roll else p.model.get_val(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
        weight = p.model.get_val("mass", units="lbm") * GRAV_ENGLISH_LBM
        fuelflow = p.model.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units="lbm/s")
        dmass_dr = p.model.get_val("dmass_dr", units="lbm/ft")
        dt_dr = p.model.get_val("dt_dr", units="s/ft")
        tas = p.model.get_val("TAS", units="ft/s")
        iwing = p.model.get_val(Aircraft.Wing.INCIDENCE, units="deg")
        alpha = p.model.get_val("alpha", units="deg")
        throttle = p.model.get_val(Dynamic.Mission.THROTTLE, units="unitless")

        c_alphai = np.cos(np.radians(alpha - iwing))
        s_alphai = np.sin(np.radians(alpha - iwing))

        c_gamma = np.cos(np.radians(gamma))
        s_gamma = np.sin(np.radians(gamma))

        # 1. Test that forces balance along the velocity axis
        assert_near_equal(drag + thrust_req * s_gamma,
                          thrust_req * c_alphai, tolerance=1.0E-12)

        # 2. Test that forces balance normal to the velocity axis
        assert_near_equal(lift + thrust_req * s_alphai,
                          weight * c_gamma, tolerance=1.0E-12)

        # 3. Test that dt_dr is the inverse of true airspeed
        assert_near_equal(tas, 1/dt_dr, tolerance=1.0E-12)

        # 4. Test that the inverse of dt_dr is true airspeed
        assert_near_equal(tas, 1/dt_dr, tolerance=1.0E-12)

        # 5. Test that fuelflow (lbf/s) * dt_dr (s/ft) is equal to dmass_dr
        assert_near_equal(fuelflow * dt_dr, dmass_dr, tolerance=1.0E-12)

    def test_steady_level_flight(self):

        for ground_roll in [False]:
            with self.subTest(msg=f"ground_roll={ground_roll}"):
                self._test_unsteady_solved_ode(ground_roll=ground_roll)


if __name__ == "__main__":
    unittest.main()
