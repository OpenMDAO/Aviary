import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.interface.default_phase_info.two_dof import aero
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_control_iter_group import \
    UnsteadyControlIterGroup
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_flight_conditions import \
    UnsteadySolvedFlightConditions
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TestUnsteadyAlphaThrustIterGroup(unittest.TestCase):

    def _test_unsteady_alpha_thrust_iter_group(self, ground_roll=False):
        nn = 5

        p = om.Problem()

        # TODO: paramport
        param_port = ParamPort()

        p.model.add_subsystem("params", param_port, promotes=["*"])

        fc = UnsteadySolvedFlightConditions(num_nodes=nn,
                                            input_speed_type=SpeedType.TAS,
                                            ground_roll=ground_roll)

        p.model.add_subsystem("fc", subsys=fc,
                              promotes_inputs=["*"],
                              promotes_outputs=["*"])

        g = UnsteadyControlIterGroup(num_nodes=nn,
                                     ground_roll=ground_roll,
                                     clean=True,
                                     core_subsystems=[aero])

        p.model.add_subsystem("iter_group",
                              subsys=g,
                              promotes_inputs=["*"],
                              promotes_outputs=["*"])

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
            p.set_val("dh_dr", 0.0 * np.ones(nn), units=None)
            p.set_val("d2h_dr2", 0.0 * np.ones(nn), units="1/NM")

        p.set_val("thrust_req", 8000 * np.ones(nn), units="lbf")

        p.run_model()

        drag = p.model.get_val(Dynamic.Mission.DRAG, units="lbf")
        lift = p.model.get_val(Dynamic.Mission.LIFT, units="lbf")
        thrust_req = p.model.get_val("thrust_req", units="lbf")
        gamma = 0 if ground_roll else p.model.get_val(
            Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
        weight = p.model.get_val("mass", units="lbm") * GRAV_ENGLISH_LBM
        iwing = p.model.get_val(Aircraft.Wing.INCIDENCE, units="deg")
        alpha = iwing if ground_roll else p.model.get_val("alpha", units="deg")

        c_alphai = np.cos(np.radians(alpha - iwing))
        s_alphai = np.sin(np.radians(alpha - iwing))

        c_gamma = np.cos(np.radians(gamma))
        s_gamma = np.sin(np.radians(gamma))

        # 1. Test that forces balance along the velocity axis
        assert_near_equal(drag + thrust_req * s_gamma, thrust_req * c_alphai)

        # 2. Test that forces balance normal to the velocity axis
        assert_near_equal(lift + thrust_req * s_alphai, weight * c_gamma)

    def test_iter_group(self):
        for ground_roll in [False]:
            with self.subTest(msg=f"ground_roll={ground_roll}"):
                self._test_unsteady_alpha_thrust_iter_group(ground_roll=ground_roll)


if __name__ == '__main__':
    unittest.main()
