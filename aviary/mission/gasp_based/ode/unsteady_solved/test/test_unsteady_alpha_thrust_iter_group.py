import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_control_iter_group import \
    UnsteadyControlIterGroup
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_flight_conditions import \
    UnsteadySolvedFlightConditions
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.enums import LegacyCode, SpeedType
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic


class TestUnsteadyAlphaThrustIterGroup(unittest.TestCase):
    """
    Test the UnsteadyControlIterGroup.
    """

    def _test_unsteady_alpha_thrust_iter_group(self, ground_roll=False):
        nn = 5

        # just need aero subsystem
        aero = CoreAerodynamicsBuilder(code_origin=LegacyCode.GASP)

        p = om.Problem()

        fc = UnsteadySolvedFlightConditions(num_nodes=nn,
                                            input_speed_type=SpeedType.TAS,
                                            ground_roll=ground_roll)

        p.model.add_subsystem("fc", subsys=fc,
                              promotes_inputs=["*"],
                              promotes_outputs=["*"])

        g = UnsteadyControlIterGroup(num_nodes=nn,
                                     ground_roll=ground_roll,
                                     clean=True,
                                     aviary_options=get_option_defaults(),
                                     core_subsystems=[aero])

        ig = p.model.add_subsystem("iter_group",
                                   subsys=g,
                                   promotes_inputs=["*"],
                                   promotes_outputs=["*"])

        if ground_roll:
            ig.set_input_defaults("alpha", np.zeros(nn), units="deg")

        p.setup(force_alloc_complex=True)

        set_params_for_unit_tests(p)

        p.final_setup()

        p.set_val(Dynamic.Mission.SPEED_OF_SOUND, 968.076 * np.ones(nn), units="ft/s")
        p.set_val(
            Dynamic.Mission.DENSITY, 0.000659904 * np.ones(nn), units="slug/ft**3"
        )
        p.set_val(Dynamic.Mission.VELOCITY, 487 * np.ones(nn), units="kn")
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

        cpd = p.check_partials(out_stream=None, method="cs", step=1.01e-40,
                               excludes=["*params*", "*aero*"])
        assert_check_partials(cpd, atol=1e-10, rtol=1e-10)

    def test_iter_group(self):
        # issue #494: why not ground_roll in [True] ?
        for ground_roll in [False]:
            with self.subTest(msg=f"ground_roll={ground_roll}"):
                self._test_unsteady_alpha_thrust_iter_group(ground_roll=ground_roll)


if __name__ == '__main__':
    unittest.main()
