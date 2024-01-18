import unittest

import openmdao.api as om

from aviary.mission.flops_based.ode.mission_EOM import MissionEOM
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_data.flops_data.full_mission_test_data import \
    data
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic


class MissionEOMTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        time, _ = data.get_item('time')

        prob.model.add_subsystem(
            "mission_EOM",
            MissionEOM(num_nodes=len(time)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        do_validation_test(self.prob,
                           'full_mission_test_data',
                           input_validation_data=data,
                           output_validation_data=data,
                           input_keys=[Dynamic.Mission.DRAG,
                                       Dynamic.Mission.MASS,
                                       Dynamic.Mission.THRUST_TOTAL,
                                       Dynamic.Mission.THRUST_MAX_TOTAL,
                                       Dynamic.Mission.VELOCITY,
                                       Dynamic.Mission.VELOCITY_RATE],
                           output_keys=[Dynamic.Mission.ALTITUDE_RATE,
                                        # TODO: why does altitude_rate_max fail for cruise?
                                        #    - actual: 760.55416759
                                        #    - desired: 3.86361517135375
                                        # Dynamic.Mission.ALTITUDE_RATE_MAX,
                                        Dynamic.Mission.DISTANCE_RATE,
                                        Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                                        Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS],
                           tol=1e-12)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
