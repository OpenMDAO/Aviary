import unittest

import openmdao.api as om

from aviary.mission.flops_based.ode.specific_energy_rate import SpecificEnergyRate
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_data.flops_data.full_mission_test_data import \
    data
from aviary.validation_cases.validation_tests import do_validation_test
from aviary.variable_info.variables import Dynamic


class SpecificEnergyRateTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        time, _ = data.get_item('time')

        prob.model.add_subsystem(
            "specific_energy_rate",
            SpecificEnergyRate(num_nodes=len(time)),
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
                                       Dynamic.Mission.VELOCITY],
                           output_keys=Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                           tol=1e-12)

    # TODO IO test will fail until mission variable hirerarchy implemented
    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
