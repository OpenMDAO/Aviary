import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.flops_based.mach_number import MachNumber
from aviary.variable_info.variables import Dynamic


class MachNumberTest(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            Dynamic.Mission.MACH,
            MachNumber(num_nodes=1),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        # for key, temp in FLOPS_Test_Data.items():
        # TODO currently no way to use FLOPS test case data for mission components

        self.prob.set_val(Dynamic.Mission.VELOCITY, val=347, units='ft/s')
        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, val=1045, units='ft/s')
        self.prob.run_model()

        tol = 1e-3
        assert_near_equal(
            self.prob.get_val(Dynamic.Mission.MACH, units='unitless'), 0.332, tol
        )  # check the value of each output

        # TODO resolve partials wrt gravity (decide on implementation of gravity)
        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(
            partial_data, atol=1e-6, rtol=1e-6
        )  # check the partial derivatives

    # TODO IO test will fail until mission variable hirerarchy implemented
    # def test_IO(self):
    #     assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
