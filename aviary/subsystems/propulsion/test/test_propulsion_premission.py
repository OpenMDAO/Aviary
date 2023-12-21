import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.propulsion.propulsion_premission import (
    PropulsionPreMission, PropulsionSum)
from aviary.utils.aviary_values import AviaryValues
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft


class PropulsionPreMissionTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case(self):
        aviary_values = get_flops_inputs('LargeSingleAisle2FLOPS')
        options = aviary_values

        self.prob.model = PropulsionPreMission(aviary_options=options)

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST, options.get_val(
            Aircraft.Engine.SCALED_SLS_THRUST, units='lbf'))

        self.prob.run_model()

        sls_thrust = self.prob.get_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST)

        expected_sls_thrust = np.array([54602.])

        assert_near_equal(sls_thrust, expected_sls_thrust, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_propulsion_sum(self):
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([1, 2, 5]))
        # it doesn't matter what goes in engine models, as long as it is length 3
        options.set_val('engine_models', [1, 1, 1])
        self.prob.model = om.Group()
        self.prob.model.add_subsystem('propsum',
                                      PropulsionSum(aviary_options=options),
                                      promotes=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST,
                          np.array([1000, 3000, 13200]), units='lbf')

        self.prob.run_model()

        total_thrust = self.prob.get_val(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')

        expected_thrust = 73000.0

        assert_near_equal(total_thrust, expected_thrust, tolerance=1e-12)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
