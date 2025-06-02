import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.subsystems.propulsion.throttle_allocation import ThrottleAllocator
from aviary.variable_info.enums import ThrottleAllocation
from aviary.variable_info.variables import Aircraft


class ThrottleAllocationTest(unittest.TestCase):
    def setUp(self):
        self.options = {
            Aircraft.Engine.NUM_ENGINES: np.array([1, 1, 1]),
        }

    def test_derivs_fixed_or_static(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            'comp',
            ThrottleAllocator(
                num_nodes=4, throttle_allocation=ThrottleAllocation.FIXED, **self.options
            ),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('throttle_allocations', val=np.array([0.24, 0.55]))
        prob.set_val('aggregate_throttle', val=np.array([0.3, 0.41, 0.52, 0.64]))
        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e-10, rtol=1e-10)

    def test_derivs_dynamic(self):
        prob = om.Problem()
        model = prob.model
        model.add_subsystem(
            'comp',
            ThrottleAllocator(
                num_nodes=4, throttle_allocation=ThrottleAllocation.DYNAMIC, **self.options
            ),
            promotes=['*'],
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(
            'throttle_allocations', val=np.array([0.24, 0.55, 0.33, 0.33, 0.6, 0.1, 0.1, 0.6])
        )
        prob.set_val('aggregate_throttle', val=np.array([0.3, 0.41, 0.52, 0.64]))
        prob.run_model()

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
