import numpy as np

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import ThrottleAllocation
from aviary.variable_info.variables import Dynamic


class ThrottleAllocator(om.ExplicitComponent):
    """
    Component that computes the throttle values for multiplpe engine types based on
    the settings for the phase.
    """
    def initialize(self):
        self.options.declare(
            'num_nodes',
            types=int,
            lower=0
        )
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )
        self.options.declare(
            'throttle_allocation', default=ThrottleAllocation.FIXED,
            types=ThrottleAllocation,
            desc='Flag that determines how to handle throttles for multiple engines.'
        )

    def setup(self):
        options: AviaryValues = self.options['aviary_options']
        nn = self.options['num_nodes']
        engine_models = options.get_val('engine_models')
        alloc_mode = self.options['throttle_allocation']
        num_engines = len(engine_models)

        self.add_input(
            "aggregate_throttle",
            np.ones(nn),
            units="unitless",
            desc="Solver-controlled aggregate throttle."
        )

        if alloc_mode == ThrottleAllocation.DYNAMIC_PARAMETER:
            alloc_shape = (nn, num_engines - 1)
        else:
            alloc_shape = (num_engines - 1, )

        self.add_input(
            "throttle_allocations",
            np.ones(alloc_shape) * 1.0 / num_engines,
            units="unitless",
            desc="Throttle allocation for engines."
        )

        self.add_output(
            Dynamic.Mission.THROTTLE,
            np.ones((nn, num_engines)),
            units="unitless",
            desc="Throttle setting for all engines."
        )

        if alloc_mode == ThrottleAllocation.DYNAMIC_PARAMETER:
            alloc_shape = nn
        else:
            alloc_shape = 1

        self.add_output(
            "throttle_allocation_sum",
            np.ones(alloc_shape),
            desc="Sum of the optimizer allocation values. Constrain to less than 1.0."
        )

        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options: AviaryValues = self.options['aviary_options']
        nn = self.options['num_nodes']
        alloc_mode = self.options['throttle_allocation']

        agg_throttle = inputs["aggregate_throttle"]
        allocation = inputs["throttle_allocations"]

        if alloc_mode == ThrottleAllocation.DYNAMIC_PARAMETER:
            outputs[Dynamic.Mission.THROTTLE][:, :-1] = np.einsum("i,jk->ik", agg_throttle, allocation)
        else:
            outputs[Dynamic.Mission.THROTTLE][:, :-1] = np.einsum("i,j->ij", agg_throttle, allocation)

        sum_alloc = np.sum(allocation)

        outputs[Dynamic.Mission.THROTTLE][:, -1] = agg_throttle * (1.0 - sum_alloc)

        outputs["throttle_allocation_sum"] = sum_alloc

    def Zcompute_partials(self, inputs, partials, discrete_inputs=None):
        options: AviaryValues = self.options['aviary_options']
        nn = self.options['num_nodes']

        agg_throttle = inputs["aggregate_throttle"]
        allocation = inputs["throttle_allocations"]

        partials[Dynamic.Mission.THROTTLE, "aggregate_throttle"] = allocation

        partials[Dynamic.Mission.THROTTLE, "throttle_allocations"] = agg_throttle
