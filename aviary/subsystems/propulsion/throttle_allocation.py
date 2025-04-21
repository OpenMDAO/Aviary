import numpy as np
import openmdao.api as om

from aviary.variable_info.enums import ThrottleAllocation
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic


class ThrottleAllocator(om.ExplicitComponent):
    """
    Component that computes the throttle values for multiple engine types based on
    the settings for the phase.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, lower=0)
        self.options.declare(
            'throttle_allocation',
            default=ThrottleAllocation.FIXED,
            types=ThrottleAllocation,
            desc='Flag that determines how to handle throttles for multiple engines.',
        )

        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        nn = self.options['num_nodes']
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        alloc_mode = self.options['throttle_allocation']

        self.add_input(
            'aggregate_throttle',
            np.ones(nn),
            units='unitless',
            desc='Solver-controlled aggregate throttle.',
        )

        if alloc_mode == ThrottleAllocation.DYNAMIC:
            alloc_shape = (nn, num_engine_type - 1)
        else:
            alloc_shape = (num_engine_type - 1,)

        self.add_input(
            'throttle_allocations',
            np.ones(alloc_shape) * 1.0 / num_engine_type,
            units='unitless',
            desc='Throttle allocation for engines.',
        )

        self.add_output(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            np.ones((nn, num_engine_type)),
            units='unitless',
            desc='Throttle setting for all engines.',
        )

        if alloc_mode == ThrottleAllocation.DYNAMIC:
            alloc_shape = nn
        else:
            alloc_shape = 1

        self.add_output(
            'throttle_allocation_sum',
            np.ones(alloc_shape),
            desc='Sum of the optimizer allocation values. Constrain to less than 1.0.',
        )

        cols = np.repeat(np.arange(nn), num_engine_type)
        rows = np.arange(nn * num_engine_type)
        self.declare_partials(
            of=[Dynamic.Vehicle.Propulsion.THROTTLE],
            wrt=['aggregate_throttle'],
            rows=rows,
            cols=cols,
        )

        if alloc_mode == ThrottleAllocation.DYNAMIC:
            a = num_engine_type
            b = a - 1
            row = np.arange(a)
            col = np.arange(b)
            rows = np.repeat(row, b)
            cols = np.tile(col, num_engine_type)
            all_rows = np.tile(rows, nn) + a * np.repeat(np.arange(nn), a * b)
            all_cols = np.tile(cols, nn) + b * np.repeat(np.arange(nn), a * b)
            self.declare_partials(
                of=[Dynamic.Vehicle.Propulsion.THROTTLE],
                wrt=['throttle_allocations'],
                rows=all_rows,
                cols=all_cols,
            )

            rows = np.repeat(np.arange(nn), b)
            cols = np.arange(nn * b)
            self.declare_partials(
                of=['throttle_allocation_sum'],
                wrt=['throttle_allocations'],
                rows=rows,
                cols=cols,
                val=1.0,
            )
        else:
            self.declare_partials(
                of=[Dynamic.Vehicle.Propulsion.THROTTLE], wrt=['throttle_allocations']
            )
            self.declare_partials(
                of=['throttle_allocation_sum'], wrt=['throttle_allocations'], val=1.0
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        alloc_mode = self.options['throttle_allocation']

        agg_throttle = inputs['aggregate_throttle']
        allocation = inputs['throttle_allocations']

        if alloc_mode == ThrottleAllocation.DYNAMIC:
            outputs[Dynamic.Vehicle.Propulsion.THROTTLE][:, :-1] = np.einsum(
                'i,ij->ij', agg_throttle, allocation
            )
            sum_alloc = np.sum(allocation, axis=1)
        else:
            outputs[Dynamic.Vehicle.Propulsion.THROTTLE][:, :-1] = np.einsum(
                'i,j->ij', agg_throttle, allocation
            )
            sum_alloc = np.sum(allocation)

        outputs[Dynamic.Vehicle.Propulsion.THROTTLE][:, -1] = agg_throttle * (1.0 - sum_alloc)

        outputs['throttle_allocation_sum'] = sum_alloc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        nn = self.options['num_nodes']
        alloc_mode = self.options['throttle_allocation']
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        agg_throttle = inputs['aggregate_throttle']
        allocation = inputs['throttle_allocations']

        if alloc_mode == ThrottleAllocation.DYNAMIC:
            sum_alloc = np.sum(allocation, axis=1)
            allocs = np.vstack((allocation.T, 1.0 - sum_alloc))
            partials[Dynamic.Vehicle.Propulsion.THROTTLE, 'aggregate_throttle'] = allocs.T.ravel()

            ne = num_engine_type - 1
            mask1 = np.eye(ne)
            mask2 = -np.ones(ne)
            mask = np.vstack((mask1, mask2)).ravel()

            deriv = np.outer(agg_throttle, mask).reshape((nn * (ne + 1), ne))
            partials[Dynamic.Vehicle.Propulsion.THROTTLE, 'throttle_allocations'] = deriv.ravel()

        else:
            sum_alloc = np.sum(allocation)
            allocs = np.hstack((allocation, 1.0 - sum_alloc))
            partials[Dynamic.Vehicle.Propulsion.THROTTLE, 'aggregate_throttle'] = np.tile(
                allocs, nn
            )

            ne = num_engine_type - 1
            mask1 = np.eye(ne)
            mask2 = -np.ones(ne)
            mask = np.vstack((mask1, mask2)).ravel()

            deriv = np.outer(agg_throttle, mask).reshape((nn * (ne + 1), ne))
            partials[Dynamic.Vehicle.Propulsion.THROTTLE, 'throttle_allocations'] = deriv

        # sum_alloc = np.sum(allocation)

        # outputs[Dynamic.Vehicle.Propulsion.THROTTLE][:, -1] = agg_throttle * (1.0 - sum_alloc)

        # outputs["throttle_allocation_sum"] = sum_alloc
