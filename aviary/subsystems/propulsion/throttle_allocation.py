import numpy as np

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
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

    def setup(self):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_models = options.get_val('engine_models')
        num_engines = len(engine_models)

        self.add_input(
            Dynamic.Mission.THROTTLE,
            np.ones(nn),
            desc="Solver-controlled aggregate throttle."
        )

        for j in range(num_engines - 1):
            self.add_input(
                f"throttle_alloc_engine_{j}",
                np.ones(nn),
                desc=f"Throttle allocation for engine {j}."
            )

        for j in range(num_engines):
            self.add_output(
                f"throttle_{j}",
                np.ones(nn),
                desc=f"Throttle setting for engine {j}."
            )

        # TODO: make this scaler if we use static parameters.
        self.add_input(
            "throttle_allocation_sum",
            np.ones(nn),
            desc="Sum of the optimizer allocation values. Constrain to less than 1.0."
        )

        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_models = options.get_val('engine_models')
        num_engines = len(engine_models)

        agg_throttle = inputs[Dynamic.Mission.THROTTLE]

        sum_alloc = 0.0
        for j in range(num_engines):

            if j < num_engines - 1:
                allocation = inputs[f"throttle_alloc_engine_{j}"]
                sum_alloc += allocation
            else:
                allocation = 1.0 - sum_alloc

            outputs[f"throttle_{j}"] = allocation * agg_throttle

        outputs["throttle_allocation_sum"] = sum_alloc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_models = options.get_val('engine_models')
        num_engines = len(engine_models)

        agg_throttle = inputs[Dynamic.Mission.THROTTLE]

        sum_alloc = 0.0

        for j in range(num_engines):

            if j < num_engines - 1:
                allocation = inputs[f"throttle_alloc_engine_{j}"]
                sum_alloc += allocation

                partials[f"throttle_{j}", f"throttle_alloc_engine_{j}"] = agg_throttle
                partials[f"throttle_{num_engines - 1}", f"throttle_alloc_engine_{j}"] = -agg_throttle

            else:
                allocation = 1.0 - sum_alloc

            partials[f"throttle_{j}", Dynamic.Mission.THROTTLE] = allocation
