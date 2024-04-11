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

        self.add_input(
            Dynamic.Mission.THROTTLE,
            np.ones(nn),
            desc="Solver-controlled aggregate throttle."
        )

        for engine in engine_models[:-1]:
            self.add_input(
                f"throttle_alloc_engine_{engine.name}",
                np.ones(nn),
                desc=f"Throttle allocation for engine '{engine.name}'."
            )

        for engine in engine_models:
            self.add_output(
                f"throttle_{engine.name}",
                np.ones(nn),
                desc=f"Throttle setting for engine '{engine.name}'."
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
        for engine in engine_models[:-1]:
            name = engine.name

            allocation = inputs[f"throttle_alloc_engine_{name}"]
            sum_alloc += allocation

            outputs[f"throttle_{name}"] = allocation * agg_throttle

        allocation = 1.0 - sum_alloc
        outputs[f"throttle_{engine_models[-1].name}"] = allocation * agg_throttle

        outputs["throttle_allocation_sum"] = sum_alloc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        nn = self.options['num_nodes']
        options: AviaryValues = self.options['aviary_options']
        engine_models = options.get_val('engine_models')
        num_engines = len(engine_models)

        agg_throttle = inputs[Dynamic.Mission.THROTTLE]

        sum_alloc = 0.0
        last_throttle = f"throttle_{engine_models[-1].name}"

        for engine in engine_models:
            name = engine.name
            input_name = f"throttle_alloc_engine_{name}"

            if j < num_engines - 1:
                allocation = inputs[input_name]
                sum_alloc += allocation

                partials[f"throttle_{name}", input_name] = agg_throttle
                partials[last_throttle, input_name] = -agg_throttle

            else:
                allocation = 1.0 - sum_alloc

            partials[f"throttle_{name}", Dynamic.Mission.THROTTLE] = allocation
