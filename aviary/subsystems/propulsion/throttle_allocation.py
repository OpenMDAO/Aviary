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
                1.0,
                desc=f"Throttle allocation for engine {j}."
            )

        for j in range(num_engines):
            self.add_output(
                f"throttle_{j}",
                1.0,
                desc=f"Throttle setting for engine {j}."
            )

        self.declare_partials(of='*', wrt='*')
