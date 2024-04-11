import numpy as np

import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues


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