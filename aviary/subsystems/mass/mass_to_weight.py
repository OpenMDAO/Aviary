import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Mission


class MassToWeight(om.ExplicitComponent):
    """Component to convert mass to weight."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            'mass',
            val=np.ones(nn),
            units='kg',
            desc='mass of the aircraft',
        )

        self.add_output(
            'weight',
            val=np.ones(nn),
            units='N',
            desc='weight of the aircraft',
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        grav_metric = self.options[Mission.GRAVITY][0]
        self.declare_partials(
            'weight', 'mass', rows=arange, cols=arange, val=np.full(nn, grav_metric)
        )

    def compute(self, inputs, outputs):
        grav_metric = self.options[Mission.GRAVITY][0]
        outputs['weight'] = inputs['mass'] * grav_metric
