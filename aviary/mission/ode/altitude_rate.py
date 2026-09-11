import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic, Mission
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option


class AltitudeRate(om.ExplicitComponent):
    """
    Rutowski "Energy Approach to the General Aircraft Performance Problem", doi 10.2514/8.2956
    Equation 6.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']
        add_aviary_input(
            self,
            Dynamic.Mission.SPECIFIC_ENERGY_RATE,
            shape=nn,
            units='m/s',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY_RATE,
            shape=nn,
            units='m/s**2',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='m/s',
        )
        add_aviary_output(
            self,
            Dynamic.Mission.ALTITUDE_RATE,
            shape=nn,
            units='m/s',
        )

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Mission.ALTITUDE_RATE,
            [
                Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.VELOCITY,
            ],
            rows=arange,
            cols=arange,
            val=1,
        )

    def compute(self, inputs, outputs):
        grav_metric = self.options[Mission.GRAVITY][0]
        specific_power = inputs[Dynamic.Mission.SPECIFIC_ENERGY_RATE]
        acceleration = inputs[Dynamic.Mission.VELOCITY_RATE]
        velocity = inputs[Dynamic.Mission.VELOCITY]

        outputs[Dynamic.Mission.ALTITUDE_RATE] = (
            specific_power - (velocity * acceleration) / grav_metric
        )

    def compute_partials(self, inputs, J):
        grav_metric = self.options[Mission.GRAVITY][0]
        acceleration = inputs[Dynamic.Mission.VELOCITY_RATE]
        velocity = inputs[Dynamic.Mission.VELOCITY]

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY_RATE] = -velocity / grav_metric
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = -acceleration / grav_metric
