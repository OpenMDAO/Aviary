import numpy as np
import openmdao.api as om

import aviary.constants as constants
from aviary.variable_info.variables import Dynamic


class AltitudeRate(om.ExplicitComponent):
    """
    Rutowski "Energy Approach to the General Aircraft Performance Problem", doi 10.2514/8.2956
    Equation 6
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(Dynamic.Vehicle.SPECIFIC_ENERGY_RATE, val=np.ones(
            nn), desc='current specific power', units='m/s')
        self.add_input(Dynamic.Atmosphere.VELOCITY_RATE, val=np.ones(
            nn), desc='current acceleration', units='m/s**2')
        self.add_input(
            Dynamic.Atmosphere.VELOCITY,
            val=np.ones(nn),
            desc='current velocity',
            units='m/s')
        self.add_output(
            Dynamic.Atmosphere.ALTITUDE_RATE,
            val=np.ones(nn),
            desc='current climb rate',
            units='m/s',
        )

    def compute(self, inputs, outputs):
        gravity = constants.GRAV_METRIC_FLOPS
        specific_power = inputs[Dynamic.Vehicle.SPECIFIC_ENERGY_RATE]
        acceleration = inputs[Dynamic.Atmosphere.VELOCITYITY_RATE]
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]

        outputs[Dynamic.Atmosphere.ALTITUDE_RATE] = (
            specific_power - (velocity * acceleration) / gravity
        )

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Atmosphere.ALTITUDE_RATE,
            [
                Dynamic.Vehicle.SPECIFIC_ENERGY_RATE,
                Dynamic.Atmosphere.VELOCITYITY_RATE,
                Dynamic.Atmosphere.VELOCITY,
            ],
            rows=arange,
            cols=arange,
            val=1,
        )

    def compute_partials(self, inputs, J):
        gravity = constants.GRAV_METRIC_FLOPS
        acceleration = inputs[Dynamic.Atmosphere.VELOCITYITY_RATE]
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]

        J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Atmosphere.VELOCITYITY_RATE] = (
            -velocity / gravity
        )
        J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Atmosphere.VELOCITY] = (
            -acceleration / gravity
        )
