import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic, Mission
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option


class SpecificEnergyRate(om.ExplicitComponent):
    """
    Rutowski "Energy Approach to the General Aircraft Performance Problem", doi 10.2514/8.2956
    Equation 5.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            desc='current velocity',
            units='m/s',
        )
        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, desc='current mass', units='kg')
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            shape=nn,
            desc='current thrust',
            units='N',
        )
        add_aviary_input(self, Dynamic.Vehicle.DRAG, shape=nn, desc='current drag', units='N')
        add_aviary_output(
            self,
            Dynamic.Mission.SPECIFIC_ENERGY_RATE,
            shape=nn,
            desc='current specific power',
            units='m/s',
        )

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Mission.SPECIFIC_ENERGY_RATE,
            [
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        grav_metric = self.options[Mission.GRAVITY][0]

        velocity = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * grav_metric
        outputs[Dynamic.Mission.SPECIFIC_ENERGY_RATE] = velocity * (thrust - drag) / weight

    def compute_partials(self, inputs, J):
        grav_metric = self.options[Mission.GRAVITY][0]

        velocity = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * grav_metric

        J[Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Mission.VELOCITY] = (thrust - drag) / weight
        J[
            Dynamic.Mission.SPECIFIC_ENERGY_RATE,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
        ] = velocity / weight
        J[Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Vehicle.DRAG] = -velocity / weight
        J[Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Vehicle.MASS] = (
            -grav_metric * velocity * (thrust - drag) / (weight) ** 2
        )
