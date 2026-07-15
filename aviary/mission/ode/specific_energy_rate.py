import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_METRIC_FLOPS as gravity
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Dynamic


class SpecificEnergyRate(om.ExplicitComponent):
    """
    Rutowski "Energy Approach to the General Aircraft Performance Problem", doi 10.2514/8.2956
    Equation 5.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

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

    def compute(self, inputs, outputs):
        velocity = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * gravity
        outputs[Dynamic.Mission.SPECIFIC_ENERGY_RATE] = velocity * (thrust - drag) / weight

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

    def compute_partials(self, inputs, J):
        velocity = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * gravity

        J[Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Mission.VELOCITY] = (thrust - drag) / weight
        J[
            Dynamic.Mission.SPECIFIC_ENERGY_RATE,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
        ] = velocity / weight
        J[Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Vehicle.DRAG] = -velocity / weight
        J[Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Vehicle.MASS] = (
            -gravity * velocity * (thrust - drag) / (weight) ** 2
        )
