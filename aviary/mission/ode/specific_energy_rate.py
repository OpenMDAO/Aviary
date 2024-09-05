import numpy as np
import openmdao.api as om
from aviary.constants import GRAV_METRIC_FLOPS as gravity
from aviary.variable_info.variables import Dynamic


class SpecificEnergyRate(om.ExplicitComponent):
    """
    Rutowski "Energy Approach to the General Aircraft Performance Problem", doi 10.2514/8.2956
    Equation 5
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            Dynamic.Atmosphere.VELOCITY,
            val=np.ones(nn),
            desc='current velocity',
            units='m/s',
        )
        self.add_input(
            Dynamic.Vehicle.MASS,
            val=np.ones(nn),
            desc='current mass',
            units='kg')
        self.add_input(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=np.ones(nn),
                       desc='current thrust', units='N')
        self.add_input(
            Dynamic.Vehicle.DRAG,
            val=np.ones(nn),
            desc='current drag',
            units='N')
        self.add_output(Dynamic.Vehicle.SPECIFIC_ENERGY_RATE, val=np.ones(
            nn), desc='current specific power', units='m/s')

    def compute(self, inputs, outputs):
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * gravity
        outputs[Dynamic.Vehicle.SPECIFIC_ENERGY_RATE] = velocity * \
            (thrust - drag) / weight

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(
            Dynamic.Vehicle.SPECIFIC_ENERGY_RATE,
            [
                Dynamic.Atmosphere.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            rows=arange,
            cols=arange,
        )

    def compute_partials(self, inputs, J):
        velocity = inputs[Dynamic.Atmosphere.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * gravity

        J[Dynamic.Vehicle.SPECIFIC_ENERGY_RATE, Dynamic.Atmosphere.VELOCITY] = (
            thrust - drag
        ) / weight
        J[
            Dynamic.Vehicle.SPECIFIC_ENERGY_RATE,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
        ] = (
            velocity / weight
        )
        J[Dynamic.Vehicle.SPECIFIC_ENERGY_RATE, Dynamic.Vehicle.DRAG] = -velocity / weight
        J[Dynamic.Vehicle.SPECIFIC_ENERGY_RATE, Dynamic.Vehicle.MASS] = -gravity\
            * velocity * (thrust - drag) / (weight)**2
