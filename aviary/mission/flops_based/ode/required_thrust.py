import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_METRIC_FLOPS as gravity
from aviary.variable_info.variables import Dynamic


class RequiredThrust(om.ExplicitComponent):
    """
    Computes the required thrust using the equation:
    thrust_required = drag + (altitude_rate*gravity/velocity + velocity_rate) * mass.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(Dynamic.Vehicle.DRAG, val=np.zeros(nn), units='N', desc='drag force')
        self.add_input(
            Dynamic.Mission.ALTITUDE_RATE,
            val=np.zeros(nn),
            units='m/s',
            desc='rate of change of altitude',
        )
        self.add_input(
            Dynamic.Mission.VELOCITY,
            val=np.zeros(nn),
            units='m/s',
            desc=Dynamic.Mission.VELOCITY,
        )
        self.add_input(
            Dynamic.Mission.VELOCITY_RATE,
            val=np.zeros(nn),
            units='m/s**2',
            desc='rate of change of velocity',
        )
        self.add_input(
            Dynamic.Vehicle.MASS, val=np.zeros(nn), units='kg', desc='mass of the aircraft'
        )
        self.add_output('thrust_required', val=np.zeros(nn), units='N', desc='required thrust')

        ar = np.arange(nn)
        self.declare_partials('thrust_required', Dynamic.Vehicle.DRAG, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Mission.ALTITUDE_RATE, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Mission.VELOCITY, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Mission.VELOCITY_RATE, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Vehicle.MASS, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        drag = inputs[Dynamic.Vehicle.DRAG]
        altitude_rate = inputs[Dynamic.Mission.ALTITUDE_RATE]
        velocity = inputs[Dynamic.Mission.VELOCITY]
        velocity_rate = inputs[Dynamic.Mission.VELOCITY_RATE]
        mass = inputs[Dynamic.Vehicle.MASS]

        thrust_required = drag + (altitude_rate * gravity / velocity + velocity_rate) * mass

        outputs['thrust_required'] = thrust_required

    def compute_partials(self, inputs, partials):
        altitude_rate = inputs[Dynamic.Mission.ALTITUDE_RATE]
        velocity = inputs[Dynamic.Mission.VELOCITY]
        velocity_rate = inputs[Dynamic.Mission.VELOCITY_RATE]
        mass = inputs[Dynamic.Vehicle.MASS]

        partials['thrust_required', Dynamic.Vehicle.DRAG] = 1.0
        partials['thrust_required', Dynamic.Mission.ALTITUDE_RATE] = gravity / velocity * mass
        partials['thrust_required', Dynamic.Mission.VELOCITY] = (
            -altitude_rate * gravity / velocity**2 * mass
        )
        partials['thrust_required', Dynamic.Mission.VELOCITY_RATE] = mass
        partials['thrust_required', Dynamic.Vehicle.MASS] = (
            altitude_rate * gravity / velocity + velocity_rate
        )
