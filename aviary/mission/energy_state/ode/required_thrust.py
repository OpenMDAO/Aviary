import numpy as np
import openmdao.api as om
from aviary.variable_info.functions import add_aviary_input, add_aviary_option

from aviary.variable_info.variables import Dynamic, Mission


class RequiredThrust(om.ExplicitComponent):
    """
    Computes the required thrust using the equation:
    thrust_required = drag + (altitude_rate*gravity/velocity + velocity_rate) * mass.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.DRAG, shape=nn, units='N', desc='drag force')
        add_aviary_input(
            self,
            Dynamic.Mission.ALTITUDE_RATE,
            shape=nn,
            units='m/s',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='m/s',
            desc=Dynamic.Mission.VELOCITY,
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY_RATE,
            shape=nn,
            units='m/s**2',
        )
        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')
        self.add_output('thrust_required', shape=nn, units='N')

        ar = np.arange(nn)
        self.declare_partials('thrust_required', Dynamic.Vehicle.DRAG, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Mission.ALTITUDE_RATE, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Mission.VELOCITY, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Mission.VELOCITY_RATE, rows=ar, cols=ar)
        self.declare_partials('thrust_required', Dynamic.Vehicle.MASS, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        grav_metric = self.options[Mission.GRAVITY][0]
        drag = inputs[Dynamic.Vehicle.DRAG]
        altitude_rate = inputs[Dynamic.Mission.ALTITUDE_RATE]
        velocity = inputs[Dynamic.Mission.VELOCITY]
        velocity_rate = inputs[Dynamic.Mission.VELOCITY_RATE]
        mass = inputs[Dynamic.Vehicle.MASS]

        thrust_required = drag + (altitude_rate * grav_metric / velocity + velocity_rate) * mass

        outputs['thrust_required'] = thrust_required

    def compute_partials(self, inputs, partials):
        grav_metric = self.options[Mission.GRAVITY][0]
        altitude_rate = inputs[Dynamic.Mission.ALTITUDE_RATE]
        velocity = inputs[Dynamic.Mission.VELOCITY]
        velocity_rate = inputs[Dynamic.Mission.VELOCITY_RATE]
        mass = inputs[Dynamic.Vehicle.MASS]

        partials['thrust_required', Dynamic.Vehicle.DRAG] = 1.0
        partials['thrust_required', Dynamic.Mission.ALTITUDE_RATE] = grav_metric / velocity * mass
        partials['thrust_required', Dynamic.Mission.VELOCITY] = (
            -altitude_rate * grav_metric / velocity**2 * mass
        )
        partials['thrust_required', Dynamic.Mission.VELOCITY_RATE] = mass
        partials['thrust_required', Dynamic.Vehicle.MASS] = (
            altitude_rate * grav_metric / velocity + velocity_rate
        )
