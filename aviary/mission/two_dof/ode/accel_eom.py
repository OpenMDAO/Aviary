import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Dynamic, Mission


class AccelerationRates(om.ExplicitComponent):
    """
    Compute the TAS rate, distance rate, and mass flow rate for a level flight acceleration phase.

    Equation comes from climb subroutine of GASP code.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(
            self,
            Dynamic.Vehicle.MASS,
            shape=nn,
            units='kg',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.DRAG,
            shape=nn,
            units='N',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            shape=nn,
            units='N',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='m/s',
        )

        self.add_output(
            Dynamic.Mission.VELOCITY_RATE,
            shape=nn,
            units='m/s**2',
        )
        add_aviary_output(
            self,
            Dynamic.Mission.DISTANCE_RATE,
            shape=nn,
            units='m/s',
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials(
            Dynamic.Mission.VELOCITY_RATE,
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE,
            [Dynamic.Mission.VELOCITY],
            rows=arange,
            cols=arange,
            val=1.0,
        )

    def compute(self, inputs, outputs):
        mass = inputs[Dynamic.Vehicle.MASS]
        drag = inputs[Dynamic.Vehicle.DRAG]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        TAS = inputs[Dynamic.Mission.VELOCITY]

        outputs[Dynamic.Mission.VELOCITY_RATE] = (thrust - drag) / mass
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS

    def compute_partials(self, inputs, J):
        mass = inputs[Dynamic.Vehicle.MASS]
        drag = inputs[Dynamic.Vehicle.DRAG]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.MASS] = -(thrust - drag) / (mass**2)
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.DRAG] = -1 / mass
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = 1 / mass
