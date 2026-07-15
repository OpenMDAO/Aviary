import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Dynamic


class AccelerationRates(om.ExplicitComponent):
    """
    Compute the TAS rate, distance rate, and mass flow rate for a level flight acceleration phase.

    Equation comes from climb subroutine of GASP code.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(
            self,
            Dynamic.Vehicle.MASS,
            shape=nn,
            units='lbm',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.DRAG,
            shape=nn,
            units='lbf',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            shape=nn,
            units='lbf',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='ft/s',
        )

        self.add_output(
            Dynamic.Mission.VELOCITY_RATE,
            shape=nn,
            units='ft/s**2',
        )
        add_aviary_output(
            self,
            Dynamic.Mission.DISTANCE_RATE,
            shape=nn,
            units='ft/s',
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
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        drag = inputs[Dynamic.Vehicle.DRAG]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        TAS = inputs[Dynamic.Mission.VELOCITY]

        outputs[Dynamic.Mission.VELOCITY_RATE] = (GRAV_ENGLISH_GASP / weight) * (thrust - drag)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS

    def compute_partials(self, inputs, J):
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        drag = inputs[Dynamic.Vehicle.DRAG]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.MASS] = (
            -(GRAV_ENGLISH_GASP / weight**2) * (thrust - drag) * GRAV_ENGLISH_LBM
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.DRAG] = -(GRAV_ENGLISH_GASP / weight)
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            GRAV_ENGLISH_GASP / weight
        )
