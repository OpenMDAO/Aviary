import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM
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
        arange = np.arange(nn)

        self.add_input(
            Dynamic.Vehicle.MASS,
            val=np.ones(nn) * 1e6,
            units='lbm',
            desc='total mass of the aircraft',
        )
        self.add_input(
            Dynamic.Vehicle.DRAG,
            val=np.zeros(nn),
            units='lbf',
            desc='drag on aircraft',
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            val=np.zeros(nn),
            units='lbf',
            desc='total thrust',
        )
        self.add_input(
            Dynamic.Mission.VELOCITY,
            val=np.zeros(nn),
            units='ft/s',
            desc='true air speed',
        )

        self.add_output(
            Dynamic.Mission.VELOCITY_RATE,
            val=np.zeros(nn),
            units='ft/s**2',
            desc='rate of change of true air speed',
        )
        self.add_output(
            Dynamic.Mission.DISTANCE_RATE,
            val=np.zeros(nn),
            units='ft/s',
            desc='rate of change of horizontal distance covered',
        )

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
