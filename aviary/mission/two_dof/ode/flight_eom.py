import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Dynamic


class EOMRates(om.ExplicitComponent):
    """Computes the distance and altitude rates for the 2dof equations of motion."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='ft/s',
        )

        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            shape=nn,
            units='lbf',
        )
        add_aviary_input(self, Dynamic.Vehicle.DRAG, shape=nn, units='lbf')
        add_aviary_input(
            self,
            Dynamic.Vehicle.MASS,
            shape=nn,
            units='lbm',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            shape=nn,
            units='rad',
        )

        add_aviary_output(
            self,
            Dynamic.Mission.ALTITUDE_RATE,
            shape=nn,
            units='ft/s',
        )
        add_aviary_output(
            self,
            Dynamic.Mission.DISTANCE_RATE,
            shape=nn,
            units='ft/s',
        )
        self.add_output(
            'required_lift',
            shape=nn,
            units='lbf',
            desc='lift required in order to maintain calculated flight path angle',
        )
        add_aviary_output(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, shape=nn, units='rad')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials(
            Dynamic.Mission.ALTITUDE_RATE,
            [
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE,
            [
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'required_lift',
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
            ],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        TAS = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        gamma = np.arcsin((thrust - drag) / weight)

        outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs['required_lift'] = weight * np.cos(gamma) - thrust * np.sin(alpha)
        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE] = gamma

    def compute_partials(self, inputs, J):
        TAS = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        gamma = np.arcsin((thrust - drag) / weight)

        fact = (1 - ((thrust - drag) / weight) ** 2) ** (-0.5)
        dGamma_dThrust = fact / weight
        dGamma_dDrag = -fact / weight
        dGamma_dWeight = fact * (drag - thrust) / weight**2

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            TAS * np.cos(gamma) * dGamma_dThrust
        )
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Vehicle.DRAG] = TAS * np.cos(gamma) * dGamma_dDrag
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Vehicle.MASS] = (
            TAS * np.cos(gamma) * dGamma_dWeight * GRAV_ENGLISH_LBM
        )

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            -TAS * np.sin(gamma) * dGamma_dThrust
        )
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.DRAG] = -TAS * np.sin(gamma) * dGamma_dDrag
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.MASS] = (
            -TAS * np.sin(gamma) * dGamma_dWeight * GRAV_ENGLISH_LBM
        )

        J['required_lift', Dynamic.Vehicle.MASS] = (
            np.cos(gamma) - weight * np.sin(gamma) * dGamma_dWeight
        ) * GRAV_ENGLISH_LBM
        J['required_lift', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = -weight * np.sin(
            gamma
        ) * dGamma_dThrust - np.sin(alpha)
        J['required_lift', Dynamic.Vehicle.DRAG] = -weight * np.sin(gamma) * dGamma_dDrag
        J['required_lift', Dynamic.Vehicle.ANGLE_OF_ATTACK] = -thrust * np.cos(alpha)

        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            dGamma_dThrust
        )
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Vehicle.DRAG] = dGamma_dDrag
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Vehicle.MASS] = (
            dGamma_dWeight * GRAV_ENGLISH_LBM
        )
