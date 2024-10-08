import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.variables import Dynamic


class DescentRates(om.ExplicitComponent):
    """Descent rate equations of motion."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]
        arange = np.arange(nn)

        self.add_input(
            Dynamic.Mission.VELOCITY,
            val=np.zeros(nn),
            units="ft/s",
            desc="true air speed",
        )

        self.add_input(Dynamic.Mission.THRUST_TOTAL, val=np.zeros(nn),
                       units="lbf", desc="net thrust")
        self.add_input(
            Dynamic.Mission.DRAG,
            val=np.zeros(nn),
            units="lbf",
            desc="net drag on aircraft")
        self.add_input(
            Dynamic.Mission.MASS,
            val=np.zeros(nn),
            units="lbm",
            desc="mass of aircraft",
        )
        self.add_input(
            "alpha",
            val=np.ones(nn),
            units="rad",
            desc="angle of attack of aircraft",
        )

        self.add_output(
            Dynamic.Mission.ALTITUDE_RATE,
            val=np.zeros(nn),
            units="ft/s",
            desc="rate of change of altitude",
        )
        self.add_output(
            Dynamic.Mission.DISTANCE_RATE,
            val=np.zeros(nn),
            units="ft/s",
            desc="rate of change of horizontal distance covered",
        )
        self.add_output(
            "required_lift",
            val=np.zeros(nn),
            units="lbf",
            desc="lift required in order to maintain calculated flight path angle",
        )
        self.add_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            val=np.ones(nn),
            units="rad",
            desc="flight path angle",
        )

        self.declare_partials(Dynamic.Mission.ALTITUDE_RATE,
                              [Dynamic.Mission.VELOCITY,
                               Dynamic.Mission.THRUST_TOTAL,
                               Dynamic.Mission.DRAG,
                               Dynamic.Mission.MASS],
                              rows=arange,
                              cols=arange)
        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE,
            [Dynamic.Mission.VELOCITY, Dynamic.Mission.THRUST_TOTAL,
                Dynamic.Mission.DRAG, Dynamic.Mission.MASS],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            "required_lift",
            [Dynamic.Mission.MASS, Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.DRAG, "alpha"],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                              [Dynamic.Mission.THRUST_TOTAL,
                               Dynamic.Mission.DRAG,
                               Dynamic.Mission.MASS],
                              rows=arange,
                              cols=arange)

    def compute(self, inputs, outputs):

        TAS = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        drag = inputs[Dynamic.Mission.DRAG]
        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM
        alpha = inputs["alpha"]

        gamma = (thrust - drag) / weight

        outputs[Dynamic.Mission.ALTITUDE_RATE] = alt_rate = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs["required_lift"] = weight * np.cos(gamma) - thrust * np.sin(alpha)
        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE] = gamma

    def compute_partials(self, inputs, J):

        TAS = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        drag = inputs[Dynamic.Mission.DRAG]
        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM
        alpha = inputs["alpha"]

        gamma = (thrust - drag) / weight

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.THRUST_TOTAL] = TAS * \
            np.cos(gamma) / weight
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.DRAG] = TAS * \
            np.cos(gamma) * (-1 / weight)
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.MASS] = TAS * \
            np.cos(gamma) * (-(thrust - drag) / weight**2) * GRAV_ENGLISH_LBM

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.THRUST_TOTAL] = - \
            TAS * np.sin(gamma) / weight
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.DRAG] = - \
            TAS * np.sin(gamma) * (-1 / weight)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.MASS] = (
            -TAS * np.sin(gamma) * (-(thrust - drag) / weight**2) * GRAV_ENGLISH_LBM
        )

        J["required_lift", Dynamic.Mission.MASS] = (
            np.cos(gamma) - weight * np.sin(
                (thrust - drag) / weight
            ) * (-(thrust - drag) / weight**2)
        ) * GRAV_ENGLISH_LBM
        J["required_lift", Dynamic.Mission.THRUST_TOTAL] = - \
            weight * np.sin(gamma) / weight - np.sin(alpha)
        J["required_lift", Dynamic.Mission.DRAG] = - \
            weight * np.sin(gamma) * (-1 / weight)
        J["required_lift", "alpha"] = -thrust * np.cos(alpha)

        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.THRUST_TOTAL] = 1 / weight
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.DRAG] = -1 / weight
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.MASS] = - \
            (thrust - drag) / weight**2 * GRAV_ENGLISH_LBM
