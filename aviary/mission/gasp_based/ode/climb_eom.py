import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.variables import Dynamic


class ClimbRates(om.ExplicitComponent):
    """
    Compute the altitude rate, distance rate, required lift, and flight path angle for
    an aircraft in a climb phase of flight.

    Equations come from climb subroutine of GASP code.
    """

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
        self.declare_partials("required_lift",
                              [Dynamic.Mission.MASS,
                               Dynamic.Mission.THRUST_TOTAL,
                               Dynamic.Mission.DRAG],
                              rows=arange,
                              cols=arange)
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

        gamma = np.arcsin((thrust - drag) / weight)

        outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs["required_lift"] = weight * np.cos(gamma)
        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE] = gamma

    def compute_partials(self, inputs, J):

        TAS = inputs[Dynamic.Mission.VELOCITY]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        drag = inputs[Dynamic.Mission.DRAG]
        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM

        gamma = np.arcsin((thrust - drag) / weight)

        dGamma_dThrust = (1 - ((thrust - drag) / weight) ** 2) ** (-0.5) / weight
        dGamma_dDrag = (1 - ((thrust - drag) / weight) ** 2) ** (-0.5) / (-weight)
        dGamma_dWeight = (
            (1 - ((thrust - drag) / weight) ** 2) ** (-0.5)
            * (drag - thrust)
            / weight**2
        )

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.THRUST_TOTAL] = TAS * \
            np.cos(gamma) * dGamma_dThrust
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.DRAG] = TAS * \
            np.cos(gamma) * dGamma_dDrag
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.MASS] = \
            TAS * np.cos(gamma) * dGamma_dWeight * GRAV_ENGLISH_LBM

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.THRUST_TOTAL] = - \
            TAS * np.sin(gamma) * dGamma_dThrust
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.DRAG] = - \
            TAS * np.sin(gamma) * dGamma_dDrag
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.MASS] = \
            -TAS * np.sin(gamma) * dGamma_dWeight * GRAV_ENGLISH_LBM

        J["required_lift", Dynamic.Mission.MASS] = (
            np.cos(gamma) - weight * np.sin(gamma) * dGamma_dWeight
        ) * GRAV_ENGLISH_LBM
        J["required_lift", Dynamic.Mission.THRUST_TOTAL] = - \
            weight * np.sin(gamma) * dGamma_dThrust
        J["required_lift", Dynamic.Mission.DRAG] = -weight * np.sin(gamma) * dGamma_dDrag

        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.THRUST_TOTAL] = dGamma_dThrust
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.DRAG] = dGamma_dDrag
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE,
            Dynamic.Mission.MASS] = dGamma_dWeight * GRAV_ENGLISH_LBM
