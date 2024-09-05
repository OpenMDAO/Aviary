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
            Dynamic.Atmosphere.VELOCITY,
            val=np.zeros(nn),
            units="ft/s",
            desc="true air speed",
        )

        self.add_input(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=np.zeros(nn),
                       units="lbf", desc="net thrust")
        self.add_input(
            Dynamic.Vehicle.DRAG,
            val=np.zeros(nn),
            units="lbf",
            desc="net drag on aircraft")
        self.add_input(
            Dynamic.Vehicle.MASS,
            val=np.zeros(nn),
            units="lbm",
            desc="mass of aircraft",
        )

        self.add_output(
            Dynamic.Atmosphere.ALTITUDE_RATE,
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
            Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
            val=np.ones(nn),
            units="rad",
            desc="flight path angle",
        )

        self.declare_partials(
            Dynamic.Atmosphere.ALTITUDE_RATE,
            [
                Dynamic.Atmosphere.VELOCITY,
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
                Dynamic.Atmosphere.VELOCITY,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            "required_lift",
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
            [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
            ],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):

        TAS = inputs[Dynamic.Atmosphere.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM

        gamma = np.arcsin((thrust - drag) / weight)

        outputs[Dynamic.Atmosphere.ALTITUDE_RATE] = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs["required_lift"] = weight * np.cos(gamma)
        outputs[Dynamic.Vehicle.FLIGHT_PATH_ANGLE] = gamma

    def compute_partials(self, inputs, J):

        TAS = inputs[Dynamic.Atmosphere.VELOCITY]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM

        gamma = np.arcsin((thrust - drag) / weight)

        dGamma_dThrust = (1 - ((thrust - drag) / weight) ** 2) ** (-0.5) / weight
        dGamma_dDrag = (1 - ((thrust - drag) / weight) ** 2) ** (-0.5) / (-weight)
        dGamma_dWeight = (
            (1 - ((thrust - drag) / weight) ** 2) ** (-0.5)
            * (drag - thrust)
            / weight**2
        )

        J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Atmosphere.VELOCITY] = np.sin(gamma)
        J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            TAS * np.cos(gamma) * dGamma_dThrust
        )
        J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Vehicle.DRAG] = (
            TAS * np.cos(gamma) * dGamma_dDrag
        )
        J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Vehicle.MASS] = (
            TAS * np.cos(gamma) * dGamma_dWeight * GRAV_ENGLISH_LBM
        )

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Atmosphere.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            -TAS * np.sin(gamma) * dGamma_dThrust
        )
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.DRAG] = - \
            TAS * np.sin(gamma) * dGamma_dDrag
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.MASS] = \
            -TAS * np.sin(gamma) * dGamma_dWeight * GRAV_ENGLISH_LBM

        J["required_lift", Dynamic.Vehicle.MASS] = (
            np.cos(gamma) - weight * np.sin(gamma) * dGamma_dWeight
        ) * GRAV_ENGLISH_LBM
        J["required_lift", Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            -weight * np.sin(gamma) * dGamma_dThrust
        )
        J["required_lift", Dynamic.Vehicle.DRAG] = -weight * np.sin(gamma) * dGamma_dDrag

        J[
            Dynamic.Vehicle.FLIGHT_PATH_ANGLE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL
        ] = dGamma_dThrust
        J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE, Dynamic.Vehicle.DRAG] = dGamma_dDrag
        J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
            Dynamic.Vehicle.MASS] = dGamma_dWeight * GRAV_ENGLISH_LBM
