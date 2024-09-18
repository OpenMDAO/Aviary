import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM, MU_TAKEOFF
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class AscentEOM(om.ExplicitComponent):
    """
    Ascent equation of motion: Compute the velocity rate, flight path angle rate, 
    altitude rate, distance rate, normal forces, fuselage pitch angle, load factor
    and angle of attack rate.
    """

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_input(Dynamic.Mission.MASS, val=np.ones(nn),
                       desc="aircraft mass", units="lbm")
        self.add_input(Dynamic.Mission.THRUST_TOTAL, val=np.ones(
            nn), desc=Dynamic.Mission.THRUST_TOTAL, units="lbf")
        self.add_input(
            Dynamic.Mission.LIFT,
            val=np.ones(nn),
            desc=Dynamic.Mission.LIFT,
            units="lbf")
        self.add_input(
            Dynamic.Mission.DRAG,
            val=np.ones(nn),
            desc=Dynamic.Mission.DRAG,
            units="lbf")
        self.add_input(Dynamic.Mission.VELOCITY, val=np.ones(nn),
                       desc="Velocity", units="ft/s")
        self.add_input(Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.ones(nn),
                       desc="flight path angle", units="rad")
        add_aviary_input(self, Aircraft.Wing.INCIDENCE, val=0, units="deg")

        self.add_input("alpha", val=np.ones(nn), desc="angle of attack", units="deg")

        self.add_output(Dynamic.Mission.VELOCITY_RATE, val=np.ones(nn),
                        desc="Velocity rate", units="ft/s**2")
        self.add_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, val=np.ones(nn), desc="flight path angle rate", units="rad/s"
        )
        self.add_output(
            Dynamic.Mission.ALTITUDE_RATE,
            val=np.ones(nn),
            desc="altitude rate",
            units="ft/s")
        self.add_output(
            Dynamic.Mission.DISTANCE_RATE, val=np.ones(nn), desc="distance rate", units="ft/s"
        )
        self.add_output(
            "normal_force", val=np.ones(nn), desc="normal forces", units="lbf"
        )
        self.add_output(
            "fuselage_pitch", val=np.ones(nn), desc="fuselage pitch angle", units="deg"
        )
        self.add_output(
            "load_factor", val=np.ones(nn), desc="load factor", units="unitless")
        self.add_output(
            "alpha_rate", val=np.ones(nn), desc="angle of attack rate", units="deg/s"
        )

    def setup_partials(self):
        arange = np.arange(self.options["num_nodes"])

        self.declare_partials(
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            [Dynamic.Mission.THRUST_TOTAL, "alpha",
                Dynamic.Mission.LIFT, Dynamic.Mission.MASS, Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.VELOCITY],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, [
                              Aircraft.Wing.INCIDENCE])
        self.declare_partials(
            "load_factor",
            [Dynamic.Mission.LIFT, Dynamic.Mission.MASS,
                Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.THRUST_TOTAL, "alpha"],
            rows=arange,
            cols=arange,
        )
        self.declare_partials("load_factor", [Aircraft.Wing.INCIDENCE])

        self.declare_partials(
            Dynamic.Mission.VELOCITY_RATE,
            [Dynamic.Mission.THRUST_TOTAL, "alpha", Dynamic.Mission.DRAG,
                Dynamic.Mission.MASS, Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.LIFT],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(Dynamic.Mission.VELOCITY_RATE, [Aircraft.Wing.INCIDENCE])
        self.declare_partials(
            Dynamic.Mission.ALTITUDE_RATE, [
                Dynamic.Mission.VELOCITY, Dynamic.Mission.FLIGHT_PATH_ANGLE], rows=arange, cols=arange)
        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE, [
                Dynamic.Mission.VELOCITY, Dynamic.Mission.FLIGHT_PATH_ANGLE], rows=arange, cols=arange
        )
        self.declare_partials(
            "normal_force",
            [Dynamic.Mission.MASS, Dynamic.Mission.LIFT,
                Dynamic.Mission.THRUST_TOTAL, "alpha"],
            rows=arange,
            cols=arange,
        )
        self.declare_partials("normal_force", [Aircraft.Wing.INCIDENCE])
        self.declare_partials(
            "fuselage_pitch", Dynamic.Mission.FLIGHT_PATH_ANGLE, rows=arange, cols=arange, val=180 / np.pi,
        )
        self.declare_partials("fuselage_pitch", "alpha", rows=arange, cols=arange, val=1)
        self.declare_partials("fuselage_pitch", Aircraft.Wing.INCIDENCE, val=-1)

    def compute(self, inputs, outputs):
        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Mission.LIFT]
        incremented_drag = inputs[Dynamic.Mission.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        alpha = inputs["alpha"]

        mu = 0.0

        nn = self.options["num_nodes"]

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)
        normal_force = weight - incremented_lift - thrust_across_flightpath
        normal_force[normal_force < 0] = 0.0

        outputs[Dynamic.Mission.VELOCITY_RATE] = (
            (
                thrust_along_flightpath
                - incremented_drag
                - weight * np.sin(gamma)
                - mu * normal_force
            )
            * GRAV_ENGLISH_GASP
            / weight
        )

        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE] = (
            (thrust_across_flightpath + incremented_lift - weight * np.cos(gamma))
            * GRAV_ENGLISH_GASP
            / (TAS * weight)
        )

        outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs["normal_force"] = normal_force
        outputs["fuselage_pitch"] = gamma * 180 / np.pi - i_wing + alpha

        outputs["alpha_rate"] = np.zeros(nn)

        load_factor = (incremented_lift + thrust_across_flightpath) / (
            weight * np.cos(gamma)
        )

        outputs["load_factor"] = load_factor

    def compute_partials(self, inputs, J):
        nn = self.options["num_nodes"]

        weight = inputs[Dynamic.Mission.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Mission.LIFT]
        incremented_drag = inputs[Dynamic.Mission.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        alpha = inputs["alpha"]

        mu = MU_TAKEOFF
        mu = 0.0

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)

        dTAlF_dThrust = np.cos((alpha - i_wing) * np.pi / 180)
        dTAlF_dAlpha = -thrust * np.sin((alpha - i_wing) * np.pi / 180) * np.pi / 180
        dTAlF_dIwing = thrust * np.sin((alpha - i_wing) * np.pi / 180) * np.pi / 180

        dTAcF_dThrust = np.sin((alpha - i_wing) * np.pi / 180)
        dTAcF_dAlpha = thrust * np.cos((alpha - i_wing) * np.pi / 180) * np.pi / 180
        dTAcF_dIwing = -thrust * np.cos((alpha - i_wing) * np.pi / 180) * np.pi / 180

        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.THRUST_TOTAL] = dTAcF_dThrust * \
            GRAV_ENGLISH_GASP / (TAS * weight)
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, "alpha"] = dTAcF_dAlpha * \
            GRAV_ENGLISH_GASP / (TAS * weight)
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Aircraft.Wing.INCIDENCE] = (
            dTAcF_dIwing * GRAV_ENGLISH_GASP / (TAS * weight)
        )
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            Dynamic.Mission.LIFT] = GRAV_ENGLISH_GASP / (TAS * weight)
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.MASS] = (GRAV_ENGLISH_GASP / TAS) * GRAV_ENGLISH_LBM * (
            -thrust_across_flightpath / weight**2 - incremented_lift / weight**2
        )
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            weight * np.sin(gamma) * GRAV_ENGLISH_GASP / (TAS * weight)
        )
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.VELOCITY] = -(
            (thrust_across_flightpath + incremented_lift - weight * np.cos(gamma))
            * GRAV_ENGLISH_GASP
            / (TAS**2 * weight)
        )

        J["load_factor", Dynamic.Mission.LIFT] = 1 / (weight * np.cos(gamma))
        J["load_factor", Dynamic.Mission.MASS] = -(incremented_lift + thrust_across_flightpath) / (
            weight**2 * np.cos(gamma)
        ) * GRAV_ENGLISH_LBM
        J["load_factor", Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight * (np.cos(gamma)) ** 2)
            * (-np.sin(gamma))
        )
        J["load_factor", Dynamic.Mission.THRUST_TOTAL] = dTAcF_dThrust / \
            (weight * np.cos(gamma))
        J["load_factor", "alpha"] = dTAcF_dAlpha / (weight * np.cos(gamma))
        J["load_factor", Aircraft.Wing.INCIDENCE] = dTAcF_dIwing / (
            weight * np.cos(gamma)
        )

        normal_force1 = weight - incremented_lift - thrust_across_flightpath
        normal_force = np.where(normal_force1 < 0, np.zeros(nn), normal_force1)

        dNF_dWeight = np.ones(nn)
        dNF_dWeight[normal_force1 < 0] = 0

        dNF_dLift = -np.ones(nn)
        dNF_dLift[normal_force1 < 0] = 0

        dNF_dThrust = -np.ones(nn) * dTAcF_dThrust
        dNF_dThrust[normal_force1 < 0] = 0

        dNF_dAlpha = -np.ones(nn) * dTAcF_dAlpha
        dNF_dAlpha[normal_force1 < 0] = 0

        dNF_dIwing = -np.ones(nn) * dTAcF_dIwing
        dNF_dIwing[normal_force1 < 0] = 0

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.THRUST_TOTAL] = (
            (dTAlF_dThrust - mu * dNF_dThrust) * GRAV_ENGLISH_GASP / weight
        )
        J[Dynamic.Mission.VELOCITY_RATE, "alpha"] = (
            (dTAlF_dAlpha - mu * dNF_dAlpha) * GRAV_ENGLISH_GASP / weight
        )
        J[Dynamic.Mission.VELOCITY_RATE, Aircraft.Wing.INCIDENCE] = (
            (dTAlF_dIwing - mu * dNF_dIwing) * GRAV_ENGLISH_GASP / weight
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.DRAG] = -GRAV_ENGLISH_GASP / weight
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.MASS] = (
            GRAV_ENGLISH_GASP * GRAV_ENGLISH_LBM
            * (
                weight * (-np.sin(gamma) - mu * dNF_dWeight)
                - (
                    thrust_along_flightpath
                    - incremented_drag
                    - weight * np.sin(gamma)
                    - mu * normal_force
                )
            )
            / weight**2
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = - \
            np.cos(gamma) * GRAV_ENGLISH_GASP
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.LIFT] = GRAV_ENGLISH_GASP * \
            (-mu * dNF_dLift) / weight

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
        J[Dynamic.Mission.ALTITUDE_RATE,
            Dynamic.Mission.FLIGHT_PATH_ANGLE] = TAS * np.cos(gamma)

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE,
            Dynamic.Mission.FLIGHT_PATH_ANGLE] = -TAS * np.sin(gamma)

        J["normal_force", Dynamic.Mission.MASS] = dNF_dWeight * GRAV_ENGLISH_LBM
        J["normal_force", Dynamic.Mission.LIFT] = dNF_dLift
        J["normal_force", Dynamic.Mission.THRUST_TOTAL] = dNF_dThrust
        J["normal_force", "alpha"] = dNF_dAlpha
        J["normal_force", Aircraft.Wing.INCIDENCE] = dNF_dIwing
