import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM, MU_TAKEOFF
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class FlightPathEOM(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mu = MU_TAKEOFF

    def initialize(self):
        self.options.declare("num_nodes", types=int)
        self.options.declare("ground_roll", types=bool, default=False,
                             desc="True if the aircraft is confined to the ground. Removes altitude rate as an "
                                  "output and adjust the TAS rate equation.")

    def setup(self):
        nn = self.options["num_nodes"]
        ground_roll = self.options["ground_roll"]

        self.add_input(
            Dynamic.Vehicle.MASS, val=np.ones(nn), desc="aircraft mass", units="lbm"
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            val=np.ones(nn),
            desc=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units="lbf",
        )
        self.add_input(
            Dynamic.Vehicle.LIFT,
            val=np.ones(nn),
            desc=Dynamic.Vehicle.LIFT,
            units="lbf",
        )
        self.add_input(
            Dynamic.Vehicle.DRAG,
            val=np.ones(nn),
            desc=Dynamic.Vehicle.DRAG,
            units="lbf",
        )
        self.add_input(
            Dynamic.Atmosphere.VELOCITY,
            val=np.ones(nn),
            desc="true air speed",
            units="ft/s",
        )
        self.add_input(
            Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
            val=np.ones(nn),
            desc="flight path angle",
            units="rad",
        )

        add_aviary_input(self, Aircraft.Wing.INCIDENCE, val=0)

        self.add_output(
            Dynamic.Atmosphere.VELOCITY_RATE,
            val=np.ones(nn),
            desc="TAS rate",
            units="ft/s**2",
            tags=['dymos.state_rate_source:velocity', 'dymos.state_units:kn'],
        )

        if not ground_roll:
            self._mu = 0.0
            self.add_output(
                Dynamic.Atmosphere.ALTITUDE_RATE,
                val=np.ones(nn),
                desc="altitude rate",
                units="ft/s",
                tags=['dymos.state_rate_source:altitude', 'dymos.state_units:ft'],
            )
            self.add_output(
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE,
                val=np.ones(nn),
                desc="flight path angle rate",
                units="rad/s",
                tags=[
                    'dymos.state_rate_source:flight_path_angle',
                    'dymos.state_units:rad',
                ],
            )
            self.add_input("alpha", val=np.ones(nn), desc="angle of attack", units="deg")

        self.add_output(
            Dynamic.Mission.DISTANCE_RATE,
            val=np.ones(nn),
            desc="distance rate",
            units="ft/s",
            tags=[
                'dymos.state_rate_source:distance',
                'dymos.state_units:ft'])
        self.add_output("normal_force", val=np.ones(
            nn), desc="normal forces", units="lbf")
        self.add_output("fuselage_pitch", val=np.ones(
            nn), desc="fuselage pitch angle", units="deg")
        self.add_output("load_factor", val=np.ones(
            nn), desc="load factor", units="unitless")
        # Possible nice-to-have TODO: alpha as an output for groundroll

    def setup_partials(self):
        arange = np.arange(self.options["num_nodes"], dtype=int)
        ground_roll = self.options["ground_roll"]

        self.declare_partials(
            "load_factor",
            [
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials("load_factor", [Aircraft.Wing.INCIDENCE])

        self.declare_partials(
            Dynamic.Atmosphere.VELOCITY_RATE,
            [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
                Dynamic.Vehicle.LIFT,
            ],
            rows=arange,
            cols=arange,
        )

        self.declare_partials(
            Dynamic.Atmosphere.VELOCITY_RATE, [Aircraft.Wing.INCIDENCE]
        )

        if not ground_roll:
            self.declare_partials(
                Dynamic.Atmosphere.ALTITUDE_RATE,
                [Dynamic.Atmosphere.VELOCITY, Dynamic.Vehicle.FLIGHT_PATH_ANGLE],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE,
                [
                    Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                    "alpha",
                    Dynamic.Vehicle.LIFT,
                    Dynamic.Vehicle.MASS,
                    Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
                    Dynamic.Atmosphere.VELOCITY,
                ],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE, [Aircraft.Wing.INCIDENCE]
            )
            self.declare_partials(
                "normal_force",
                "alpha",
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                "fuselage_pitch", "alpha", rows=arange, cols=arange
            )

            self.declare_partials(
                "load_factor",
                "alpha",
                rows=arange,
                cols=arange,
            )
            self.declare_partials("load_factor", [Aircraft.Wing.INCIDENCE])

            self.declare_partials(
                Dynamic.Atmosphere.VELOCITY_RATE,
                "alpha",
                rows=arange,
                cols=arange,
            )

        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE,
            [Dynamic.Atmosphere.VELOCITY, Dynamic.Vehicle.FLIGHT_PATH_ANGLE],
            rows=arange,
            cols=arange,
        )
        # self.declare_partials("alpha_rate", ["*"], val=0.0)
        self.declare_partials(
            "normal_force",
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials("normal_force", [Aircraft.Wing.INCIDENCE])
        self.declare_partials(
            "fuselage_pitch",
            Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
            rows=arange,
            cols=arange,
            val=180 / np.pi,
        )
        self.declare_partials("fuselage_pitch", [Aircraft.Wing.INCIDENCE])

    def compute(self, inputs, outputs):

        mu = MU_TAKEOFF if self.options['ground_roll'] else 0.0

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Atmosphere.VELOCITY]
        gamma = inputs[Dynamic.Vehicle.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        if self.options["ground_roll"]:
            alpha = inputs[Aircraft.Wing.INCIDENCE]
        else:
            alpha = inputs["alpha"]

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)
        normal_force = weight - incremented_lift - thrust_across_flightpath

        outputs[Dynamic.Atmosphere.VELOCITY_RATE] = (
            (
                thrust_along_flightpath
                - incremented_drag
                - weight * np.sin(gamma)
                - mu * normal_force
            )
            * GRAV_ENGLISH_GASP
            / weight
        )

        if not self.options['ground_roll']:
            outputs[Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE] = (
                (thrust_across_flightpath + incremented_lift - weight * np.cos(gamma))
                * GRAV_ENGLISH_GASP
                / (TAS * weight)
            )
            outputs[Dynamic.Atmosphere.ALTITUDE_RATE] = TAS * np.sin(gamma)

        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)

        outputs["normal_force"] = normal_force

        outputs["fuselage_pitch"] = gamma * 180 / np.pi - i_wing + alpha

        load_factor = (incremented_lift + thrust_across_flightpath) / (
            weight * np.cos(gamma)
        )

        outputs["load_factor"] = load_factor

    def compute_partials(self, inputs, J):
        mu = MU_TAKEOFF if self.options['ground_roll'] else 0.0

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Atmosphere.VELOCITY]
        gamma = inputs[Dynamic.Vehicle.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        if self.options["ground_roll"]:
            alpha = i_wing
        else:
            alpha = inputs["alpha"]

        nn = self.options["num_nodes"]

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)

        dTAlF_dThrust = np.cos((alpha - i_wing) * np.pi / 180)
        dTAlF_dAlpha = -thrust * np.sin((alpha - i_wing) * np.pi / 180) * np.pi / 180
        dTAlF_dIwing = thrust * np.sin((alpha - i_wing) * np.pi / 180) * np.pi / 180

        dTAcF_dThrust = np.sin((alpha - i_wing) * np.pi / 180)
        dTAcF_dAlpha = thrust * np.cos((alpha - i_wing) * np.pi / 180) * np.pi / 180
        dTAcF_dIwing = -thrust * np.cos((alpha - i_wing) * np.pi / 180) * np.pi / 180

        J["load_factor", Dynamic.Vehicle.LIFT] = 1 / (weight * np.cos(gamma))
        J["load_factor", Dynamic.Vehicle.MASS] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight**2 * np.cos(gamma))
            * GRAV_ENGLISH_LBM
        )
        J["load_factor", Dynamic.Vehicle.FLIGHT_PATH_ANGLE] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight * (np.cos(gamma)) ** 2)
            * (-np.sin(gamma))
        )
        J["load_factor", Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dTAcF_dThrust / (
            weight * np.cos(gamma)
        )

        normal_force = weight - incremented_lift - thrust_across_flightpath
        # normal_force = np.where(normal_force1 < 0, np.zeros(nn), normal_force1)

        dNF_dWeight = np.ones(nn)
        # dNF_dWeight[normal_force1 < 0] = 0

        dNF_dLift = -np.ones(nn)
        # dNF_dLift[normal_force1 < 0] = 0

        dNF_dThrust = -np.ones(nn) * dTAcF_dThrust
        # dNF_dThrust[normal_force1 < 0] = 0

        dNF_dIwing = -np.ones(nn) * dTAcF_dIwing
        # dNF_dIwing[normal_force1 < 0] = 0

        J[Dynamic.Atmosphere.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            (dTAlF_dThrust - mu * dNF_dThrust) * GRAV_ENGLISH_GASP / weight
        )

        J[Dynamic.Atmosphere.VELOCITY_RATE, Dynamic.Vehicle.DRAG] = (
            -GRAV_ENGLISH_GASP / weight
        )
        J[Dynamic.Atmosphere.VELOCITY_RATE, Dynamic.Vehicle.MASS] = (
            GRAV_ENGLISH_GASP
            * GRAV_ENGLISH_LBM
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
        J[Dynamic.Atmosphere.VELOCITY_RATE, Dynamic.Vehicle.FLIGHT_PATH_ANGLE] = (
            -np.cos(gamma) * GRAV_ENGLISH_GASP
        )
        J[Dynamic.Atmosphere.VELOCITY_RATE, Dynamic.Vehicle.LIFT] = (
            GRAV_ENGLISH_GASP * (-mu * dNF_dLift) / weight
        )

        # TODO: check partials, esp. for alphas
        if not self.options['ground_roll']:
            J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Atmosphere.VELOCITY] = np.sin(
                gamma
            )
            J[Dynamic.Atmosphere.ALTITUDE_RATE, Dynamic.Vehicle.FLIGHT_PATH_ANGLE] = (
                TAS * np.cos(gamma)
            )

            J[
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ] = (
                dTAcF_dThrust * GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE, "alpha"] = (
                dTAcF_dAlpha * GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE, Aircraft.Wing.INCIDENCE] = (
                dTAcF_dIwing * GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE, Dynamic.Vehicle.LIFT] = (
                GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE, Dynamic.Vehicle.MASS] = (
                (GRAV_ENGLISH_GASP / TAS)
                * GRAV_ENGLISH_LBM
                * (-thrust_across_flightpath / weight**2 - incremented_lift / weight**2)
            )
            J[
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Vehicle.FLIGHT_PATH_ANGLE,
            ] = (
                weight * np.sin(gamma) * GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Vehicle.FLIGHT_PATH_ANGLE_RATE, Dynamic.Atmosphere.VELOCITY] = -(
                (thrust_across_flightpath + incremented_lift - weight * np.cos(gamma))
                * GRAV_ENGLISH_GASP
                / (TAS**2 * weight)
            )

            dNF_dAlpha = -np.ones(nn) * dTAcF_dAlpha
            # dNF_dAlpha[normal_force1 < 0] = 0
            J[Dynamic.Atmosphere.VELOCITY_RATE, "alpha"] = (
                (dTAlF_dAlpha - mu * dNF_dAlpha) * GRAV_ENGLISH_GASP / weight
            )
            J["normal_force", "alpha"] = dNF_dAlpha
            J["fuselage_pitch", "alpha"] = 1
            J["load_factor", "alpha"] = dTAcF_dAlpha / (weight * np.cos(gamma))
            J[Dynamic.Atmosphere.VELOCITY_RATE, Aircraft.Wing.INCIDENCE] = (
                (dTAlF_dIwing - mu * dNF_dIwing) * GRAV_ENGLISH_GASP / weight
            )
            J["normal_force", Aircraft.Wing.INCIDENCE] = dNF_dIwing
            J["fuselage_pitch", Aircraft.Wing.INCIDENCE] = -1
            J["load_factor", Aircraft.Wing.INCIDENCE] = dTAcF_dIwing / \
                (weight * np.cos(gamma))

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Atmosphere.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Vehicle.FLIGHT_PATH_ANGLE] = (
            -TAS * np.sin(gamma)
        )

        J["normal_force", Dynamic.Vehicle.MASS] = dNF_dWeight * GRAV_ENGLISH_LBM
        J["normal_force", Dynamic.Vehicle.LIFT] = dNF_dLift
        J["normal_force", Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dNF_dThrust
