import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM, MU_TAKEOFF
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic

DEG2RAD = np.pi / 180.0
RAD2DEG = 1.0 / DEG2RAD


class TakeoffEOM(om.ExplicitComponent):
    """
    2-degree of freedom EOM for takeoff phases.

    Compute the rates for the velocity, altitude, flight path angle, and angle of attack states.
    This can be used for the groundroll, rotation, and ascent phases. The angle of attack rate
    is fixed in this phase, and is set to the value in the "rotation_pitch_rate" option, which is
    3.33 degrees/second when "rotation" is True. Otherwise, the angle of attack is fixed for the
    duration of the phase.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._mu = MU_TAKEOFF

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare(
            'ground_roll',
            types=bool,
            default=False,
            desc='True if the aircraft is confined to the ground. Removes altitude rate as an '
            'output and adjusts the TAS rate equation.',
        )

        self.options.declare(
            'rotation',
            types=bool,
            default=False,
            desc='True if the aircraft is pitching up, but the rear wheels are still on the '
            'ground.',
        )

        # TODO: Make this a hiearchy variable with a default.
        self.options.declare(
            'rotation_pitch_rate',
            types=float,
            default=3.33,
            desc='Pitch rate during rotation in degrees/second.'
        )

    def setup(self):
        nn = self.options['num_nodes']
        ground_roll = self.options['ground_roll']

        self.add_input(Dynamic.Vehicle.MASS, val=np.ones(nn), desc='aircraft mass', units='lbm')
        self.add_input(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            val=np.ones(nn),
            desc=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )
        self.add_input(
            Dynamic.Vehicle.LIFT,
            val=np.ones(nn),
            desc=Dynamic.Vehicle.LIFT,
            units='lbf',
        )
        self.add_input(
            Dynamic.Vehicle.DRAG,
            val=np.ones(nn),
            desc=Dynamic.Vehicle.DRAG,
            units='lbf',
        )
        self.add_input(
            Dynamic.Mission.VELOCITY,
            val=np.ones(nn),
            desc='true air speed',
            units='ft/s',
        )
        self.add_input(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            val=np.ones(nn),
            desc='flight path angle',
            units='rad',
        )

        add_aviary_input(self, Aircraft.Wing.INCIDENCE, val=0)

        self.add_output(
            Dynamic.Mission.VELOCITY_RATE,
            val=np.ones(nn),
            desc='TAS rate',
            units='ft/s**2',
        )

        self.add_output(
            Dynamic.Mission.ALTITUDE_RATE,
            val=np.ones(nn),
            desc='altitude rate',
            units='ft/s',
        )

        self.add_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            val=np.ones(nn),
            desc='flight path angle rate',
            units='rad/s',
        )

        if not ground_roll:
            self.add_input(
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                val=np.ones(nn),
                desc='angle of attack',
                units='deg',
            )

        self.add_output(
            Dynamic.Mission.DISTANCE_RATE,
            val=np.ones(nn),
            desc='distance rate',
            units='ft/s',
        )

        self.add_output('normal_force', val=np.ones(nn), desc='normal forces', units='lbf')
        self.add_output('fuselage_pitch', val=np.ones(nn), desc='fuselage pitch angle', units='deg')
        self.add_output('load_factor', val=np.ones(nn), desc='load factor', units='unitless')
        self.add_output(
            'angle_of_attack_rate', val=np.ones(nn), desc='angle of attack rate', units='deg/s'
        )

    def setup_partials(self):
        ground_roll = self.options['ground_roll']

        arange = np.arange(self.options['num_nodes'], dtype=int)

        self.declare_partials(
            'load_factor',
            [
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials('load_factor', [Aircraft.Wing.INCIDENCE])

        self.declare_partials(
            Dynamic.Mission.VELOCITY_RATE,
            [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Vehicle.LIFT,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(Dynamic.Mission.VELOCITY_RATE, [Aircraft.Wing.INCIDENCE])

        self.declare_partials(
            Dynamic.Mission.ALTITUDE_RATE,
            [Dynamic.Mission.VELOCITY, Dynamic.Mission.FLIGHT_PATH_ANGLE],
            rows=arange,
            cols=arange,
        )

        self.declare_partials(
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, [Aircraft.Wing.INCIDENCE])

        if not ground_roll:
            self.declare_partials(
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                rows=arange,
                cols=arange,
            )

            self.declare_partials(
                'normal_force',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                'fuselage_pitch',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                rows=arange,
                cols=arange,
            )

            self.declare_partials(
                'load_factor',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                rows=arange,
                cols=arange,
            )
            self.declare_partials(
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                rows=arange,
                cols=arange,
            )

        self.declare_partials(
            Dynamic.Mission.DISTANCE_RATE,
            [Dynamic.Mission.VELOCITY, Dynamic.Mission.FLIGHT_PATH_ANGLE],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'normal_force',
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials('normal_force', [Aircraft.Wing.INCIDENCE])

        self.declare_partials(
            'fuselage_pitch',
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            rows=arange,
            cols=arange,
            val=RAD2DEG,
        )
        self.declare_partials('fuselage_pitch', [Aircraft.Wing.INCIDENCE])

        self.declare_partials('angle_of_attack_rate', ['*'], 0.0)

    def compute(self, inputs, outputs):
        ground_roll = self.options['ground_roll']
        rotation = self.options['rotation']

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]

        if ground_roll:
            alpha = inputs[Aircraft.Wing.INCIDENCE]
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * DEG2RAD)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * DEG2RAD)

        if ground_roll or rotation:
            mu = MU_TAKEOFF
            normal_force = weight - incremented_lift - thrust_across_flightpath
            normal_force[normal_force < 0] = 0.0
            outputs['normal_force'] = normal_force

        else:
            mu = 0.0
            normal_force = 0.0
            outputs['normal_force'][:] = normal_force

        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        outputs[Dynamic.Mission.VELOCITY_RATE] = (
            (
                thrust_along_flightpath
                - incremented_drag
                - weight * sin_gamma
                - mu * normal_force
            )
            * GRAV_ENGLISH_GASP
            / weight
        )

        if ground_roll or rotation:
            outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE][:] = 0.0
        else:
            outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE] = (
                (thrust_across_flightpath + incremented_lift - weight * cos_gamma)
                * GRAV_ENGLISH_GASP
                / (TAS * weight)
            )

        outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * sin_gamma
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * cos_gamma

        outputs['fuselage_pitch'] = gamma * RAD2DEG - i_wing + alpha

        load_factor = (incremented_lift + thrust_across_flightpath) / (weight * cos_gamma)
        outputs['load_factor'] = load_factor

        if rotation:
            outputs['angle_of_attack_rate'][:] = self.options['rotation_pitch_rate']
        else:
            outputs['angle_of_attack_rate'][:] = 0.0

    def compute_partials(self, inputs, J):
        ground_roll = self.options['ground_roll']
        rotation = self.options['rotation']
        nn = self.options['num_nodes']

        mu = MU_TAKEOFF if (ground_roll or rotation) else 0.0

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]

        if ground_roll:
            alpha = i_wing
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        cos_alpha = np.cos((alpha - i_wing) * DEG2RAD)
        sin_alpha = np.sin((alpha - i_wing) * DEG2RAD)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        thrust_along_flightpath = thrust * cos_alpha
        thrust_across_flightpath = thrust * sin_alpha

        dTAlF_dThrust = cos_alpha
        dTAlF_dAlpha = -thrust * sin_alpha * DEG2RAD
        dTAlF_dIwing = thrust * sin_alpha * DEG2RAD

        dTAcF_dThrust = sin_alpha
        dTAcF_dAlpha = thrust * cos_alpha * DEG2RAD
        dTAcF_dIwing = -thrust * cos_alpha * DEG2RAD

        J['load_factor', Dynamic.Vehicle.LIFT] = 1 / (weight * cos_gamma)
        J['load_factor', Dynamic.Vehicle.MASS] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight**2 * cos_gamma)
            * GRAV_ENGLISH_LBM
        )
        J['load_factor', Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight * (cos_gamma) ** 2)
            * (-sin_gamma)
        )
        J['load_factor', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dTAcF_dThrust / (
            weight * cos_gamma
        )

        normal_force = weight - incremented_lift - thrust_across_flightpath
        idx = np.where(normal_force < 0)
        normal_force[idx] = 0.0

        dNF_dWeight = np.ones(nn, dtype=TAS.dtype)
        dNF_dWeight[idx] = 0

        dNF_dLift = -np.ones(nn, dtype=TAS.dtype)
        dNF_dLift[idx] = 0

        dNF_dThrust = -np.ones(nn, dtype=TAS.dtype) * dTAcF_dThrust
        dNF_dThrust[idx] = 0

        dNF_dIwing = -np.ones(nn, dtype=TAS.dtype) * dTAcF_dIwing
        dNF_dIwing[idx] = 0

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            (dTAlF_dThrust - mu * dNF_dThrust) * GRAV_ENGLISH_GASP / weight
        )

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.DRAG] = -GRAV_ENGLISH_GASP / weight
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.MASS] = (
            GRAV_ENGLISH_GASP
            * GRAV_ENGLISH_LBM
            * (
                weight * (-sin_gamma - mu * dNF_dWeight)
                - (
                    thrust_along_flightpath
                    - incremented_drag
                    - weight * sin_gamma
                    - mu * normal_force
                )
            )
            / weight**2
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -cos_gamma * GRAV_ENGLISH_GASP
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.LIFT] = (
            GRAV_ENGLISH_GASP * (-mu * dNF_dLift) / weight
        )

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = sin_gamma
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = TAS * cos_gamma

        if not (ground_roll or rotation):
            # OF flight path angle rate
            J[
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            ] = dTAcF_dThrust * GRAV_ENGLISH_GASP / (TAS * weight)
            J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
                dTAcF_dAlpha * GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Aircraft.Wing.INCIDENCE] = (
                dTAcF_dIwing * GRAV_ENGLISH_GASP / (TAS * weight)
            )
            J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Vehicle.LIFT] = GRAV_ENGLISH_GASP / (
                TAS * weight
            )
            J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Vehicle.MASS] = (
                (GRAV_ENGLISH_GASP / TAS)
                * GRAV_ENGLISH_LBM
                * (-thrust_across_flightpath / weight**2 - incremented_lift / weight**2)
            )
            J[
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ] = weight * sin_gamma * GRAV_ENGLISH_GASP / (TAS * weight)
            J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.VELOCITY] = -(
                (thrust_across_flightpath + incremented_lift - weight * cos_gamma)
                * GRAV_ENGLISH_GASP
                / (TAS**2 * weight)
            )

        if not ground_roll:
            # WRT incidence angle
            J[Dynamic.Mission.VELOCITY_RATE, Aircraft.Wing.INCIDENCE] = (
                (dTAlF_dIwing - mu * dNF_dIwing) * GRAV_ENGLISH_GASP / weight
            )
            J['normal_force', Aircraft.Wing.INCIDENCE] = dNF_dIwing
            J['fuselage_pitch', Aircraft.Wing.INCIDENCE] = -1
            J['load_factor', Aircraft.Wing.INCIDENCE] = dTAcF_dIwing / (weight * cos_gamma)

            # WRT angle of attack
            dNF_dAlpha = -np.ones(nn, dtype=TAS.dtype) * dTAcF_dAlpha
            dNF_dAlpha[idx] = 0

            J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
                (dTAlF_dAlpha - mu * dNF_dAlpha) * GRAV_ENGLISH_GASP / weight
            )
            J['normal_force', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dNF_dAlpha
            J['fuselage_pitch', Dynamic.Vehicle.ANGLE_OF_ATTACK] = 1
            J['load_factor', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dTAcF_dAlpha / (
                weight * cos_gamma
            )

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = cos_gamma
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -TAS * sin_gamma

        if ground_roll or rotation:
            J['normal_force', Dynamic.Vehicle.MASS] = dNF_dWeight * GRAV_ENGLISH_LBM
            J['normal_force', Dynamic.Vehicle.LIFT] = dNF_dLift
            J['normal_force', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dNF_dThrust
