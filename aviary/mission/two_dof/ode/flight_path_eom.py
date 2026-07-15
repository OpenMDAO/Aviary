import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class FlightPathEOM(om.ExplicitComponent):
    """2-degrees-of-freedom flight path EOM."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare(
            'ground_roll',
            types=bool,
            default=False,
            desc='True if the aircraft is confined to the ground. Removes altitude rate as an '
            'output and adjust the TAS rate equation.',
        )

    def setup(self):
        nn = self.options['num_nodes']
        ground_roll = self.options['ground_roll']

        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=(nn), units='lbm')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.THRUST_TOTAL, shape=(nn), units='lbf')
        add_aviary_input(self, Dynamic.Vehicle.LIFT, shape=(nn), units='lbf')
        add_aviary_input(self, Dynamic.Vehicle.DRAG, shape=(nn), units='lbf')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, shape=(nn), units='ft/s')
        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, shape=(nn), units='rad')
        add_aviary_input(self, Aircraft.Wing.INCIDENCE, units='deg')
        add_aviary_input(self, Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, units='unitless')

        add_aviary_output(
            self,
            Dynamic.Mission.VELOCITY_RATE,
            shape=(nn),
            units='ft/s**2',
            tags=['dymos.state_rate_source:velocity', 'dymos.state_units:kn'],
        )

        if not ground_roll:
            add_aviary_output(
                self,
                Dynamic.Mission.ALTITUDE_RATE,
                shape=(nn),
                units='ft/s',
                # tags=['dymos.state_rate_source:altitude', 'dymos.state_units:ft'],
            )
            add_aviary_output(
                self,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
                shape=(nn),
                units='rad/s',
                tags=[
                    'dymos.state_rate_source:flight_path_angle',
                    'dymos.state_units:rad',
                ],
            )
            add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=(nn), units='deg')

        add_aviary_output(
            self,
            Dynamic.Mission.DISTANCE_RATE,
            shape=(nn),
            units='ft/s',
            tags=['dymos.state_rate_source:distance', 'dymos.state_units:ft'],
        )
        self.add_output('normal_force', val=np.ones(nn), desc='normal forces', units='lbf')
        self.add_output('fuselage_pitch', val=np.ones(nn), desc='fuselage pitch angle', units='deg')
        self.add_output('load_factor', val=np.ones(nn), desc='load factor', units='unitless')
        # Possible nice-to-have TODO: alpha as an output for groundroll

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'], dtype=int)
        ground_roll = self.options['ground_roll']

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
        if ground_roll:
            self.declare_partials(
                Dynamic.Mission.VELOCITY_RATE, [Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT]
            )

        if not ground_roll:
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
                    Dynamic.Vehicle.ANGLE_OF_ATTACK,
                    Dynamic.Vehicle.LIFT,
                    Dynamic.Vehicle.MASS,
                    Dynamic.Mission.FLIGHT_PATH_ANGLE,
                    Dynamic.Mission.VELOCITY,
                ],
                rows=arange,
                cols=arange,
            )
            self.declare_partials(Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, [Aircraft.Wing.INCIDENCE])
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
            self.declare_partials('load_factor', [Aircraft.Wing.INCIDENCE])

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
        # self.declare_partials("angle_of_attack_rate", ["*"], val=0.0)
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
            val=180 / np.pi,
        )
        self.declare_partials('fuselage_pitch', [Aircraft.Wing.INCIDENCE])

    def compute(self, inputs, outputs):
        if self.options['ground_roll']:
            mu = inputs[Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT]
        else:
            mu = 0.0

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        if self.options['ground_roll']:
            alpha = inputs[Aircraft.Wing.INCIDENCE]
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)
        normal_force = weight - incremented_lift - thrust_across_flightpath

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

        if not self.options['ground_roll']:
            outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE] = (
                (thrust_across_flightpath + incremented_lift - weight * np.cos(gamma))
                * GRAV_ENGLISH_GASP
                / (TAS * weight)
            )
            outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * np.sin(gamma)

        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)

        outputs['normal_force'] = normal_force

        outputs['fuselage_pitch'] = gamma * 180 / np.pi - i_wing + alpha

        load_factor = (incremented_lift + thrust_across_flightpath) / (weight * np.cos(gamma))

        outputs['load_factor'] = load_factor

    def compute_partials(self, inputs, J):
        if self.options['ground_roll']:
            mu = inputs[Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT]
        else:
            mu = 0.0

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        if self.options['ground_roll']:
            alpha = i_wing
        else:
            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        nn = self.options['num_nodes']

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)

        dTAlF_dThrust = np.cos((alpha - i_wing) * np.pi / 180)
        dTAlF_dAlpha = -thrust * np.sin((alpha - i_wing) * np.pi / 180) * np.pi / 180
        dTAlF_dIwing = thrust * np.sin((alpha - i_wing) * np.pi / 180) * np.pi / 180

        dTAcF_dThrust = np.sin((alpha - i_wing) * np.pi / 180)
        dTAcF_dAlpha = thrust * np.cos((alpha - i_wing) * np.pi / 180) * np.pi / 180
        dTAcF_dIwing = -thrust * np.cos((alpha - i_wing) * np.pi / 180) * np.pi / 180

        J['load_factor', Dynamic.Vehicle.LIFT] = 1 / (weight * np.cos(gamma))
        J['load_factor', Dynamic.Vehicle.MASS] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight**2 * np.cos(gamma))
            * GRAV_ENGLISH_LBM
        )
        J['load_factor', Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -(incremented_lift + thrust_across_flightpath)
            / (weight * (np.cos(gamma)) ** 2)
            * (-np.sin(gamma))
        )
        J['load_factor', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dTAcF_dThrust / (
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

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            (dTAlF_dThrust - mu * dNF_dThrust) * GRAV_ENGLISH_GASP / weight
        )

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.DRAG] = -GRAV_ENGLISH_GASP / weight
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.MASS] = (
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
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -np.cos(gamma) * GRAV_ENGLISH_GASP
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.LIFT] = (
            GRAV_ENGLISH_GASP * (-mu * dNF_dLift) / weight
        )
        if self.options['ground_roll']:
            J[Dynamic.Mission.VELOCITY_RATE, Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT] = (
                -normal_force * GRAV_ENGLISH_GASP / weight
            )

        # TODO: check partials, esp. for alphas
        if not self.options['ground_roll']:
            J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
            J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = TAS * np.cos(
                gamma
            )

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
            ] = weight * np.sin(gamma) * GRAV_ENGLISH_GASP / (TAS * weight)
            J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.VELOCITY] = -(
                (thrust_across_flightpath + incremented_lift - weight * np.cos(gamma))
                * GRAV_ENGLISH_GASP
                / (TAS**2 * weight)
            )

            dNF_dAlpha = -np.ones(nn) * dTAcF_dAlpha
            # dNF_dAlpha[normal_force1 < 0] = 0
            J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
                (dTAlF_dAlpha - mu * dNF_dAlpha) * GRAV_ENGLISH_GASP / weight
            )
            J['normal_force', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dNF_dAlpha
            J['fuselage_pitch', Dynamic.Vehicle.ANGLE_OF_ATTACK] = 1
            J['load_factor', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dTAcF_dAlpha / (
                weight * np.cos(gamma)
            )
            J[Dynamic.Mission.VELOCITY_RATE, Aircraft.Wing.INCIDENCE] = (
                (dTAlF_dIwing - mu * dNF_dIwing) * GRAV_ENGLISH_GASP / weight
            )
            J['normal_force', Aircraft.Wing.INCIDENCE] = dNF_dIwing
            J['fuselage_pitch', Aircraft.Wing.INCIDENCE] = -1
            J['load_factor', Aircraft.Wing.INCIDENCE] = dTAcF_dIwing / (weight * np.cos(gamma))

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -TAS * np.sin(gamma)

        J['normal_force', Dynamic.Vehicle.MASS] = dNF_dWeight * GRAV_ENGLISH_LBM
        J['normal_force', Dynamic.Vehicle.LIFT] = dNF_dLift
        J['normal_force', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dNF_dThrust
