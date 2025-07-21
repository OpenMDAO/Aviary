import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_GASP, GRAV_ENGLISH_LBM, MU_TAKEOFF
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic


class RotationEOM(om.ExplicitComponent):
    """2-degree of freedom rotation EOM."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

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

        add_aviary_input(self, Aircraft.Wing.INCIDENCE, val=0.0, units='deg')
        self.add_input(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            val=np.ones(nn),
            desc='angle of attack',
            units='deg',
        )

        self.add_output(
            Dynamic.Mission.VELOCITY_RATE,
            val=np.ones(nn),
            desc='TAS rate',
            units='ft/s**2',
        )
        self.add_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            val=np.ones(nn),
            desc='flight path angle rate',
            units='rad/s',
        )
        self.add_output(
            Dynamic.Mission.ALTITUDE_RATE,
            val=np.ones(nn),
            desc='altitude rate',
            units='ft/s',
        )
        self.add_output(
            Dynamic.Mission.DISTANCE_RATE, val=np.ones(nn), desc='distance rate', units='ft/s'
        )
        self.add_output('normal_force', val=np.ones(nn), desc='normal forces', units='lbf')
        self.add_output('fuselage_pitch', val=np.ones(nn), desc='fuselage pitch angle', units='deg')
        self.add_output(
            'angle_of_attack_rate', val=np.ones(nn), desc='angle of attack rate', units='deg/s'
        )

        self.declare_partials('angle_of_attack_rate', ['*'])

    def setup_partials(self):
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, '*')
        self.declare_partials(
            Dynamic.Mission.VELOCITY_RATE,
            [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
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
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
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
        self.declare_partials(
            'fuselage_pitch',
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            rows=arange,
            cols=arange,
            val=1,
        )
        self.declare_partials('fuselage_pitch', Aircraft.Wing.INCIDENCE, val=-1)

    def compute(self, inputs, outputs):
        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]

        mu = MU_TAKEOFF

        nn = self.options['num_nodes']

        thrust_along_flightpath = thrust * np.cos((alpha - i_wing) * np.pi / 180)
        thrust_across_flightpath = thrust * np.sin((alpha - i_wing) * np.pi / 180)
        normal_force = np.clip(
            weight - incremented_lift - thrust_across_flightpath, a_min=0.0, a_max=None
        )

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
        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE] = np.zeros(nn)

        outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs['normal_force'] = normal_force
        outputs['fuselage_pitch'] = gamma * 180 / np.pi - i_wing + alpha
        outputs['angle_of_attack_rate'] = np.full(nn, 3.33)

    def compute_partials(self, inputs, J):
        mu = MU_TAKEOFF

        weight = inputs[Dynamic.Vehicle.MASS] * GRAV_ENGLISH_LBM
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
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

        normal_force = np.clip(
            weight - incremented_lift - thrust_across_flightpath, a_min=0.0, a_max=None
        )

        dNF_dWeight = np.ones(nn)
        dNF_dWeight[normal_force < 0] = 0

        dNF_dLift = -np.ones(nn)
        dNF_dLift[normal_force < 0] = 0

        dNF_dThrust = -np.ones(nn) * dTAcF_dThrust
        dNF_dThrust[normal_force < 0] = 0

        dNF_dAlpha = -np.ones(nn) * dTAcF_dAlpha
        dNF_dAlpha[normal_force < 0] = 0

        dNF_dIwing = -np.ones(nn) * dTAcF_dIwing
        dNF_dIwing[normal_force < 0] = 0

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            (dTAlF_dThrust - mu * dNF_dThrust) * GRAV_ENGLISH_GASP / weight
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
            (dTAlF_dAlpha - mu * dNF_dAlpha) * GRAV_ENGLISH_GASP / weight
        )
        J[Dynamic.Mission.VELOCITY_RATE, Aircraft.Wing.INCIDENCE] = (
            (dTAlF_dIwing - mu * dNF_dIwing) * GRAV_ENGLISH_GASP / weight
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

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = TAS * np.cos(gamma)

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -TAS * np.sin(gamma)

        J['normal_force', Dynamic.Vehicle.MASS] = dNF_dWeight * GRAV_ENGLISH_LBM
        J['normal_force', Dynamic.Vehicle.LIFT] = dNF_dLift
        J['normal_force', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dNF_dThrust
        J['normal_force', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dNF_dAlpha
        J['normal_force', Aircraft.Wing.INCIDENCE] = dNF_dIwing
