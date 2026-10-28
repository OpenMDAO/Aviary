import numpy as np
import openmdao.api as om

from aviary.utils.math_utils import deg2rad, rad2deg
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class GroundrollEOM(om.ExplicitComponent):
    """GASP based ground roll EOM."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            shape=nn,
            units='N',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.LIFT,
            shape=nn,
            units='N',
        )
        add_aviary_input(
            self,
            Dynamic.Vehicle.DRAG,
            shape=nn,
            units='N',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='m/s',
        )
        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, shape=nn, units='deg')
        add_aviary_input(self, Aircraft.Wing.INCIDENCE, units='deg')
        add_aviary_input(self, Dynamic.Vehicle.ANGLE_OF_ATTACK, shape=nn, units='deg')
        add_aviary_input(self, Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, units='unitless')

        add_aviary_output(
            self,
            Dynamic.Mission.VELOCITY_RATE,
            shape=nn,
            units='m/s**2',
        )
        add_aviary_output(
            self,
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            shape=nn,
            units='deg/s',
        )
        add_aviary_output(self, Dynamic.Mission.ALTITUDE_RATE, shape=nn, units='m/s')
        add_aviary_output(self, Dynamic.Mission.DISTANCE_RATE, shape=nn, units='m/s')
        self.add_output('normal_force', val=np.ones(nn), desc='normal forces', units='N')
        self.add_output('fuselage_pitch', val=np.ones(nn), desc='fuselage pitch angle', units='deg')
        self.add_output(
            'angle_of_attack_rate', val=np.ones(nn), desc='angle of attack rate', units='deg/s'
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

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
        self.declare_partials(
            Dynamic.Mission.VELOCITY_RATE,
            [
                Aircraft.Wing.INCIDENCE,
                Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
            ],
        )
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
        self.declare_partials('normal_force', Aircraft.Wing.INCIDENCE)
        self.declare_partials(
            'fuselage_pitch',
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            rows=arange,
            cols=arange,
            val=1,
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
        gravity = self.options[Mission.GRAVITY]
        mu = inputs[Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT]
        mass = inputs[Dynamic.Vehicle.MASS]
        weight = mass * gravity[0]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = deg2rad(inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE])
        i_wing = deg2rad(inputs[Aircraft.Wing.INCIDENCE])
        alpha = deg2rad(inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK])

        nn = self.options['num_nodes']

        thrust_along_flightpath = thrust * np.cos(alpha - i_wing)
        thrust_across_flightpath = thrust * np.sin(alpha - i_wing)
        normal_force = weight - incremented_lift - thrust_across_flightpath
        normal_force[normal_force < 0] = 0.0

        outputs[Dynamic.Mission.VELOCITY_RATE] = (
            thrust_along_flightpath - incremented_drag - weight * np.sin(gamma) - mu * normal_force
        ) / mass
        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE] = rad2deg(np.zeros(nn))

        outputs[Dynamic.Mission.ALTITUDE_RATE] = TAS * np.sin(gamma)
        outputs[Dynamic.Mission.DISTANCE_RATE] = TAS * np.cos(gamma)
        outputs['normal_force'] = normal_force

        outputs['fuselage_pitch'] = rad2deg(gamma - i_wing + alpha)
        outputs['angle_of_attack_rate'] = rad2deg(np.zeros(nn))

    def compute_partials(self, inputs, J):
        mu = inputs[Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT]
        gravity = self.options[Mission.GRAVITY][0]
        mass = inputs[Dynamic.Vehicle.MASS]
        weight = mass * gravity
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        incremented_lift = inputs[Dynamic.Vehicle.LIFT]
        incremented_drag = inputs[Dynamic.Vehicle.DRAG]
        TAS = inputs[Dynamic.Mission.VELOCITY]
        gamma = deg2rad(inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE])
        i_wing = deg2rad(inputs[Aircraft.Wing.INCIDENCE])
        alpha = deg2rad(inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK])

        nn = self.options['num_nodes']

        thrust_along_flightpath = thrust * np.cos(alpha - i_wing)
        thrust_across_flightpath = thrust * np.sin(alpha - i_wing)

        dTAlF_dThrust = np.cos(alpha - i_wing)
        dTAlF_dAlpha = -thrust * np.sin(alpha - i_wing) * deg2rad(1.0)
        dTAlF_dIwing = thrust * np.sin(alpha - i_wing) * deg2rad(1.0)

        dTAcF_dThrust = np.sin(alpha - i_wing)
        dTAcF_dAlpha = thrust * np.cos(alpha - i_wing) * deg2rad(1.0)
        dTAcF_dIwing = -thrust * np.cos(alpha - i_wing) * deg2rad(1.0)

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

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = (
            dTAlF_dThrust - mu * dNF_dThrust
        ) / mass
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.ANGLE_OF_ATTACK] = (
            dTAlF_dAlpha - mu * dNF_dAlpha
        ) / mass
        J[Dynamic.Mission.VELOCITY_RATE, Aircraft.Wing.INCIDENCE] = (
            dTAlF_dIwing - mu * dNF_dIwing
        ) / mass
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.DRAG] = -1 / mass
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.MASS] = (
            weight * (-np.sin(gamma) - mu * dNF_dWeight)
            - (
                thrust_along_flightpath
                - incremented_drag
                - weight * np.sin(gamma)
                - mu * normal_force
            )
        ) / mass**2
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -np.cos(gamma) * gravity * deg2rad(1.0)
        )
        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Vehicle.LIFT] = (-mu * dNF_dLift) / mass
        J[Dynamic.Mission.VELOCITY_RATE, Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT] = (
            -normal_force
        ) / mass

        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = np.sin(gamma)
        J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            TAS * np.cos(gamma) * deg2rad(1.0)
        )

        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = np.cos(gamma)
        J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -TAS * np.sin(gamma) * deg2rad(1.0)
        )

        J['normal_force', Dynamic.Vehicle.MASS] = dNF_dWeight * gravity
        J['normal_force', Dynamic.Vehicle.LIFT] = dNF_dLift
        J['normal_force', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = dNF_dThrust
        J['normal_force', Dynamic.Vehicle.ANGLE_OF_ATTACK] = dNF_dAlpha
        J['normal_force', Aircraft.Wing.INCIDENCE] = dNF_dIwing
