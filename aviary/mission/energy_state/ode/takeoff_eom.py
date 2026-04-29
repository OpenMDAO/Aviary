"""Define utilities for calculating takeoff EOMs."""

import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_METRIC_FLOPS as grav_metric
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Dynamic, Mission


class StallSpeed(om.ExplicitComponent):
    """
    Define a component for calculating the minimum speed of an aircraft required to
    produce lift.
    """

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        self.add_input('mass', val=np.ones(nn), units='kg', desc='current mass of the aircraft')

        add_aviary_input(
            self,
            Dynamic.Atmosphere.DENSITY,
            val=np.ones(nn),
            units='kg/m**3',
            desc='current atmospheric density',
        )

        self.add_input('area', val=1.0, units='m**2', desc='surface area contributing to lift')

        self.add_input(
            'lift_coefficient_max', val=1.0, units='unitless', desc='maximum lift coefficient'
        )

        self.add_output(
            'stall_speed',
            val=np.zeros(nn),
            units='m/s',
            desc='minimum speed of an aircraft required to produce lift',
        )

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(
            'stall_speed',
            ['mass', Dynamic.Atmosphere.DENSITY],
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials('stall_speed', ['area', 'lift_coefficient_max'])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mass = inputs['mass']
        density = inputs[Dynamic.Atmosphere.DENSITY]
        area = inputs['area']
        lift_coefficient_max = inputs['lift_coefficient_max']

        weight = mass * grav_metric

        stall_speed = (2.0 * weight / (density * area * lift_coefficient_max)) ** 0.5
        outputs['stall_speed'] = stall_speed

    def compute_partials(self, inputs, J, discrete_inputs=None):
        mass = inputs['mass']
        density = inputs[Dynamic.Atmosphere.DENSITY]
        area = inputs['area']
        lift_coefficient_max = inputs['lift_coefficient_max']

        weight = mass * grav_metric

        stall_speed = (2.0 * weight / (density * area * lift_coefficient_max)) ** 0.5

        J['stall_speed', 'mass'] = grav_metric / (
            stall_speed * density * area * lift_coefficient_max
        )

        J['stall_speed', Dynamic.Atmosphere.DENSITY] = -weight / (
            stall_speed * density**2 * area * lift_coefficient_max
        )

        J['stall_speed', 'area'] = -weight / (
            stall_speed * density * area**2 * lift_coefficient_max
        )

        J['stall_speed', 'lift_coefficient_max'] = -weight / (
            stall_speed * density * area * lift_coefficient_max**2
        )


class TakeoffEOM(om.Group):
    """Define a group for calculating takeoff equations of motion."""

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'climbing', default=False, types=bool, desc='mode of operation (ground roll or flight)'
        )

        options.declare(
            'friction_key',
            desc='current friction coefficient key, either rolling friction or braking friction',
        )

        options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

    def setup(self):
        options = self.options

        nn = options['num_nodes']
        climbing = options['climbing']
        friction_key = options['friction_key']
        aviary_options = options['aviary_options']
        mu = aviary_options.get_val(friction_key)

        kwargs = {'num_nodes': nn, 'climbing': climbing}

        inputs = [Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.VELOCITY]
        outputs = [Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.ALTITUDE_RATE]

        self.add_subsystem(
            'distance_rates',
            DistanceRates(**kwargs),
            promotes_inputs=inputs,
            promotes_outputs=outputs,
        )

        kwargs = {
            'num_nodes': nn,
            'climbing': climbing,
            'friction_coefficient': mu,
            'aviary_options': aviary_options,
        }

        self.add_subsystem(
            'sum_forces',
            SumForces(**kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['forces_horizontal', 'forces_vertical'],
        )

        self.add_subsystem(
            'accelerations',
            Accelerations(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'velocity_rate',
            VelocityRate(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'flight_path_angle_rate',
            FlightPathAngleRate(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'climb_gradient_forces',
            ClimbGradientForces(num_nodes=nn, aviary_options=aviary_options),
            promotes=['*'],
        )


class DistanceRates(om.ExplicitComponent):
    """
    Define a component for calculating takeoff horizontal and vertical velocity
    components.
    """

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'climbing', default=False, types=bool, desc='mode of operation (ground roll or flight)'
        )

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(nn), units='rad')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, val=np.zeros(nn), units='m/s')

        add_aviary_output(self, Dynamic.Mission.DISTANCE_RATE, val=np.zeros(nn), units='m/s')
        add_aviary_output(self, Dynamic.Mission.ALTITUDE_RATE, val=np.zeros(nn), units='m/s')

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']
        rows_cols = np.arange(nn)
        climbing = options['climbing']

        if climbing:
            self.declare_partials('*', '*', rows=rows_cols, cols=rows_cols)

        else:
            self.declare_partials(
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                dependent=False,
            )

            self.declare_partials(
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.VELOCITY,
                val=np.identity(nn),
            )

            self.declare_partials(Dynamic.Mission.ALTITUDE_RATE, '*', dependent=False)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        velocity = inputs[Dynamic.Mission.VELOCITY]

        if self.options['climbing']:
            flight_path_angle = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

            cgam = np.cos(flight_path_angle)
            range_rate = cgam * velocity

            sgam = np.sin(flight_path_angle)
            altitude_rate = sgam * velocity

            outputs[Dynamic.Mission.ALTITUDE_RATE] = altitude_rate

        else:
            range_rate = velocity

        outputs[Dynamic.Mission.DISTANCE_RATE] = range_rate

    def compute_partials(self, inputs, J, discrete_inputs=None):
        if self.options['climbing']:
            flight_path_angle = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
            velocity = inputs[Dynamic.Mission.VELOCITY]

            cgam = np.cos(flight_path_angle)
            sgam = np.sin(flight_path_angle)

            J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -sgam * velocity
            J[Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.VELOCITY] = cgam

            J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.FLIGHT_PATH_ANGLE] = cgam * velocity
            J[Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY] = sgam


class Accelerations(om.ExplicitComponent):
    """Define a component for calculating horizontal and vertical accelerations from forces."""

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.MASS, val=np.ones(nn), units='kg')

        self.add_input(
            'forces_horizontal',
            val=np.zeros(nn),
            units='N',
            desc='current sum of forces in the horizontal direction',
        )

        self.add_input(
            'forces_vertical',
            val=np.zeros(nn),
            units='N',
            desc='current sum of forces in the vertical direction',
        )

        self.add_output(
            'acceleration_horizontal',
            val=np.zeros(nn),
            desc='current horizontal acceleration',
            units='m/s**2',
        )

        self.add_output(
            'acceleration_vertical',
            val=np.zeros(nn),
            desc='current vertical acceleration',
            units='m/s**2',
        )

    def setup_partials(self):
        nn = self.options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials(
            'acceleration_horizontal',
            Dynamic.Vehicle.MASS,
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'acceleration_vertical',
            Dynamic.Vehicle.MASS,
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'acceleration_horizontal', 'forces_horizontal', rows=rows_cols, cols=rows_cols
        )

        self.declare_partials('acceleration_vertical', 'forces_horizontal', dependent=False)

        self.declare_partials('acceleration_horizontal', 'forces_vertical', dependent=False)

        self.declare_partials(
            'acceleration_vertical', 'forces_vertical', rows=rows_cols, cols=rows_cols
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mass = inputs[Dynamic.Vehicle.MASS]
        f_h = inputs['forces_horizontal']
        f_v = inputs['forces_vertical']

        a_h = f_h / mass
        outputs['acceleration_horizontal'] = a_h

        a_v = f_v / mass
        outputs['acceleration_vertical'] = a_v

    def compute_partials(self, inputs, J, discrete_inputs=None):
        mass = inputs[Dynamic.Vehicle.MASS]
        f_h = inputs['forces_horizontal']
        f_v = inputs['forces_vertical']

        m2 = mass * mass

        J['acceleration_horizontal', Dynamic.Vehicle.MASS] = -f_h / m2
        J['acceleration_vertical', Dynamic.Vehicle.MASS] = -f_v / m2

        J['acceleration_horizontal', 'forces_horizontal'] = 1.0 / mass

        J['acceleration_vertical', 'forces_vertical'] = 1.0 / mass


class VelocityRate(om.ExplicitComponent):
    """Define a component for calculating total acceleration."""

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            'acceleration_horizontal',
            val=np.zeros(nn),
            desc='current horizontal acceleration',
            units='m/s**2',
        )

        self.add_input(
            'acceleration_vertical',
            val=np.zeros(nn),
            desc='current vertical acceleration',
            units='m/s**2',
        )

        add_aviary_input(self, Dynamic.Mission.DISTANCE_RATE, val=np.zeros(nn), units='m/s')
        add_aviary_input(self, Dynamic.Mission.ALTITUDE_RATE, val=np.zeros(nn), units='m/s')

        add_aviary_output(self, Dynamic.Mission.VELOCITY_RATE, val=np.ones(nn), units='m/s**2')

        rows_cols = np.arange(nn)

        self.declare_partials('*', '*', rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        a_h = inputs['acceleration_horizontal']
        a_v = inputs['acceleration_vertical']
        v_h = inputs[Dynamic.Mission.DISTANCE_RATE]
        v_v = inputs[Dynamic.Mission.ALTITUDE_RATE]

        v_mag = np.sqrt(v_h**2 + v_v**2)
        outputs[Dynamic.Mission.VELOCITY_RATE] = (a_h * v_h + a_v * v_v) / v_mag

    def compute_partials(self, inputs, J, discrete_inputs=None):
        a_h = inputs['acceleration_horizontal']
        a_v = inputs['acceleration_vertical']
        v_h = inputs[Dynamic.Mission.DISTANCE_RATE]
        v_v = inputs[Dynamic.Mission.ALTITUDE_RATE]

        num = a_h * v_h + a_v * v_v
        fact = v_h**2 + v_v**2
        den = np.sqrt(fact)

        J[Dynamic.Mission.VELOCITY_RATE, 'acceleration_horizontal'] = v_h / den
        J[Dynamic.Mission.VELOCITY_RATE, 'acceleration_vertical'] = v_v / den

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.DISTANCE_RATE] = (
            a_h / den - 0.5 * num / fact ** (3 / 2) * 2.0 * v_h
        )

        J[Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.ALTITUDE_RATE] = (
            a_v / den - 0.5 * num / fact ** (3 / 2) * 2.0 * v_v
        )


class FlightPathAngleRate(om.ExplicitComponent):
    """Define a component for calculating flight path angle change rate."""

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.DISTANCE_RATE, val=np.zeros(nn), units='m/s')
        add_aviary_input(self, Dynamic.Mission.ALTITUDE_RATE, val=np.zeros(nn), units='m/s')

        self.add_input(
            'acceleration_horizontal',
            val=np.zeros(nn),
            desc='current horizontal acceleration',
            units='m/s**2',
        )

        self.add_input(
            'acceleration_vertical',
            val=np.zeros(nn),
            desc='current vertical acceleration',
            units='m/s**2',
        )

        add_aviary_output(
            self,
            Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            val=np.zeros(nn),
            units='rad/s',
        )

        rows_cols = np.arange(nn)

        self.declare_partials('*', '*', rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v_h = inputs[Dynamic.Mission.DISTANCE_RATE]
        v_v = inputs[Dynamic.Mission.ALTITUDE_RATE]
        a_h = inputs['acceleration_horizontal']
        a_v = inputs['acceleration_vertical']

        x = (a_v * v_h - a_h * v_v) / (v_h**2 + v_v**2)

        outputs[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE] = x

    def compute_partials(self, inputs, J, discrete_inputs=None):
        v_h = inputs[Dynamic.Mission.DISTANCE_RATE]
        v_v = inputs[Dynamic.Mission.ALTITUDE_RATE]
        a_h = inputs['acceleration_horizontal']
        a_v = inputs['acceleration_vertical']

        num = a_v * v_h - a_h * v_v
        den = v_h**2 + v_v**2

        df_dvh = a_v / den - num / den**2 * 2.0 * v_h

        df_dvv = -a_h / den - num / den**2 * 2.0 * v_v

        df_dah = -v_v / den

        df_dav = v_h / den

        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.DISTANCE_RATE] = df_dvh
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, Dynamic.Mission.ALTITUDE_RATE] = df_dvv
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, 'acceleration_horizontal'] = df_dah
        J[Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE, 'acceleration_vertical'] = df_dav


class SumForces(om.ExplicitComponent):
    """
    Define a component for calculating the separate sums for both the horizontal and
    vertical forces.
    """

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'climbing', default=False, types=bool, desc='mode of operation (ground roll or flight)'
        )

        options.declare(
            'friction_coefficient',
            default=0.025,
            desc='current friction coefficient, either rolling friction or braking friction',
        )

        options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.MASS, val=np.ones(nn), units='kg')
        add_aviary_input(self, Dynamic.Vehicle.LIFT, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Vehicle.DRAG, val=np.ones(nn), units='N')

        self.add_input(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(nn), units='rad')

        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(nn), units='rad')

        self.add_output(
            'forces_horizontal',
            val=np.zeros(nn),
            units='N',
            desc='current sum of forces in the horizontal direction',
        )

        self.add_output(
            'forces_vertical',
            val=np.zeros(nn),
            units='N',
            desc='current sum of forces in the vertical direction',
        )

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']
        climbing = options['climbing']

        rows_cols = np.arange(nn)

        if climbing:
            self.declare_partials('forces_horizontal', Dynamic.Vehicle.MASS, dependent=False)

            self.declare_partials(
                'forces_vertical',
                Dynamic.Vehicle.MASS,
                val=-grav_metric,
                rows=rows_cols,
                cols=rows_cols,
            )

            wrt = [
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ]

            self.declare_partials('*', wrt, rows=rows_cols, cols=rows_cols)

        else:
            aviary_options: AviaryValues = options['aviary_options']

            mu = options['friction_coefficient']
            val = -grav_metric * mu

            self.declare_partials(
                'forces_horizontal',
                Dynamic.Vehicle.MASS,
                val=val,
                rows=rows_cols,
                cols=rows_cols,
            )

            self.declare_partials(
                'forces_horizontal',
                Dynamic.Vehicle.LIFT,
                val=mu,
                rows=rows_cols,
                cols=rows_cols,
            )

            t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')
            val = np.cos(t_inc) + np.sin(t_inc) * mu

            self.declare_partials(
                'forces_horizontal',
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                val=val,
                rows=rows_cols,
                cols=rows_cols,
            )

            self.declare_partials(
                'forces_horizontal',
                Dynamic.Vehicle.DRAG,
                val=-1.0,
                rows=rows_cols,
                cols=rows_cols,
            )

            self.declare_partials(
                'forces_horizontal',
                [Dynamic.Vehicle.ANGLE_OF_ATTACK, Dynamic.Mission.FLIGHT_PATH_ANGLE],
                dependent=False,
            )

            self.declare_partials('forces_vertical', ['*'], dependent=False)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options = self.options

        climbing = options['climbing']
        aviary_options: AviaryValues = options['aviary_options']

        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')

        mass = inputs[Dynamic.Vehicle.MASS]
        lift = inputs[Dynamic.Vehicle.LIFT]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]

        weight = mass * grav_metric

        if climbing:
            # NOTE using FLOPS ROTCL
            #    - section: "COMPUTE TRAJECTORY FROM LIFTOFF UNTIL OBSTACLE HEIGHT IS
            #      REACHED"
            #    - variables: FORCH, FORCV
            alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')

            alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
            gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

            angle = alpha - alpha0 + t_inc + gamma

            t_h = thrust * np.cos(angle)
            t_v = thrust * np.sin(angle)

            c_gamma = np.cos(gamma)
            s_gamma = np.sin(gamma)

            f_h = t_h - drag * c_gamma - lift * s_gamma

            f_v = t_v - drag * s_gamma + lift * c_gamma - weight

            outputs['forces_vertical'] = f_v

        else:
            # NOTE using FLOPS GRRUN, which neglects angle of attack
            #    - FLOPS ROTCL applies angle of attack to thrust incidence only; angle of
            #      attack is not applied to any other force
            #    - variables: ACC
            mu = options['friction_coefficient']

            t_h = thrust * np.cos(t_inc)
            t_v = thrust * np.sin(t_inc)

            f_h = t_h - drag - (weight - (lift + t_v)) * mu

        outputs['forces_horizontal'] = f_h

    def compute_partials(self, inputs, J, discrete_inputs=None):
        options = self.options

        climbing = options['climbing']

        if not climbing:
            # see setup_partials()
            return

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')

        lift = inputs[Dynamic.Vehicle.LIFT]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]

        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        angle = alpha - alpha0 + t_inc + gamma

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        J['forces_horizontal', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = c_angle
        J['forces_vertical', Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = s_angle

        J['forces_horizontal', Dynamic.Vehicle.LIFT] = -s_gamma
        J['forces_vertical', Dynamic.Vehicle.LIFT] = c_gamma

        J['forces_horizontal', Dynamic.Vehicle.DRAG] = -c_gamma
        J['forces_vertical', Dynamic.Vehicle.DRAG] = -s_gamma

        J['forces_horizontal', Dynamic.Vehicle.ANGLE_OF_ATTACK] = -thrust * s_angle
        J['forces_vertical', Dynamic.Vehicle.ANGLE_OF_ATTACK] = thrust * c_angle

        J['forces_horizontal', Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            -thrust * s_angle + drag * s_gamma - lift * c_gamma
        )

        J['forces_vertical', Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
            thrust * c_angle - drag * c_gamma - lift * s_gamma
        )


class ClimbGradientForces(om.ExplicitComponent):
    """
    Define a component for calculating residual forces for evaluation of climb gradient
    criteria.
    """

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.MASS, val=np.ones(nn), units='kg')
        add_aviary_input(self, Dynamic.Vehicle.LIFT, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.THRUST_TOTAL, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Vehicle.DRAG, val=np.ones(nn), units='N')

        self.add_input(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=np.zeros(nn), units='rad')

        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE, val=np.zeros(nn), units='rad')

        self.add_output(
            'climb_gradient_forces_horizontal',
            val=np.zeros(nn),
            units='N',
            desc='current sum of forces in the horizontal direction; checking for excess thrust',
        )

        self.add_output(
            'climb_gradient_forces_vertical',
            val=np.zeros(nn),
            units='N',
            desc='current sum of forces in the vertical direction; checking for net zero'
            ' vertical force',
        )

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials(
            '*',
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ],
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'climb_gradient_forces_horizontal',
            Dynamic.Vehicle.DRAG,
            val=-1.0,
            rows=rows_cols,
            cols=rows_cols,
        )

        self.declare_partials(
            'climb_gradient_forces_vertical', Dynamic.Vehicle.DRAG, dependent=False
        )

        self.declare_partials(
            'climb_gradient_forces_horizontal', Dynamic.Vehicle.LIFT, dependent=False
        )

        self.declare_partials(
            'climb_gradient_forces_vertical',
            Dynamic.Vehicle.LIFT,
            val=1.0,
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options = self.options

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')

        mass = inputs[Dynamic.Vehicle.MASS]
        lift = inputs[Dynamic.Vehicle.LIFT]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]
        drag = inputs[Dynamic.Vehicle.DRAG]

        weight = mass * grav_metric

        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        angle = alpha - alpha0 + t_inc

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        # NOTE using FLOPS CLGRAD
        #    - variables: FORCE2, FORCV
        f_h = -drag - weight * s_gamma + thrust * c_angle
        outputs['climb_gradient_forces_horizontal'] = f_h

        f_v = lift - weight * c_gamma + thrust * s_angle
        outputs['climb_gradient_forces_vertical'] = f_v

    def compute_partials(self, inputs, J, discrete_inputs=None):
        options = self.options

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')

        mass = inputs[Dynamic.Vehicle.MASS]
        thrust = inputs[Dynamic.Vehicle.Propulsion.THRUST_TOTAL]

        weight = mass * grav_metric

        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        angle = alpha - alpha0 + t_inc

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        f_h_key = 'climb_gradient_forces_horizontal'
        f_v_key = 'climb_gradient_forces_vertical'

        J[f_h_key, Dynamic.Vehicle.MASS] = -grav_metric * s_gamma
        J[f_v_key, Dynamic.Vehicle.MASS] = -grav_metric * c_gamma

        J[f_h_key, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = c_angle
        J[f_v_key, Dynamic.Vehicle.Propulsion.THRUST_TOTAL] = s_angle

        J[f_h_key, Dynamic.Vehicle.ANGLE_OF_ATTACK] = -thrust * s_angle
        J[f_v_key, Dynamic.Vehicle.ANGLE_OF_ATTACK] = thrust * c_angle

        J[f_h_key, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -weight * c_gamma
        J[f_v_key, Dynamic.Mission.FLIGHT_PATH_ANGLE] = weight * s_gamma
