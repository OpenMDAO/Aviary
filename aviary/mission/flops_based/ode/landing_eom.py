'''
Define utilities for calculating landing EOMs.
'''
import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_METRIC_FLOPS as grav_metric
from aviary.mission.flops_based.ode.takeoff_eom import (Accelerations,
                                                        DistanceRates,
                                                        FlightPathAngleRate,
                                                        StallSpeed,
                                                        VelocityRate)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class FlareEOM(om.Group):
    '''
    Define a group for calculating equations of motion from start of flare to touchdown.
    '''

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options

        nn = options['num_nodes']
        aviary_options = options['aviary_options']

        kwargs = {
            'num_nodes': nn,
            'climbing': True}

        inputs = [Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.VELOCITY]
        outputs = [Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.ALTITUDE_RATE]

        self.add_subsystem(
            'distance_rates',
            DistanceRates(**kwargs),
            promotes_inputs=inputs,
            promotes_outputs=outputs)

        kwargs = {
            'num_nodes': nn,
            'aviary_options': aviary_options}

        inputs = [
            Dynamic.Mission.MASS, Dynamic.Mission.LIFT, Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.DRAG,
            'angle_of_attack', Dynamic.Mission.FLIGHT_PATH_ANGLE]

        outputs = ['forces_horizontal', 'forces_vertical']

        self.add_subsystem(
            'sum_forces',
            FlareSumForces(**kwargs),
            promotes_inputs=inputs,
            promotes_outputs=outputs)

        inputs = ['forces_horizontal', 'forces_vertical', Dynamic.Mission.MASS]
        outputs = ['acceleration_horizontal', 'acceleration_vertical']

        self.add_subsystem(
            'accelerations',
            Accelerations(num_nodes=nn),
            promotes_inputs=inputs,
            promotes_outputs=outputs)

        inputs = [
            'acceleration_horizontal', 'acceleration_vertical',
            Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.ALTITUDE_RATE]

        outputs = [Dynamic.Mission.VELOCITY_RATE,]

        self.add_subsystem(
            'velocity_rate',
            VelocityRate(num_nodes=nn),
            promotes_inputs=inputs,
            promotes_outputs=outputs)

        inputs = [
            Dynamic.Mission.DISTANCE_RATE, Dynamic.Mission.ALTITUDE_RATE,
            'acceleration_horizontal', 'acceleration_vertical']

        outputs = [Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE]

        self.add_subsystem(
            'flight_path_angle_rate', FlightPathAngleRate(num_nodes=nn),
            promotes_inputs=inputs,
            promotes_outputs=outputs)

        inputs = [
            Dynamic.Mission.MASS, Dynamic.Mission.LIFT, Dynamic.Mission.DRAG,
            'angle_of_attack', Dynamic.Mission.FLIGHT_PATH_ANGLE]

        outputs = ['forces_perpendicular', 'required_thrust']

        self.add_subsystem(
            'glide_slope_forces',
            GlideSlopeForces(**kwargs),
            promotes_inputs=inputs,
            promotes_outputs=outputs)

        expr = 'net_alpha_rate = flare_rate - angle_of_attack_rate'
        flare_comp = om.ExecComp(
            expr,
            flare_rate={'shape': 1, 'units': 'deg/s'},
            angle_of_attack_rate={'shape': nn, 'units': 'deg/s'},
            net_alpha_rate={'shape': nn, 'units': 'deg/s'},
            has_diag_partials=True,
        )
        self.add_subsystem(
            "flare_rate",
            flare_comp,
            promotes_inputs=[('flare_rate', Mission.Landing.FLARE_RATE),
                             'angle_of_attack_rate'],
            promotes_outputs=['net_alpha_rate']
        )


class GlideSlopeForces(om.ExplicitComponent):
    '''
    Define a component for calculating forces for evaluation of glide slope criteria.
    '''

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.MASS, val=np.ones(nn), units='kg')
        add_aviary_input(self, Dynamic.Mission.LIFT, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Mission.DRAG, val=np.ones(nn), units='N')

        self.add_input('angle_of_attack', val=np.zeros(nn), units='rad')

        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE,
                         val=np.zeros(nn), units='rad')

        self.add_output(
            'forces_perpendicular', val=np.zeros(nn), units='N',
            desc='current forces perpendicular to the thrust vector; checking for zero'
            ' perpendicular force')

        self.add_output(
            'required_thrust', val=np.zeros(nn), units='N',
            desc='current estimate of thrust required to maintain glide slope')

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials('*', '*', rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options = self.options

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')
        total_num_engines = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES)

        mass = inputs[Dynamic.Mission.MASS]
        lift = inputs[Dynamic.Mission.LIFT]
        drag = inputs[Dynamic.Mission.DRAG]

        weight = mass * grav_metric

        alpha = inputs['angle_of_attack']
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        # FLOPS measures glideslope below horizontal
        gamma = -gamma

        angle = alpha - alpha0 + t_inc

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        # NOTE using FLOPS LNDING
        #    - section: BEGIN ITERATION ON START OF FLARE ALTITUDE...
        #    - variables: FORCH, FORCV, DELFOR(K), THRU
        f_h = (drag - weight * s_gamma) / c_angle
        f_v = (weight * c_gamma - lift) / s_angle

        outputs['forces_perpendicular'] = f_h - f_v
        outputs['required_thrust'] = (f_h + f_v) / (2.)

    def compute_partials(self, inputs, J, discrete_inputs=None):
        options = self.options

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')
        total_num_engines = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES)

        mass = inputs[Dynamic.Mission.MASS]
        lift = inputs[Dynamic.Mission.LIFT]
        drag = inputs[Dynamic.Mission.DRAG]

        weight = mass * grav_metric

        alpha = inputs['angle_of_attack']
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        # FLOPS measures glideslope below horizontal
        gamma = -gamma

        angle = alpha - alpha0 + t_inc

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        forces_key = 'forces_perpendicular'
        thrust_key = 'required_thrust'

        f_h = -grav_metric * s_gamma / c_angle
        f_v = grav_metric * c_gamma / s_angle

        J[forces_key, Dynamic.Mission.MASS] = f_h - f_v
        J[thrust_key, Dynamic.Mission.MASS] = (f_h + f_v) / (2.)

        f_h = 0.
        f_v = -1. / s_angle

        J[forces_key, Dynamic.Mission.LIFT] = -f_v
        J[thrust_key, Dynamic.Mission.LIFT] = f_v / (2.)

        f_h = 1. / c_angle
        f_v = 0.

        J[forces_key, Dynamic.Mission.DRAG] = f_h
        J[thrust_key, Dynamic.Mission.DRAG] = f_h / (2.)

        # ddx(1 / cos(x)) = sec(x) * tan(x) = tan(x) / cos(x)
        # ddx(1 / sin(x)) = -csc(x) * cot(x) = -1 / (sin(x) * tan(x))
        t_angle = np.tan(angle)

        f_h = t_angle * (drag - weight * s_gamma) / c_angle
        f_v = -(weight * c_gamma - lift) / (s_angle * t_angle)

        J[forces_key, 'angle_of_attack'] = f_h - f_v
        J[thrust_key, 'angle_of_attack'] = (f_h + f_v) / (2.)

        f_h = -weight * c_gamma / c_angle
        f_v = -weight * s_gamma / s_angle

        J[forces_key, Dynamic.Mission.FLIGHT_PATH_ANGLE] = - f_h + f_v
        J[thrust_key, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -(f_h + f_v) / (2.)


class FlareSumForces(om.ExplicitComponent):
    '''
    Define a component for calculating the separate sums for both the horizontal and
    vertical forces from start of flare to landing.
    '''

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.MASS, val=np.ones(nn), units='kg')
        add_aviary_input(self, Dynamic.Mission.LIFT, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Mission.THRUST_TOTAL, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Mission.DRAG, val=np.ones(nn), units='N')

        self.add_input('angle_of_attack', val=np.zeros(nn), units='rad')

        add_aviary_input(self, Dynamic.Mission.FLIGHT_PATH_ANGLE,
                         val=np.zeros(nn), units='rad')

        self.add_output(
            'forces_horizontal', val=np.zeros(nn), units='N',
            desc='current sum of forces in the horizontal direction')

        self.add_output(
            'forces_vertical', val=np.zeros(nn), units='N',
            desc='current sum of forces in the vertical direction')

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials('forces_horizontal', Dynamic.Mission.MASS, dependent=False)

        self.declare_partials(
            'forces_vertical', Dynamic.Mission.MASS, val=-grav_metric, rows=rows_cols,
            cols=rows_cols)

        wrt = [
            Dynamic.Mission.LIFT, Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.DRAG, 'angle_of_attack',
            Dynamic.Mission.FLIGHT_PATH_ANGLE]

        self.declare_partials('*', wrt, rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options = self.options

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')

        mass = inputs[Dynamic.Mission.MASS]
        lift = inputs[Dynamic.Mission.LIFT]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        drag = inputs[Dynamic.Mission.DRAG]

        alpha = inputs['angle_of_attack']
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        # FLOPS measures glideslope below horizontal
        gamma = -gamma

        weight = mass * grav_metric

        angle = alpha - alpha0 + t_inc - gamma

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        # NOTE using FLOPS LNDING
        #    - section: COMPUTE TRAJECTORY FROM START OF FLARE TO LANDING
        #    - variables: FORCH, FORCV
        f_h = drag * c_gamma - lift * s_gamma - thrust * c_angle
        outputs['forces_horizontal'] = f_h

        f_v = lift * c_gamma + drag * s_gamma - weight + thrust * s_angle
        outputs['forces_vertical'] = f_v

    def compute_partials(self, inputs, J, discrete_inputs=None):
        options = self.options

        aviary_options: AviaryValues = options['aviary_options']

        alpha0 = aviary_options.get_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 'rad')
        t_inc = aviary_options.get_val(Mission.Takeoff.THRUST_INCIDENCE, 'rad')

        mass = inputs[Dynamic.Mission.MASS]
        lift = inputs[Dynamic.Mission.LIFT]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        drag = inputs[Dynamic.Mission.DRAG]

        alpha = inputs['angle_of_attack']
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]

        # FLOPS measures glideslope below horizontal
        gamma = -gamma

        angle = alpha - alpha0 + t_inc - gamma

        c_angle = np.cos(angle)
        s_angle = np.sin(angle)

        c_gamma = np.cos(gamma)
        s_gamma = np.sin(gamma)

        f_h_key = 'forces_horizontal'
        J[f_h_key, Dynamic.Mission.LIFT] = -s_gamma

        f_v_key = 'forces_vertical'
        J[f_v_key, Dynamic.Mission.LIFT] = c_gamma

        J[f_h_key, Dynamic.Mission.THRUST_TOTAL] = -c_angle
        J[f_v_key, Dynamic.Mission.THRUST_TOTAL] = s_angle

        J[f_h_key, Dynamic.Mission.DRAG] = c_gamma
        J[f_v_key, Dynamic.Mission.DRAG] = s_gamma

        J[f_h_key, 'angle_of_attack'] = thrust * s_angle
        J[f_v_key, 'angle_of_attack'] = thrust * c_angle

        f_h = -drag * s_gamma - lift * c_gamma - thrust * s_angle
        J[f_h_key, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -f_h

        f_v = -lift * s_gamma + drag * c_gamma - thrust * c_angle
        J[f_v_key, Dynamic.Mission.FLIGHT_PATH_ANGLE] = -f_v


class GroundSumForces(om.ExplicitComponent):
    '''
    Define a component for calculating the separate sums for both the horizontal and
    vertical forces from start of touchdown through full stop.
    '''

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'friction_coefficient', default=0.025,
            desc='current friction coefficient, either rolling friction or breaking'
            ' friction')

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.MASS, val=np.ones(nn), units='kg')
        add_aviary_input(self, Dynamic.Mission.LIFT, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Mission.THRUST_TOTAL, val=np.ones(nn), units='N')
        add_aviary_input(self, Dynamic.Mission.DRAG, val=np.ones(nn), units='N')

        self.add_output(
            'forces_horizontal', val=np.zeros(nn), units='N',
            desc='current sum of forces in the horizontal direction')

        self.add_output(
            'forces_vertical', val=np.zeros(nn), units='N',
            desc='current sum of forces in the vertical direction')

    def setup_partials(self):
        options = self.options

        nn = options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials(
            'forces_vertical', Dynamic.Mission.MASS, val=-grav_metric, rows=rows_cols,
            cols=rows_cols)

        self.declare_partials(
            'forces_vertical', Dynamic.Mission.LIFT, val=1., rows=rows_cols, cols=rows_cols)

        self.declare_partials(
            'forces_vertical', [Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.DRAG], dependent=False)

        self.declare_partials(
            'forces_horizontal', [Dynamic.Mission.MASS, Dynamic.Mission.LIFT], rows=rows_cols,
            cols=rows_cols)

        self.declare_partials(
            'forces_horizontal', Dynamic.Mission.THRUST_TOTAL, val=-1., rows=rows_cols,
            cols=rows_cols)

        self.declare_partials(
            'forces_horizontal', Dynamic.Mission.DRAG, val=1., rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        options = self.options

        nn = options['num_nodes']
        friction_coefficient = options['friction_coefficient']

        mass = inputs[Dynamic.Mission.MASS]
        lift = inputs[Dynamic.Mission.LIFT]
        thrust = inputs[Dynamic.Mission.THRUST_TOTAL]
        drag = inputs[Dynamic.Mission.DRAG]

        weight = mass * grav_metric

        f_v = lift - weight
        outputs['forces_vertical'] = f_v

        idx_sup = np.where(f_v < 0.)
        friction = np.zeros(nn)
        friction[idx_sup] = -friction_coefficient * f_v[idx_sup]

        f_h = friction + drag - thrust
        outputs['forces_horizontal'] = f_h

    def compute_partials(self, inputs, J, discrete_inputs=None):
        options = self.options

        nn = options['num_nodes']
        friction_coefficient = options['friction_coefficient']

        mass = inputs[Dynamic.Mission.MASS]
        lift = inputs[Dynamic.Mission.LIFT]

        weight = mass * grav_metric

        f_v = lift - weight

        idx_sup = np.where(f_v < 0.)
        friction = np.zeros(nn)
        friction[idx_sup] = friction_coefficient * grav_metric

        J['forces_horizontal', Dynamic.Mission.MASS] = friction

        friction = np.zeros(nn)
        friction[idx_sup] = -friction_coefficient
        J['forces_horizontal', Dynamic.Mission.LIFT] = friction
