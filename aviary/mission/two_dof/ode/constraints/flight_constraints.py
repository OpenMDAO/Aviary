import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class FlightConstraints(om.ExplicitComponent):
    """
    Compute the minimum TAS (defined as the stall speed multiplied by a safety factor of
    1.1).

    Also compute the fuselage pitch angle.

    Both equations come from the climb subroutine of GASP.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        add_aviary_input(
            self,
            Dynamic.Vehicle.MASS,
            shape=nn,
            units='kg',
        )

        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')

        add_aviary_input(
            self,
            Dynamic.Atmosphere.DENSITY,
            shape=nn,
            units='kg/m**3',
        )
        self.add_input(
            'CL_max',
            val=np.ones(nn),
            units='unitless',
            desc='maximum lift coefficient',
        )
        add_aviary_input(
            self,
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            shape=nn,
            units='rad',
        )

        add_aviary_input(self, Aircraft.Wing.INCIDENCE, units='rad')

        add_aviary_input(
            self,
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            shape=nn,
            units='rad',
        )

        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='m/s',
        )

        self.add_output(
            'theta',
            val=np.ones(nn),
            units='rad',
            desc='pitch angle of fuselage',
        )
        self.add_output(
            'TAS_violation',
            val=np.ones(nn),
            units='m/s',
            desc='value to show if minimum TAS constraint is being violated. Negative or'
            ' zero if constraint is satisfied.',
        )
        self.add_output('TAS_min', val=np.zeros(nn), units='m/s')

        self.declare_partials(
            'theta',
            [Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Vehicle.ANGLE_OF_ATTACK],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'theta',
            [
                Aircraft.Wing.INCIDENCE,
            ],
        )
        self.declare_partials(
            'TAS_violation',
            [
                Dynamic.Vehicle.MASS,
                Dynamic.Atmosphere.DENSITY,
                'CL_max',
                Dynamic.Mission.VELOCITY,
            ],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'TAS_violation',
            [
                Aircraft.Wing.AREA,
            ],
        )
        self.declare_partials(
            'TAS_min',
            [Dynamic.Vehicle.MASS, Dynamic.Atmosphere.DENSITY, 'CL_max'],
            rows=arange,
            cols=arange,
        )
        self.declare_partials(
            'TAS_min',
            [
                Aircraft.Wing.AREA,
            ],
        )

    def compute(self, inputs, outputs):
        gravity = self.options[Mission.GRAVITY][0]

        weight = inputs[Dynamic.Vehicle.MASS] * gravity
        wing_area = inputs[Aircraft.Wing.AREA]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        CL_max = inputs['CL_max']
        gamma = inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE]
        i_wing = inputs[Aircraft.Wing.INCIDENCE]
        alpha = inputs[Dynamic.Vehicle.ANGLE_OF_ATTACK]
        TAS = inputs[Dynamic.Mission.VELOCITY]

        V_stall = (2 * weight / (wing_area * rho * CL_max)) ** 0.5  # stall speed
        TAS_min = (
            1.1 * V_stall
        )  # minimum true airspeed across each node, based on stall speed and safety margin
        outputs['TAS_min'] = TAS_min

        outputs['theta'] = gamma - i_wing + alpha
        outputs['TAS_violation'] = TAS_min - TAS

    def compute_partials(self, inputs, J):
        gravity = self.options[Mission.GRAVITY][0]

        weight = inputs[Dynamic.Vehicle.MASS] * gravity
        wing_area = inputs[Aircraft.Wing.AREA]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        CL_max = inputs['CL_max']

        J['theta', Dynamic.Mission.FLIGHT_PATH_ANGLE] = 1
        J['theta', Dynamic.Vehicle.ANGLE_OF_ATTACK] = 1
        J['theta', Aircraft.Wing.INCIDENCE] = -1

        J['TAS_violation', Dynamic.Vehicle.MASS] = (
            1.1 * 0.5 * (2 / (wing_area * rho * CL_max)) ** 0.5 * weight ** (-0.5) * gravity
        )
        J['TAS_violation', Dynamic.Atmosphere.DENSITY] = (
            1.1 * (2 * weight / (wing_area * CL_max)) ** 0.5 * (-0.5) * rho ** (-1.5)
        )
        J['TAS_violation', 'CL_max'] = (
            1.1 * (2 * weight / (wing_area * rho)) ** 0.5 * (-0.5) * CL_max ** (-1.5)
        )
        J['TAS_violation', Dynamic.Mission.VELOCITY] = -1
        J['TAS_violation', Aircraft.Wing.AREA] = (
            1.1 * (2 * weight / (rho * CL_max)) ** 0.5 * (-0.5) * wing_area ** (-1.5)
        )

        J['TAS_min', Dynamic.Vehicle.MASS] = 1.1 * (
            0.5 * (2 / (wing_area * rho * CL_max)) ** 0.5 * weight ** (-0.5) * gravity
        )
        J['TAS_min', Dynamic.Atmosphere.DENSITY] = 1.1 * (
            (2 * weight / (wing_area * CL_max)) ** 0.5 * (-0.5) * rho ** (-1.5)
        )
        J['TAS_min', 'CL_max'] = 1.1 * (
            (2 * weight / (wing_area * rho)) ** 0.5 * (-0.5) * CL_max ** (-1.5)
        )
        J['TAS_min', Aircraft.Wing.AREA] = 1.1 * (
            (2 * weight / (rho * CL_max)) ** 0.5 * (-0.5) * wing_area ** (-1.5)
        )
