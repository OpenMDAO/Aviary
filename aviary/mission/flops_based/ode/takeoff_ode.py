"""Define the ODE for takeoff."""

import numpy as np
import openmdao.api as om

from aviary.mission.base_ode import BaseODE as _BaseODE
from aviary.mission.flops_based.ode.takeoff_eom import StallSpeed, TakeoffEOM
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TakeoffODE(_BaseODE):
    """Define the ODE for takeoff."""

    # region : derived type customization points
    stall_speed_lift_coefficient_name = Mission.Takeoff.LIFT_COEFFICIENT_MAX
    # endregion : derived type customization points

    def initialize(self):
        super().initialize()
        self.options.declare(
            'friction_key',
            desc='current friction coefficient key, either rolling friction or braking friction',
        )

        self.options.declare(
            'climbing',
            default=False,
            types=bool,
            desc='mode of operation (ground roll or flight)',
        )

    def setup(self):
        options = self.options

        nn = options['num_nodes']

        self.add_atmosphere()

        # NOTE: the following are potentially significant differences in implementation
        # between FLOPS and Aviary:
        #    - FLOPS detailed takeoff/landing assumes constant mass for the duration of
        #      that specific analysis.
        #    - Aviary implementation of FLOPS based detailed takeoff/landing will allow
        #      mass to vary as needed as a function of time and variation in related
        #      optimization control variables.
        self.add_subsystem(
            'stall_speed',
            StallSpeed(num_nodes=nn),
            promotes_inputs=[
                'mass',
                Dynamic.Atmosphere.DENSITY,
                ('area', Aircraft.Wing.AREA),
                ('lift_coefficient_max', self.stall_speed_lift_coefficient_name),
            ],
            promotes_outputs=[('stall_speed', 'v_stall')],
        )

        self.add_core_subsystems()

        self.add_external_subsystems()

        kwargs = {
            'num_nodes': nn,
            'climbing': options['climbing'],
            'friction_key': options['friction_key'],
            'aviary_options': options['aviary_options'],
        }

        self.add_subsystem(
            'takeoff_eom',
            TakeoffEOM(**kwargs),
            promotes_inputs=[
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ],
            promotes_outputs=[
                Dynamic.Mission.DISTANCE_RATE,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            ],
        )

        self.add_subsystem(
            'comp_v_ratio',
            om.ExecComp(
                'v_over_v_stall = v / v_stall',
                v_over_v_stall={'units': 'unitless', 'shape': nn},
                v={'units': 'm/s', 'shape': nn},
                # NOTE: FLOPS detailed takeoff stall speed is not dynamic - see above
                v_stall={'units': 'm/s', 'shape': nn},
                has_diag_partials=True,
            ),
            promotes_inputs=[('v', Dynamic.Mission.VELOCITY), 'v_stall'],
            promotes_outputs=['v_over_v_stall'],
        )

        self.set_input_defaults(Dynamic.Mission.ALTITUDE, np.zeros(nn), 'm')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, np.zeros(nn), 'm/s')
        self.set_input_defaults(Aircraft.Wing.AREA, 1.0, 'm**2')
