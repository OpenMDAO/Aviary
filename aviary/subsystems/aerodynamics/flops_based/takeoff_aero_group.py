'''
Define utilities for calculating takeoff aerodynamics.
'''
from collections.abc import Sequence

import numpy as np
import openmdao.api as om
import scipy.constants as _units

from aviary.subsystems.aerodynamics.flops_based.ground_effect import \
    GroundEffect
from aviary.subsystems.aerodynamics.gasp_based.common import AeroForces
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TakeoffAeroGroup(om.Group):
    '''
    Define a group for calculating takeoff aerodynamics.
    '''

    def initialize(self):
        options = self.options

        options.declare('num_nodes', default=1, types=int, lower=0)

        options.declare(
            'ground_altitude', default=0., types=float,
            desc='true altitude of the ground from mean sea level (m)'
        )

        options.declare(
            'use_spoilers', default=False, types=bool,
            desc='true for spoilers deployed'
        )

        options.declare(
            'spoiler_drag_coefficient', default=0.0, types=float,
            desc='spoiler drag coefficitnt'
        )

        options.declare(
            'spoiler_lift_coefficient', default=0.0, types=float,
            desc='spoiler lift coefficitnt'
        )

        options.declare(
            'angles_of_attack', types=Sequence,
            desc='sequence of angles of attack (deg); at least two values required'
        )

        options.declare(
            'lift_coefficients', types=Sequence,
            desc='sequence of lift coefficients, one for each angle of attack'
        )

        options.declare(
            'drag_coefficients', types=Sequence,
            desc='sequence of drag coefficients, one for each angle of attack'
        )

        # NOTE: FLOPS did not enforce a lower bound.
        # - Should Aviary enforce a lower bound?
        options.declare(
            'lift_coefficient_factor', default=1., types=float,
            desc='factor for takeoff lift'
        )

        options.declare(
            'drag_coefficient_factor', default=1., types=float,
            desc='factor for takeoff drag'
        )

        options.declare(
            'landing_gear_up', default=False, types=bool,
            desc='true for retracted landing gear'
        )

        options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        options = self.options

        aviary_options = options['aviary_options']

        nn = options['num_nodes']
        angles_of_attack = np.array(options['angles_of_attack']) * _units.degree

        lift_coefficient_factor = options['lift_coefficient_factor']

        lift_coefficients = \
            np.array(options['lift_coefficients']) * lift_coefficient_factor

        drag_coefficient_factor = options['drag_coefficient_factor']

        drag_coefficients = \
            np.array(options['drag_coefficients']) * drag_coefficient_factor

        inputs = ['angle_of_attack']

        takeoff_polar: om.MetaModelSemiStructuredComp = self.add_subsystem(
            'takeoff_polar',
            om.MetaModelSemiStructuredComp(
                method='slinear', extrapolate=False, vec_size=nn
            ),
            promotes_inputs=inputs,
        )

        takeoff_polar.add_input('angle_of_attack', angles_of_attack, units='rad')

        takeoff_polar.add_output('lift_coefficient', lift_coefficients, units='unitless')
        takeoff_polar.add_output('drag_coefficient', drag_coefficients, units='unitless')

        ground_altitude = options['ground_altitude']

        kwargs = {
            'num_nodes': nn,
            'ground_altitude': ground_altitude,
        }

        inputs = [
            'angle_of_attack', Dynamic.Mission.ALTITUDE, Dynamic.Mission.FLIGHT_PATH_ANGLE,
            ('minimum_drag_coefficient', Mission.Takeoff.DRAG_COEFFICIENT_MIN),
            Aircraft.Wing.ASPECT_RATIO, Aircraft.Wing.HEIGHT,
            Aircraft.Wing.SPAN
        ]

        self.add_subsystem(
            'ground_effect', GroundEffect(**kwargs), promotes_inputs=inputs
        )

        self.connect(
            'takeoff_polar.lift_coefficient', 'ground_effect.base_lift_coefficient'
        )

        self.connect(
            'takeoff_polar.drag_coefficient', 'ground_effect.base_drag_coefficient'
        )

        f = f'climb_drag_coefficient = ground_effect_drag'

        if not options['landing_gear_up']:
            gear_drag = aviary_options.get_val(Aircraft.LandingGear.DRAG_COEFFICIENT)
            f = f + f' + {gear_drag}'

        if options['use_spoilers']:
            spoiler_drag = options['spoiler_drag_coefficient']
            spoiler_lift = options['spoiler_lift_coefficient']
            f = f + f' + {spoiler_drag}'

            self.add_subsystem(
                'add_extra_lift_coefficients',
                om.ExecComp(
                    f'climb_lift_coefficient = ground_effect_lift + {spoiler_lift}',
                    climb_lift_coefficient={'units': 'unitless', 'shape': nn},
                    ground_effect_lift={'units': 'unitless', 'shape': nn},
                    has_diag_partials=True,),
                promotes_inputs=['ground_effect_lift'],
                promotes_outputs=['climb_lift_coefficient'])

            self.connect('ground_effect.lift_coefficient', 'ground_effect_lift')
            self.connect('climb_lift_coefficient', 'aero_forces.CL')

        else:
            self.connect('ground_effect.lift_coefficient', 'aero_forces.CL')

        self.add_subsystem(
            'add_extra_drag_coefficients',
            om.ExecComp(
                f,
                climb_drag_coefficient={'units': 'unitless', 'shape': nn},
                ground_effect_drag={'units': 'unitless', 'shape': nn},
                has_diag_partials=True,),
            promotes_inputs=['ground_effect_drag'],
            promotes_outputs=['climb_drag_coefficient'])

        self.connect('ground_effect.drag_coefficient', 'ground_effect_drag')
        self.connect('climb_drag_coefficient', 'aero_forces.CD')

        inputs = [Dynamic.Mission.DYNAMIC_PRESSURE, Aircraft.Wing.AREA]
        outputs = [Dynamic.Mission.LIFT, Dynamic.Mission.DRAG]

        self.add_subsystem(
            'aero_forces', AeroForces(num_nodes=nn),
            promotes_inputs=inputs, promotes_outputs=outputs
        )

        self.set_input_defaults('angle_of_attack', np.zeros(nn), 'rad')
