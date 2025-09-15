"""
Define utilities for building detailed takeoff phases and the typical takeoff trajectory.

Classes
-------
TakeoffBrakeReleaseToDecisionSpeed : a phase builder for the first phase of takeoff, from
brake release to decision speed, the maximum speed at which takeoff can be safely brought
to full stop using zero thrust while braking

TakeoffDecisionSpeedToRotate : a phase builder for the second phase of takeoff, from
decision speed to rotation

TakeoffDecisionSpeedBrakeDelay : a phase builder for the second phase of aborted takeoff,
from decision speed to brake application

TakeoffRotateToLiftoff : a phase builder for the third phase of takeoff, from rotation to
liftoff

TakeoffLiftoffToObstacle : a phase builder for the fourth phase of takeoff, from liftoff
to clearing the required obstacle

TakeoffObstacleToMicP2 : a phase builder for the fifth phase of takeoff, from
clearing the required obstacle to the P2 mic lication; this phase is required for
acoustic calculations

TakeoffMicP2ToEngineCutback : a phase builder for the sixth phase of takeoff, from the
P2 mic location to engine cutback; this phase is required for acoustic calculations

TakeoffEngineCutback : a phase builder for the seventh phase of takeoff, from
start to finish of engine cutback; this phase is required for acoustic calculations

TakeoffEngineCutbackToMicP1 : a phase builder for the eighth phase of takeoff, from
engine cutback to the P1 mic lication; this phase is required for acoustic calculations

TakeoffMicP1ToClimb : a phase builder for the ninth phase of takeoff, from
P1 mic location to climb; this phase is required for acoustic calculations

TakeoffBrakeToAbort : a phase builder for the last phase of aborted takeoff, from brake
application to full stop

TakeoffTrajectory : a trajectory builder for detailed takeoff
"""

from collections import namedtuple

import dymos as dm
import openmdao.api as om
from openmdao.solvers.solver import NonlinearSolver, LinearSolver

from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessParameter,
    InitialGuessPolynomialControl,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


# VR_RATIO = 1.18


def _init_initial_guess_meta_data(cls: PhaseBuilderBase):
    """Create default initial guess meta data preset with common items."""
    cls._initial_guesses_meta_data_ = {}

    cls._add_initial_guess_meta_data(
        InitialGuessIntegrationVariable(),
        desc='initial guess for initial time and duration specified as a tuple',
    )

    cls._add_initial_guess_meta_data(
        InitialGuessState('distance'), desc='initial guess for horizontal distance traveled'
    )

    cls._add_initial_guess_meta_data(InitialGuessState('velocity'), desc='initial guess for speed')

    cls._add_initial_guess_meta_data(InitialGuessState('mass'), desc='initial guess for mass')
    cls._add_initial_guess_meta_data(
        InitialGuessState(Dynamic.Mission.ALTITUDE), desc='initial guess for altitude'
    )

    cls._add_initial_guess_meta_data(
        InitialGuessControl('throttle'), desc='initial guess for throttle'
    )

    cls._add_initial_guess_meta_data(InitialGuessParameter(Dynamic.Vehicle.ANGLE_OF_ATTACK))

    cls._add_initial_guess_meta_data(InitialGuessParameter(Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE))

    cls._add_initial_guess_meta_data(InitialGuessParameter(Dynamic.Mission.FLIGHT_PATH_ANGLE))
    cls._add_initial_guess_meta_data(InitialGuessParameter('dV1'))
    cls._add_initial_guess_meta_data(InitialGuessParameter('dVEF'))

    return cls


class BalancedFieldPhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='max_duration',
            default=1000.0,
            units='s',
            desc='Upper bound on duration for this phase.',
        )

        self.declare(
            name='time_duration_ref', default=10.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='distance_max', default=1000.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='max_velocity', default=100.0, units='ft/s', desc='Upper bound for velocity.'
        )

        self.declare(
            name='terminal_condition',
            values=[
                'VEF',
                'V1',
                'VR',
                'LIFTOFF',
                'CLIMB_GRADIENT',
                'MAX_ALPHA',
                'OBSTACLE',
                'STOP',
            ],
            allow_none=True,
            default=None,
            desc='The condition which governs the end of the phase.',
        )

        self.declare(
            name='climbing',
            types=bool,
            default=False,
            desc='If False, assume aircraft is operating on the runway.',
        )

        self.declare(
            name='friction_key',
            types=str,
            default=Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
            desc='Friction Coefficient to use in the phase',
        )

        self.declare(
            name='pitch_control',
            values=['ALPHA_FIXED', 'ALPHA_RATE_FIXED', 'GAMMA_FIXED'],
            default='ALPHA_FIXED',
            desc='Specifies how alpha is controlled - Alpha can be a fixed parameter, specified via a fixed rate parameter, '
            'or whether the climb gradient is constant',
        )

        self.declare(
            name='nonlinear_solver',
            types=(NonlinearSolver,),
            allow_none=True,
            default=None,
            desc='Nonlinear solver applied to the phase if it needs to solve for the terminal speed condition.',
        )

        self.declare(
            name='linear_solver',
            types=(LinearSolver,),
            allow_none=True,
            default=None,
            desc='Linear solver applied to the phase if it needs to solve for the terminal speed condition.',
        )


@_init_initial_guess_meta_data
class BalancedFieldPhaseBuilder(PhaseBuilderBase):
    """
    Define a phase builder for detailed takeoff phases.

    Attributes
    ----------
    name : str ('takeoff_brake_release')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (1000.0, 's')
            - time_duration_ref (10.0, 's')
            - distance_max (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - time
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack

    ode_class : type (None)
        advanced: the type of system defining the ODE

    transcription : "Dymos transcription object" (None)
        advanced: an object providing the transcription technique of the
        optimal control problem

    default_name : str
        class attribute: derived type customization point; the default value
        for name

    default_ode_class : type
        class attribute: derived type customization point; the default value
        for ode_class used by build_phase

    Methods
    -------
    build_phase
    make_default_transcription
    """

    __slots__ = ()

    default_name = 'detailed_takeoff'
    default_ode_class = TakeoffODE
    default_options_class = BalancedFieldPhaseOptions

    def build_phase(self, aviary_options=None):
        """
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (empty)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options['max_duration']
        duration_ref = user_options.get_val('time_duration_ref', units)
        climbing = user_options['climbing']

        phase.set_time_options(
            fix_initial=True,
            duration_bounds=(0.001, max_duration),
            duration_ref=duration_ref,
            units=units,
        )

        distance_max, units = user_options['distance_max']

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=True,
            lower=0,
            ref=distance_max,
            defect_ref=distance_max,
            units=units,
            upper=distance_max,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
        )

        max_velocity, units = user_options['max_velocity']

        phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=True,
            lower=0,
            ref=max_velocity,
            defect_ref=max_velocity,
            units=units,
            upper=max_velocity,
            rate_source=Dynamic.Mission.VELOCITY_RATE,
        )

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=True,
            fix_final=False,
            lower=0.0,
            upper=1e9,
            ref=5e4,
            units='kg',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Vehicle.MASS,
        )

        if climbing:
            phase.add_state(
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                fix_initial=True,
                lower=0,
                ref=1.0,
                upper=1.5,
                defect_ref=1.0,
                units='rad',
                rate_source=Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            )

            phase.add_state(
                Dynamic.Mission.ALTITUDE,
                fix_initial=False,
                lower=0,
                ref=1.0,
                defect_ref=1.0,
                units='ft',
                rate_source=Dynamic.Mission.ALTITUDE_RATE,
            )

        # TODO: Energy phase places this under an if num_engines > 0.
        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            opt=False,
        )

        phase.add_parameter('VR_ratio', val=1.05, units='unitless', opt=False)

        if user_options['pitch_control'] == 'ALPHA_FIXED':
            phase.add_parameter(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=0.0, opt=False, units='deg')
        elif user_options['pitch_control'] == 'ALPHA_RATE_FIXED':
            phase.add_parameter(
                Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE, val=2.0, opt=False, units='deg/s'
            )
            phase.add_state(
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                fix_initial=True,
                fix_final=False,
                ref=1.0,
                units='deg',
                rate_source=Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE,
                targets=Dynamic.Vehicle.ANGLE_OF_ATTACK,
            )
        else:
            phase.add_timeseries_output(Dynamic.Vehicle.ANGLE_OF_ATTACK, units='deg')

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        # Define velocity to go based on definition of terminal speed, if applicable.
        terminal_condition = user_options['terminal_condition']
        if terminal_condition == 'V1':
            # Propagate until speed is the decision speed.
            # In balanced field applications, dV1 will be
            # set such that a rejected takeoff and a
            # takeoff to a 35 ft altitude for obstacle clearance
            # require the same range.
            phase.add_calc_expr(
                f'v_to_go = velocity - (VR_ratio * v_stall - dV1)',
                velocity={'units': 'kn'},
                dV1={'units': 'kn'},
                v_to_go={'units': 'kn'},
                v_stall={'units': 'kn'},
                VR_ratio={'units': 'unitless'},
            )
            phase.add_parameter(
                'dV1',
                opt=False,
                val=1.0,
                units='kn',
                desc='Decision speed delta below rotation speed.',
            )
            phase.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
        elif terminal_condition == 'VR':
            # Propagate until speed is the rotation speed.
            phase.add_calc_expr(
                'v_to_go = velocity - (VR_ratio * v_stall)',
                velocity={'units': 'kn'},
                VR_ratio={'units': 'unitless'},
                v_stall={'units': 'kn'},
                v_to_go={'units': 'kn'},
            )
            phase.add_boundary_balance(
                param='t_duration',
                name='v_to_go',
                tgt_val=0.0,
                loc='final',
                lower=-1000,
                upper=1000,
                ref=10.0,
            )
        elif terminal_condition == 'VEF':
            # Propagate until engine failure.
            # Note we expect dVEF to be negative here.
            # In balanced field applications, dVEF will be set
            # such that it occurs {pilot_reaction_time} seconds
            # before V1.
            phase.add_calc_expr(
                'v_to_go = velocity - (VR_ratio * v_stall - dV1 - dVEF)',
                velocity={'units': 'kn'},
                VR_ratio={'units': 'unitless'},
                dV1={'units': 'kn'},
                v_to_go={'units': 'kn'},
                v_stall={'units': 'kn'},
                dVEF={'units': 'kn'},
            )
            phase.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
            phase.add_parameter(
                'dVEF',
                opt=False,
                val=3.0,
                units='kn',
                desc='Engine failure speed delta below decision speed.',
            )
            phase.add_parameter(
                'dV1',
                opt=False,
                val=1.0,
                units='kn',
                desc='Decision speed delta below rotation speed.',
            )
        elif terminal_condition == 'MAX_ALPHA':
            # Propagate until velocity is the desired literal value.
            phase.add_boundary_balance(
                param='t_duration',
                name=Dynamic.Vehicle.ANGLE_OF_ATTACK,
                tgt_val=12.0,
                lower=0.0001,
                upper=1000,
                eq_units='deg',
                loc='final',
            )
        elif terminal_condition == 'LIFTOFF':
            # Propagate until velocity is the desired literal value.
            phase.add_boundary_balance(
                param='t_duration',
                name='takeoff_eom.ground_normal_force',
                tgt_val=0.0,
                lower=0.0001,
                upper=1000,
                eq_units='MN',
                loc='final',
            )
        elif terminal_condition == 'CLIMB_GRADIENT':
            phase.add_calc_expr(
                'climb_gradient = tan(flight_path_angle)',
                climb_gradient={'units': 'unitless'},
                flight_path_angle={'units': 'rad'},
            )
            phase.add_boundary_balance(
                param='t_duration',
                name='climb_gradient',
                tgt_val=0.024,
                lower=0.0001,
                upper=1000,
                eq_units='unitless',
                loc='final',
            )
        elif terminal_condition == 'OBSTACLE':
            phase.add_boundary_balance(
                param='t_duration',
                name=Dynamic.Mission.ALTITUDE,
                tgt_val=35,
                lower=0.0001,
                upper=1000,
                eq_units='ft',
                loc='final',
            )
        elif terminal_condition == 'STOP':
            phase.add_boundary_balance(
                param='t_duration',
                name=Dynamic.Mission.VELOCITY,
                tgt_val=5.0,
                lower=0.0001,
                upper=1000,
                eq_units='kn',
                loc='final',
            )
        elif terminal_condition is None:
            # Propagate for t_duration
            pass
        else:
            raise ValueError(
                f'Unrecognized value for terminal_speed ({terminal_condition}).'
                'Must be one of "VEF", "VR", "V1" or a literal numerical value.'
            )

        if phase.boundary_balance_options:
            phase.options['auto_order'] = True

        if user_options['nonlinear_solver'] is not None:
            phase.nonlinear_solver = user_options['nonlinear_solver']

        if user_options['linear_solver'] is not None:
            phase.linear_solver = user_options['linear_solver']

        phase.add_timeseries_output(
            'v_over_v_stall', output_name='v_over_v_stall', units='unitless'
        )

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        transcription = dm.PicardShooting(num_segments=1, nodes_per_seg=10)

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {
            'climbing': self.user_options['climbing'],
            'pitch_control': self.user_options['pitch_control'],
            'friction_key': self.user_options['friction_key'],
        }
