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
from aviary.variable_info.variables import Dynamic, Mission


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
        InitialGuessControl('throttle'), desc='initial guess for throttle'
    )

    cls._add_initial_guess_meta_data(
        InitialGuessParameter(Dynamic.Vehicle.ANGLE_OF_ATTACK)
    )

    cls._add_initial_guess_meta_data(
        InitialGuessParameter(Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE)
    )

    cls._add_initial_guess_meta_data(
        InitialGuessParameter(Dynamic.Mission.FLIGHT_PATH_ANGLE)
    )

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
            values=['VEF', 'V1', 'VR', 'LIFTOFF', 'CLIMB_GRADIENT', 'OBSTACLE'],
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
            name='pitch_control',
            values=['alpha_fixed', 'alpha_rate_fixed', 'gamma_fixed'],
            default='alpha_fixed',
            desc='Specifies how alpha is controlled - Alpha can be a fixed parameter, specified via a fixed rate parameter, ' \
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

        if user_options['pitch_control'] == 'alpha_fixed':
            phase.add_parameter(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=0.0, opt=False, units='deg')
        elif user_options['pitch_control'] == 'alpha_rate_fixed':
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
                'v_to_go = velocity - (dV1 + v_stall)',
                velocity={'units': 'kn'},
                dV1={'units': 'kn'},
                v_to_go={'units': 'kn'},
                v_stall={'units': 'kn'},
            )
            phase.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
        elif terminal_condition == 'VR':
            # Propagate until speed is the rotation speed.
            # phase.add_calc_expr(
            #     'v_to_go = velocity - (dVR + dV1 + v_stall)',
            #     velocity={'units': 'kn'},
            #     dv1={'units': 'kn'},
            #     dVR={'units': 'kn'},
            #     v_to_go={'units': 'kn'},
            # )
            pass
            phase.add_boundary_balance(
                param='t_duration', name='v_over_v_stall', tgt_val=1.05, loc='final'
            )
        elif terminal_condition == 'VEF':
            # Propagate until engine failure.
            # Note we expect dVEF to be negative here.
            # In balanced field applications, dVEF will be set
            # such that it occurs {pilot_reaction_time} seconds
            # before V1.
            phase.add_calc_expr(
                'v_to_go = velocity - (dVEF + dV1 + v_stall)',
                velocity={'units': 'kn'},
                dV1={'units': 'kn'},
                dVEF={'units': 'kn'},
                v_to_go={'units': 'kn'},
            )
            phase.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
        elif terminal_condition == 'LIFTOFF':
            # Propagate until velocity is the desired literal value.
            phase.add_boundary_balance(
                param='t_duration',
                name='takeoff_eom.ground_normal_force',
                tgt_val=0.0,
                lower=0.0001,
                upper=20,
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
                upper=20,
                eq_units='unitless',
                loc='final',
            )
        elif terminal_condition == 'OBSTACLE':
            phase.add_boundary_balance(
                param='t_duration',
                name=Dynamic.Mission.ALTITUDE,
                tgt_val=35,
                lower=0.0001,
                upper=100,
                eq_units='ft',
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

        phase.add_parameter(
            'dV1', opt=False, val=10.0, units='kn', desc='Decision speed delta above stall speed.'
        )
        # phase.add_parameter('dVEF', opt=False, val=10.0, desc='Decision speed delta below decision speed.')

        if phase.boundary_balance_options:
            phase.options['auto_order'] = True

        if user_options['nonlinear_solver'] is not None:
            phase.nonlinear_solver = user_options['nonlinear_solver']

        if user_options['linear_solver'] is not None:
            phase.linear_solver = user_options['linear_solver']

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
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
        }


class BalancedFieldTrajectoryBuilder:
    """
    Define a trajectory builder for detailed takeoff.

    Identify, collect, and call the necessary phase builders to create a typical takeoff
    trajectory.
    """
    MappedPhase = namedtuple('MappedPhase', ('phase', 'phase_builder'))

    default_name = 'detailed_takeoff'

    def __init__(self, name=None):
        if name is None:
            name = self.default_name

        self.name = name

        self._brake_release_to_decision_speed = None

        self._phases = {}
        self._traj = None

    def get_phase_names(self):
        """Return a list of base names for available phases."""
        keys = list(self._phases)

        return keys

    def get_phase(self, key) -> dm.Phase:
        """
        Return the phase associated with the specified base name.

        Raises
        ------
        KeyError
            if the specified base name is not found
        """
        mapped_phase = self._phases[key]

        return mapped_phase.phase

    def set_brake_release_to_decision_speed(self, phase_builder: PhaseBuilderBase):
        """
        Assign a phase builder for the beginning of takeoff to the time when the pilot
        must choose either to liftoff or halt the aircraft.
        """
        self._brake_release_to_decision_speed = phase_builder

    # def set_decision_speed_to_rotate(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the short distance between achieving decision speed
    #     and beginning the rotation phase.
    #     """
    #     self._decision_speed_to_rotate = phase_builder

    # def set_rotate_to_liftoff(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the short distance required to rotate the aircraft
    #     to achieve liftoff.
    #     """
    #     self._rotate_to_liftoff = phase_builder

    # def set_liftoff_to_climb_gradient(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign the phase builder for the phase from liftoff until the required climb gradient is reached.

    #     Args:
    #         phase_builder (PhaseBuilderBase): _description_
    #     """
    #     self._liftoff_to_climb_gradient = phase_builder

    # def set_climb_gradient_to_obstacle(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign the phase builder for the phase from achievement of climb gradient until obstacle clearance.

    #     Args:
    #         phase_builder (PhaseBuilderBase): _description_
    #     """
    #     self._climb_gradient_to_obstacle = phase_builder

    # def set_liftoff_to_obstacle(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the short period between liftoff and clearing the
    #     required obstacle.
    #     """
    #     self._liftoff_to_obstacle = phase_builder

    # def set_obstacle_to_mic_p2(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the fifth phase of takeoff, from clearing the required
    #     obstacle to the p2 mic loation. This phase is required for acoustic calculations.
    #     """
    #     self._obstacle_to_mic_p2 = phase_builder

    # def set_mic_p2_to_engine_cutback(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the sixth phase of takeoff, from the p2 mic location
    #     to engine cutback. This phase is required for acoustic calculations.
    #     """
    #     self._mic_p2_to_engine_cutback = phase_builder

    # def set_engine_cutback(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the seventh phase of takeoff, from start to
    #     finish of engine cutback. This phase is required for acoustic calculations.
    #     """
    #     self._engine_cutback = phase_builder

    # def set_engine_cutback_to_mic_p1(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the eighth phase of takeoff, engine cutback
    #     to the P1 mic location. This phase is required for acoustic calculations.
    #     """
    #     self._engine_cutback_to_mic_p1 = phase_builder

    # def set_mic_p1_to_climb(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for the ninth phase of takeoff, from P1 mic
    #     location to climb. This phase is required for acoustic calculations.
    #     """
    #     self._mic_p1_to_climb = phase_builder

    # def set_decision_speed_to_brake(self, phase_builder: PhaseBuilderBase):
    #     """
    #     Assign a phase builder for delayed braking when the engine fails.

    #     Note, this phase is optional. It is only required if balanced field length
    #     calculations are required.
    #     """
    #     self._decision_speed_to_brake = phase_builder

    # def set_brake_to_abort(self, phase_builder: PhaseBuilderBase, balanced_field_ref=8_000.0):
    #     """
    #     Assign a phase builder for braking to fullstop after engine failure.

    #     Note, this phase is optional. It is only required if balanced field length
    #     calculations are required.

    #     Parameters
    #     ----------
    #     phase_builder : PhaseBuilderBase

    #     balanced_field_ref : float (8_000.0)
    #         the ref value to use for the linkage constraint for the final range
    #         between the liftoff-to-obstacle and the decision-speed-to-abort phases;

    #     Notes
    #     -----
    #     The default value for `balanced_field_ref` is appropriate total takeoff distances
    #     calculated in 'ft' for larger commercial passenger transports traveling in the
    #     continental United States. International travel of similar aircraft may require a
    #     larger value, while a smaller aircraft with a shorter range may require a smaller
    #     value.
    #     """
    #     self._brake_to_abort = phase_builder
    #     self._balanced_field_ref = balanced_field_ref

    def build_trajectory(
        self, *, aviary_options: AviaryValues, model: om.Group = None, traj: dm.Trajectory = None
    ) -> dm.Trajectory:
        """
        Return a new trajectory for detailed takeoff analysis.

        Call only after assigning phase builders for required phases.

        Parameters
        ----------
        aviary_options : AviaryValues
            collection of Aircraft/Mission specific options

        model : openmdao.api.Group (None)
            the model handling trajectory parameter setup; if `None`, trajectory
            parameter setup will not be handled

        traj : dymos.Trajectory (None)
            the trajectory to update; if `None`, a new trajetory will be updated and
            returned

        Returns
        -------
        the updated trajectory; if the specified trajectory is `None`, a new trajectory
        will be updated and returned

        Notes
        -----
        Do not modify this object or any of its referenced data between the call to
        `build_trajectory()` and the call to `apply_initial_guesses()`, or the behavior
        is undefined, no diagnostic required.
        """
        if traj is None:
            traj = dm.Trajectory()

        self._traj = traj

        self._add_phases(aviary_options)
        self._link_phases()

        if model is not None:
            phase_names = self.get_phase_names()

            # This is a level 3 method that uses the default subsystems.
            # We need to create parameters for just the inputs we have.
            # They mostly come from the low-speed aero subsystem.

            aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, LegacyCode('FLOPS'))

            kwargs = {'method': 'low_speed'}

            params = aero.get_parameters(aviary_options, **kwargs)

            # takeoff introduces this one.
            params[Mission.Takeoff.LIFT_COEFFICIENT_MAX] = {
                'shape': (1,),
                'static_target': True,
            }

            ext_params = {}
            for phase in self._phases.keys():
                ext_params[phase] = params

            print(ext_params)
            exit(0)

            setup_trajectory_params(
                model, traj, aviary_options, phase_names, external_parameters=ext_params
            )

        return traj

    def apply_initial_guesses(self, prob: om.Problem, traj_name):
        """
        Call `prob.set_val()` for states/parameters/etc. for each phase in this
        trajectory.

        Call only after `build_trajectory()` and `prob.setup()`.

        Returns
        -------
        not_applied : dict[str, list[str]]
            for any phase with missing initial guesses that cannot be applied, a list of
            those missing initial guesses; if a given phase has no missing initial
            guesses, the returned mapping will not contain the name of that phase
        """
        not_applied = {}
        phase_builder: PhaseBuilderBase = None

        for phase, phase_builder in self._phases.values():
            tmp = phase_builder.apply_initial_guesses(prob, traj_name, phase)

            if tmp:
                not_applied[phase_builder.name] = tmp

        return not_applied

    def _add_phases(self, aviary_options: AviaryValues):
        self._phases = {}

        self._add_phase(self._brake_release_to_decision_speed, aviary_options)

        # self._add_phase(self._decision_speed_to_rotate, aviary_options)

        # self._add_phase(self._rotate_to_liftoff, aviary_options)

        # self._add_phase(self._liftoff_to_climb_gradient, aviary_options)

        # self._add_phase(self._climb_gradient_to_obstacle, aviary_options)

        # obstacle_to_mic_p2 = self._obstacle_to_mic_p2

        # if obstacle_to_mic_p2 is not None:
        #     self._add_phase(obstacle_to_mic_p2, aviary_options)

        #     self._add_phase(self._mic_p2_to_engine_cutback, aviary_options)

        #     self._add_phase(self._engine_cutback, aviary_options)

        #     self._add_phase(self._engine_cutback_to_mic_p1, aviary_options)

        #     self._add_phase(self._mic_p1_to_climb, aviary_options)

        # decision_speed_to_brake = self._decision_speed_to_brake

        # if decision_speed_to_brake is not None:
        #     self._add_phase(decision_speed_to_brake, aviary_options)

        #     self._add_phase(self._brake_to_abort, aviary_options)

    def _link_phases(self):
        traj: dm.Trajectory = self._traj

        # brake_release_name = self._brake_release_to_decision_speed.name
        # decision_speed_name = self._decision_speed_to_rotate.name
        # rotate_name = self._rotate_to_liftoff.name
        # liftoff_name = self._liftoff_to_climb_gradient.name
        # climb_gradient_name = self._climb_gradient_to_obstacle.name

        # traj.link_phases(
        #     [brake_release_name, decision_speed_name, rotate_name, liftoff_name, climb_gradient_name],
        #     vars=['time', 'distance', 'velocity', 'mass'],
        #     connected=True,
        # )

        # traj.link_phases([rotate_name, liftoff_name], vars=['angle_of_attack',], connected=True)

        # traj.link_phases([liftoff_name, climb_gradient_name], vars=['flight_path_angle', 'altitude'], connected=True)

    def _add_phase(self, phase_builder: PhaseBuilderBase, aviary_options: AviaryValues):
        name = phase_builder.name
        phase = phase_builder.build_phase(aviary_options)

        self._traj.add_phase(name, phase)

        self._phases[name] = self.MappedPhase(phase, phase_builder)
