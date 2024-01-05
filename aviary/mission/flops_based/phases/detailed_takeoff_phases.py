'''
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
'''
from collections import namedtuple

import dymos as dm
import openmdao.api as om

from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.flops_based.phases.phase_builder_base import (
    InitialGuessControl, InitialGuessParameter, InitialGuessPolynomialControl,
    InitialGuessState, InitialGuessTime, PhaseBuilderBase)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Dynamic as _Dynamic
from aviary.variable_info.variables import Mission

Dynamic = _Dynamic.Mission


def _init_initial_guess_meta_data(cls: PhaseBuilderBase):
    '''
    Create default initial guess meta data preset with common items.
    '''
    cls._initial_guesses_meta_data_ = {}

    cls._add_initial_guess_meta_data(
        InitialGuessTime(),
        desc='initial guess for initial time and duration specified as a tuple')

    cls._add_initial_guess_meta_data(
        InitialGuessState('range'),
        desc='initial guess for horizontal distance traveled')

    cls._add_initial_guess_meta_data(
        InitialGuessState('velocity'),
        desc='initial guess for speed')

    cls._add_initial_guess_meta_data(
        InitialGuessState('mass'),
        desc='initial guess for mass')

    cls._add_initial_guess_meta_data(
        InitialGuessControl('throttle'),
        desc='initial guess for throttle')

    return cls


@_init_initial_guess_meta_data
class TakeoffBrakeReleaseToDecisionSpeed(PhaseBuilderBase):
    '''
    Define a phase builder for the first phase of takeoff, from brake release to decision
    speed, the maximum speed at which takeoff can be safely brought to full stop using
    zero thrust while braking.

    Attributes
    ----------
    name : str ('takeoff_brake_release')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (1000.0, 's')
            - duration_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_brake_release'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)

        phase.set_time_options(
            fix_initial=True, duration_bounds=(1, max_duration),
            duration_ref=duration_ref, units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=True, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=True, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=True, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        # TODO: Energy phase places this under an if num_engines > 0.
        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        phase.add_parameter('angle_of_attack', val=0.0, opt=False, units='deg')

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        transcription = dm.Radau(num_segments=3, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': False,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffBrakeReleaseToDecisionSpeed._add_meta_data('max_duration', val=1000.0, units='s')

TakeoffBrakeReleaseToDecisionSpeed._add_meta_data('duration_ref', val=10.0, units='s')

TakeoffBrakeReleaseToDecisionSpeed._add_meta_data('max_range', val=1000.0, units='ft')

TakeoffBrakeReleaseToDecisionSpeed._add_meta_data(
    'max_velocity', val=100.0, units='ft/s')

TakeoffBrakeReleaseToDecisionSpeed._add_initial_guess_meta_data(
    InitialGuessParameter('angle_of_attack'))


@_init_initial_guess_meta_data
class TakeoffDecisionSpeedToRotate(PhaseBuilderBase):
    '''
    Define a phase builder for the second phase of takeoff, from decision speed to
    rotation.

    Attributes
    ----------
    name : str ('takeoff_decision_speed')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (1000.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_decision_speed'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        # TODO: Energy phase places this under an if num_engines > 0.
        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        phase.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=1.2)

        phase.add_parameter('angle_of_attack', val=0.0, opt=False, units='deg')

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            'v_over_v_stall', output_name='v_over_v_stall', units='unitless'
        )

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        transcription = dm.Radau(num_segments=3, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': False,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffDecisionSpeedToRotate._add_meta_data('max_duration', val=1000.0, units='s')

TakeoffDecisionSpeedToRotate._add_meta_data('duration_ref', val=1.0, units='s')

TakeoffDecisionSpeedToRotate._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffDecisionSpeedToRotate._add_meta_data('max_range', val=1000.0, units='ft')

TakeoffDecisionSpeedToRotate._add_meta_data('max_velocity', val=100.0, units='ft/s')

TakeoffDecisionSpeedToRotate._add_initial_guess_meta_data(
    InitialGuessParameter('angle_of_attack'))


@_init_initial_guess_meta_data
class TakeoffDecisionSpeedBrakeDelay(TakeoffDecisionSpeedToRotate):
    '''
    Define a phase builder for the second phase of aborted takeoff, from decision speed
    to brake application.

    Attributes
    ----------
    name : str ('takeoff_decision_speed')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (1000.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_brake_delay'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)
        phase.set_time_options(fix_duration=True)
        return phase


TakeoffDecisionSpeedBrakeDelay._add_meta_data('max_duration', val=1000.0, units='s')

TakeoffDecisionSpeedBrakeDelay._add_meta_data('duration_ref', val=1.0, units='s')

TakeoffDecisionSpeedBrakeDelay._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffDecisionSpeedBrakeDelay._add_meta_data('max_range', val=1000.0, units='ft')

TakeoffDecisionSpeedBrakeDelay._add_meta_data('max_velocity', val=100.0, units='ft/s')

TakeoffDecisionSpeedBrakeDelay._add_initial_guess_meta_data(
    InitialGuessParameter('angle_of_attack'))


@_init_initial_guess_meta_data
class TakeoffRotateToLiftoff(PhaseBuilderBase):
    '''
    Define a phase builder for the third phase of takeoff, from rotation to liftoff.

    Attributes
    ----------
    name : str ('takeoff_rotate')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (5.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - max_angle_of_attack (10.0, 'deg')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_rotate'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        max_angle_of_attack, units = user_options.get_item('max_angle_of_attack')

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        phase.add_polynomial_control(
            'angle_of_attack', opt=True, units=units, order=1,
            lower=0, upper=max_angle_of_attack,
            ref=max_angle_of_attack)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        phase.add_timeseries_output(
            'v_over_v_stall', output_name='v_over_v_stall', units='unitless'
        )

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        transcription = dm.Radau(num_segments=3, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': False,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffRotateToLiftoff._add_meta_data('max_duration', val=5.0, units='s')

TakeoffRotateToLiftoff._add_meta_data('duration_ref', val=1.0, units='s')

TakeoffRotateToLiftoff._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffRotateToLiftoff._add_meta_data('max_range', val=1000.0, units='ft')

TakeoffRotateToLiftoff._add_meta_data('max_velocity', val=100.0, units='ft/s')

TakeoffRotateToLiftoff._add_meta_data('max_angle_of_attack', val=10.0, units='deg')

TakeoffRotateToLiftoff._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('angle_of_attack'))


@_init_initial_guess_meta_data
class TakeoffLiftoffToObstacle(PhaseBuilderBase):
    '''
    Define a phase builder for the fourth phase of takeoff, from liftoff to clearing the
    required obstacle.

    Attributes
    ----------
    name : str ('takeoff_liftoff')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - flight_path_angle_ref (5., 'deg')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack
            - altitude
            - Dynamic.Mission.FLIGHT_PATH_ANGLE

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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_liftoff'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        altitude_ref, units = user_options.get_item('altitude_ref')

        phase.add_state(
            Dynamic.ALTITUDE, fix_initial=True, lower=0, ref=altitude_ref,
            defect_ref=altitude_ref, units=units, upper=altitude_ref,
            rate_source=Dynamic.ALTITUDE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        flight_path_angle_ref, units = user_options.get_item('flight_path_angle_ref')

        phase.add_state(
            Dynamic.FLIGHT_PATH_ANGLE, fix_initial=True, lower=0,
            ref=flight_path_angle_ref, upper=flight_path_angle_ref,
            defect_ref=flight_path_angle_ref, units=units,
            rate_source=Dynamic.FLIGHT_PATH_ANGLE_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        lower_angle_of_attack, units = user_options.get_item('lower_angle_of_attack')
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            'angle_of_attack', opt=True, units=units,
            lower=lower_angle_of_attack, upper=upper_angle_of_attack,
            ref=angle_of_attack_ref)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        obstacle_height, units = aviary_options.get_item(
            Mission.Takeoff.OBSTACLE_HEIGHT)

        if obstacle_height is None:
            raise TypeError(
                f'missing required aviary_option: {Mission.Takeoff.OBSTACLE_HEIGHT}')

        airport_altitude = aviary_options.get_val(
            Mission.Takeoff.AIRPORT_ALTITUDE, units)

        h = obstacle_height + airport_altitude

        phase.add_boundary_constraint(
            Dynamic.ALTITUDE, loc='final', equals=h, ref=h, units=units, linear=True)

        phase.add_path_constraint(
            'v_over_v_stall', lower=1.25, ref=2.0)

        phase.add_boundary_constraint('eoms.forces_vertical', loc='initial', equals=0,
                                      ref=100000)

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': True,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffLiftoffToObstacle._add_meta_data('max_duration', val=100., units='s')

TakeoffLiftoffToObstacle._add_meta_data('duration_ref', val=1., units='s')

TakeoffLiftoffToObstacle._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffLiftoffToObstacle._add_meta_data('max_range', val=1000., units='ft')

TakeoffLiftoffToObstacle._add_meta_data('max_velocity', val=100., units='ft/s')

TakeoffLiftoffToObstacle._add_meta_data('altitude_ref', val=1., units='ft')

TakeoffLiftoffToObstacle._add_meta_data('flight_path_angle_ref', val=5., units='deg')

TakeoffLiftoffToObstacle._add_meta_data('lower_angle_of_attack', val=-10., units='deg')

TakeoffLiftoffToObstacle._add_meta_data('upper_angle_of_attack', val=15., units='deg')

TakeoffLiftoffToObstacle._add_meta_data('angle_of_attack_ref', val=10., units='deg')

TakeoffLiftoffToObstacle._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'))

TakeoffLiftoffToObstacle._add_initial_guess_meta_data(InitialGuessState('altitude'))

TakeoffLiftoffToObstacle._add_initial_guess_meta_data(
    InitialGuessState(Dynamic.FLIGHT_PATH_ANGLE))


@_init_initial_guess_meta_data
class TakeoffObstacleToMicP2(PhaseBuilderBase):
    '''
    Define a phase builder for the fifth phase of takeoff, from clearing the required
    obstacle to the P2 mic location. This phase is required for acoustic calculations.

    Attributes
    ----------
    name : str ('takeoff_climb')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - flight_path_angle_ref (5., 'deg')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')
            - mic_altitude (1.0, 'ft')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack
            - altitude
            - Dynamic.Mission.FLIGHT_PATH_ANGLE

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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_climb'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        altitude_ref, units = user_options.get_item('altitude_ref')

        phase.add_state(
            Dynamic.ALTITUDE, fix_initial=False, lower=0, ref=altitude_ref,
            defect_ref=altitude_ref, units=units,
            rate_source=Dynamic.ALTITUDE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        flight_path_angle_ref, units = user_options.get_item('flight_path_angle_ref')

        phase.add_state(
            Dynamic.FLIGHT_PATH_ANGLE, fix_initial=False, lower=0,
            ref=flight_path_angle_ref,
            defect_ref=flight_path_angle_ref, units=units,
            rate_source=Dynamic.FLIGHT_PATH_ANGLE_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        lower_angle_of_attack, units = user_options.get_item('lower_angle_of_attack')
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            'angle_of_attack', opt=True, units=units,
            lower=lower_angle_of_attack, upper=upper_angle_of_attack,
            ref=angle_of_attack_ref)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        final_alt, units = user_options.get_item('mic_altitude')

        airport_altitude = aviary_options.get_val(
            Mission.Takeoff.AIRPORT_ALTITUDE, units)

        h = final_alt + airport_altitude

        phase.add_boundary_constraint(
            Dynamic.ALTITUDE, loc='final', equals=h, ref=h, units=units, linear=True)

        phase.add_boundary_constraint(
            'v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        num_segments_climb = 7
        transcription = dm.Radau(num_segments=num_segments_climb, order=3,
                                 compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': True,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffObstacleToMicP2._add_meta_data('max_duration', val=100., units='s')

TakeoffObstacleToMicP2._add_meta_data('duration_ref', val=1., units='s')

TakeoffObstacleToMicP2._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffObstacleToMicP2._add_meta_data('max_range', val=1000., units='ft')

TakeoffObstacleToMicP2._add_meta_data('max_velocity', val=100., units='ft/s')

TakeoffObstacleToMicP2._add_meta_data('altitude_ref', val=1., units='ft')

TakeoffObstacleToMicP2._add_meta_data('flight_path_angle_ref', val=5., units='deg')

TakeoffObstacleToMicP2._add_meta_data('lower_angle_of_attack', val=-10., units='deg')

TakeoffObstacleToMicP2._add_meta_data('upper_angle_of_attack', val=15., units='deg')

TakeoffObstacleToMicP2._add_meta_data('angle_of_attack_ref', val=10., units='deg')

TakeoffObstacleToMicP2._add_meta_data('mic_altitude', val=1.0, units='ft')

TakeoffObstacleToMicP2._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'))

TakeoffObstacleToMicP2._add_initial_guess_meta_data(InitialGuessState('altitude'))

TakeoffObstacleToMicP2._add_initial_guess_meta_data(
    InitialGuessState(Dynamic.FLIGHT_PATH_ANGLE))


@_init_initial_guess_meta_data
class TakeoffMicP2ToEngineCutback(PhaseBuilderBase):
    '''
    Define a phase builder for the sixth phase of takeoff, from the P2 mic
    location to engine cutback. This phase is required for acoustic calculations.

    Attributes
    ----------
    name : str ('takeoff_climb')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - flight_path_angle_ref (5., 'deg')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')
            - final_range (1000.0, 'ft)

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack
            - altitude
            - Dynamic.Mission.FLIGHT_PATH_ANGLE

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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_climb'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        altitude_ref, units = user_options.get_item('altitude_ref')

        phase.add_state(
            Dynamic.ALTITUDE, fix_initial=False, lower=0, ref=altitude_ref,
            defect_ref=altitude_ref, units=units,
            rate_source=Dynamic.ALTITUDE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        flight_path_angle_ref, units = user_options.get_item('flight_path_angle_ref')

        phase.add_state(
            Dynamic.FLIGHT_PATH_ANGLE, fix_initial=False, lower=0,
            ref=flight_path_angle_ref,
            defect_ref=flight_path_angle_ref, units=units,
            rate_source=Dynamic.FLIGHT_PATH_ANGLE_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        lower_angle_of_attack, units = user_options.get_item('lower_angle_of_attack')
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            'angle_of_attack', opt=True, units=units,
            lower=lower_angle_of_attack, upper=upper_angle_of_attack,
            ref=angle_of_attack_ref)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        # start engine cutback phase at this range, where this phase ends
        # TODO: what is the difference between max_range and final_range?
        #    - should final_range replace max_range?
        #    - is there any reason to support both in this phase?
        final_range, units = user_options.get_item('final_range')

        phase.add_boundary_constraint(
            Dynamic.RANGE, loc='final', equals=final_range, ref=final_range,
            units=units, linear=True)

        phase.add_boundary_constraint(
            'v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        num_segments_climb = 7
        transcription = dm.Radau(num_segments=num_segments_climb, order=3,
                                 compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': True,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffMicP2ToEngineCutback._add_meta_data('max_duration', val=100., units='s')

TakeoffMicP2ToEngineCutback._add_meta_data('duration_ref', val=1., units='s')

TakeoffMicP2ToEngineCutback._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffMicP2ToEngineCutback._add_meta_data('max_range', val=1000., units='ft')

TakeoffMicP2ToEngineCutback._add_meta_data('max_velocity', val=100., units='ft/s')

TakeoffMicP2ToEngineCutback._add_meta_data('altitude_ref', val=1., units='ft')

TakeoffMicP2ToEngineCutback._add_meta_data('flight_path_angle_ref', val=5., units='deg')

TakeoffMicP2ToEngineCutback._add_meta_data(
    'lower_angle_of_attack', val=-10., units='deg')

TakeoffMicP2ToEngineCutback._add_meta_data('upper_angle_of_attack', val=15., units='deg')

TakeoffMicP2ToEngineCutback._add_meta_data('angle_of_attack_ref', val=10., units='deg')

TakeoffMicP2ToEngineCutback._add_meta_data('final_range', val=1000., units='ft')

TakeoffMicP2ToEngineCutback._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'))

TakeoffMicP2ToEngineCutback._add_initial_guess_meta_data(InitialGuessState('altitude'))

TakeoffMicP2ToEngineCutback._add_initial_guess_meta_data(
    InitialGuessState(Dynamic.FLIGHT_PATH_ANGLE))


@_init_initial_guess_meta_data
class TakeoffEngineCutback(PhaseBuilderBase):
    '''
    Define a phase builder for the seventh phase of takeoff, from start to
    finish of engine cutback. This phase is required for acoustic calculations.

    Attributes
    ----------
    name : str ('takeoff_climb')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - flight_path_angle_ref (5., 'deg')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack
            - altitude
            - Dynamic.Mission.FLIGHT_PATH_ANGLE

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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_climb'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        initial_ref, units = user_options.get_item('initial_ref')

        phase.set_time_options(
            fix_initial=False, fix_duration=True,
            initial_bounds=(1, initial_ref),
            initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        altitude_ref, units = user_options.get_item('altitude_ref')

        phase.add_state(
            Dynamic.ALTITUDE, fix_initial=False, lower=0, ref=altitude_ref,
            defect_ref=altitude_ref, units=units,
            rate_source=Dynamic.ALTITUDE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        flight_path_angle_ref, units = user_options.get_item('flight_path_angle_ref')

        phase.add_state(
            Dynamic.FLIGHT_PATH_ANGLE, fix_initial=False, lower=0,
            ref=flight_path_angle_ref,
            defect_ref=flight_path_angle_ref, units=units,
            rate_source=Dynamic.FLIGHT_PATH_ANGLE_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        lower_angle_of_attack, units = user_options.get_item('lower_angle_of_attack')
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            'angle_of_attack', opt=True, units=units,
            lower=lower_angle_of_attack, upper=upper_angle_of_attack,
            ref=angle_of_attack_ref)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        phase.add_boundary_constraint(
            'v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        num_segments_climb = 7
        transcription = dm.Radau(num_segments=num_segments_climb, order=3,
                                 compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': True,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffEngineCutback._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffEngineCutback._add_meta_data('max_range', val=1000., units='ft')

TakeoffEngineCutback._add_meta_data('max_velocity', val=100., units='ft/s')

TakeoffEngineCutback._add_meta_data('altitude_ref', val=1., units='ft')

TakeoffEngineCutback._add_meta_data('flight_path_angle_ref', val=5., units='deg')

TakeoffEngineCutback._add_meta_data('lower_angle_of_attack', val=-10., units='deg')

TakeoffEngineCutback._add_meta_data('upper_angle_of_attack', val=15., units='deg')

TakeoffEngineCutback._add_meta_data('angle_of_attack_ref', val=10., units='deg')

TakeoffEngineCutback._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'))

TakeoffEngineCutback._add_initial_guess_meta_data(InitialGuessState('altitude'))

TakeoffEngineCutback._add_initial_guess_meta_data(
    InitialGuessState(Dynamic.FLIGHT_PATH_ANGLE))


@_init_initial_guess_meta_data
class TakeoffEngineCutbackToMicP1(PhaseBuilderBase):
    '''
    Define a phase builder for the eighth phase of takeoff, from engine cutback
    to the P1 mic location. This phase is required for acoustic calculations.

    Attributes
    ----------
    name : str ('takeoff_climb')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - flight_path_angle_ref (5., 'deg')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')
            - mic_range (21325.0, 'ft')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack
            - altitude
            - Dynamic.Mission.FLIGHT_PATH_ANGLE

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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_climb'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        altitude_ref, units = user_options.get_item('altitude_ref')

        phase.add_state(
            Dynamic.ALTITUDE, fix_initial=False, lower=0, ref=altitude_ref,
            defect_ref=altitude_ref, units=units,
            rate_source=Dynamic.ALTITUDE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        flight_path_angle_ref, units = user_options.get_item('flight_path_angle_ref')

        phase.add_state(
            Dynamic.FLIGHT_PATH_ANGLE, fix_initial=False, lower=0,
            ref=flight_path_angle_ref,
            defect_ref=flight_path_angle_ref, units=units,
            rate_source=Dynamic.FLIGHT_PATH_ANGLE_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        lower_angle_of_attack, units = user_options.get_item('lower_angle_of_attack')
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            'angle_of_attack', opt=True, units=units,
            lower=lower_angle_of_attack, upper=upper_angle_of_attack,
            ref=angle_of_attack_ref)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        mic_range, units = user_options.get_item('mic_range')

        phase.add_boundary_constraint(
            Dynamic.RANGE, loc='final', equals=mic_range, ref=mic_range,
            units=units, linear=True)

        phase.add_boundary_constraint(
            'v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        num_segments_climb = 7
        transcription = dm.Radau(num_segments=num_segments_climb, order=3,
                                 compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': True,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffEngineCutbackToMicP1._add_meta_data('max_duration', val=100., units='s')

TakeoffEngineCutbackToMicP1._add_meta_data('duration_ref', val=1., units='s')

TakeoffEngineCutbackToMicP1._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffEngineCutbackToMicP1._add_meta_data('max_range', val=1000., units='ft')

TakeoffEngineCutbackToMicP1._add_meta_data('max_velocity', val=100., units='ft/s')

TakeoffEngineCutbackToMicP1._add_meta_data('altitude_ref', val=1., units='ft')

TakeoffEngineCutbackToMicP1._add_meta_data('flight_path_angle_ref', val=5., units='deg')

TakeoffEngineCutbackToMicP1._add_meta_data(
    'lower_angle_of_attack', val=-10., units='deg')

TakeoffEngineCutbackToMicP1._add_meta_data('upper_angle_of_attack', val=15., units='deg')

TakeoffEngineCutbackToMicP1._add_meta_data('angle_of_attack_ref', val=10., units='deg')

TakeoffEngineCutbackToMicP1._add_meta_data('mic_range', val=1000.0, units='ft')

TakeoffEngineCutbackToMicP1._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'))

TakeoffEngineCutbackToMicP1._add_initial_guess_meta_data(InitialGuessState('altitude'))

TakeoffEngineCutbackToMicP1._add_initial_guess_meta_data(
    InitialGuessState(Dynamic.FLIGHT_PATH_ANGLE))


@_init_initial_guess_meta_data
class TakeoffMicP1ToClimb(PhaseBuilderBase):
    '''
    Define a phase builder for the ninth phase of takeoff, from P1 mic
    location to climb. This phase is required for acoustic calculations.

    Attributes
    ----------
    name : str ('takeoff_climb')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - flight_path_angle_ref (5., 'deg')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')
            - mic_range (1000.0, 'ft')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - velocity
            - mass
            - throttle
            - angle_of_attack
            - altitude
            - Dynamic.Mission.FLIGHT_PATH_ANGLE

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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_climb'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        altitude_ref, units = user_options.get_item('altitude_ref')

        phase.add_state(
            Dynamic.ALTITUDE, fix_initial=False, lower=0, ref=altitude_ref,
            defect_ref=altitude_ref, units=units,
            rate_source=Dynamic.ALTITUDE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, lower=0, ref=max_velocity,
            defect_ref=max_velocity, units=units, upper=max_velocity,
            rate_source=Dynamic.VELOCITY_RATE)

        flight_path_angle_ref, units = user_options.get_item('flight_path_angle_ref')

        phase.add_state(
            Dynamic.FLIGHT_PATH_ANGLE, fix_initial=False, lower=0,
            ref=flight_path_angle_ref,
            defect_ref=flight_path_angle_ref, units=units,
            rate_source=Dynamic.FLIGHT_PATH_ANGLE_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        lower_angle_of_attack, units = user_options.get_item('lower_angle_of_attack')
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            'angle_of_attack', opt=True, units=units,
            lower=lower_angle_of_attack, upper=upper_angle_of_attack,
            ref=angle_of_attack_ref)

        phase.add_timeseries_output(
            Dynamic.DRAG, output_name=Dynamic.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.THRUST_TOTAL,
            output_name=Dynamic.THRUST_TOTAL, units='lbf'
        )

        mic_range, units = user_options.get_item('mic_range')

        phase.add_boundary_constraint(
            Dynamic.RANGE, loc='final', equals=mic_range, ref=mic_range,
            units=units, linear=True)

        phase.add_boundary_constraint(
            'v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        num_segments_climb = 7
        transcription = dm.Radau(num_segments=num_segments_climb, order=3,
                                 compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': True,
            'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


TakeoffMicP1ToClimb._add_meta_data('max_duration', val=100., units='s')

TakeoffMicP1ToClimb._add_meta_data('duration_ref', val=1., units='s')

TakeoffMicP1ToClimb._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffMicP1ToClimb._add_meta_data('max_range', val=1000., units='ft')

TakeoffMicP1ToClimb._add_meta_data('max_velocity', val=100., units='ft/s')

TakeoffMicP1ToClimb._add_meta_data('altitude_ref', val=1., units='ft')

TakeoffMicP1ToClimb._add_meta_data('flight_path_angle_ref', val=5., units='deg')

TakeoffMicP1ToClimb._add_meta_data('lower_angle_of_attack', val=-10., units='deg')

TakeoffMicP1ToClimb._add_meta_data('upper_angle_of_attack', val=15., units='deg')

TakeoffMicP1ToClimb._add_meta_data('angle_of_attack_ref', val=10., units='deg')

TakeoffMicP1ToClimb._add_meta_data('mic_range', val=1000., units='ft')

TakeoffMicP1ToClimb._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'))

TakeoffMicP1ToClimb._add_initial_guess_meta_data(InitialGuessState('altitude'))

TakeoffMicP1ToClimb._add_initial_guess_meta_data(
    InitialGuessState(Dynamic.FLIGHT_PATH_ANGLE))


@_init_initial_guess_meta_data
class TakeoffBrakeToAbort(PhaseBuilderBase):
    '''
    Define a phase builder for the last phase of aborted takeoff, from brake application
    to full stop.

    Attributes
    ----------
    name : str ('takeoff_abort')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (1000.0, 's')
            - duration_ref (1.0, 's')
            - initial_ref (10.0, 's')
            - max_range (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
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
    validate_options
    assign_default_options
    '''
    __slots__ = ()

    # region : derived type customization points
    _meta_data_ = {}

    default_name = 'takeoff_abort'

    default_ode_class = TakeoffODE
    # endregion : derived type customization points

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        max_duration, units = user_options.get_item('max_duration')
        duration_ref = user_options.get_val('duration_ref', units)
        initial_ref = user_options.get_val('initial_ref', units)

        phase.set_time_options(
            fix_initial=False, duration_bounds=(1, max_duration),
            initial_bounds=(1, initial_ref),
            duration_ref=duration_ref, initial_ref=initial_ref,
            units=units)

        max_range, units = user_options.get_item('max_range')

        phase.add_state(
            Dynamic.RANGE, fix_initial=False, lower=0, ref=max_range,
            defect_ref=max_range, units=units, upper=max_range,
            rate_source=Dynamic.RANGE_RATE)

        max_velocity, units = user_options.get_item('max_velocity')

        phase.add_state(
            Dynamic.VELOCITY, fix_initial=False, fix_final=True,
            lower=0, ref=max_velocity, upper=max_velocity,
            defect_ref=max_velocity, units=units,
            rate_source=Dynamic.VELOCITY_RATE)

        phase.add_state(
            Dynamic.MASS, fix_initial=False, fix_final=False,
            lower=0.0, upper=1e9, ref=5e4, defect_ref=5e4, units='kg',
            rate_source=Dynamic.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.MASS,
        )

        phase.add_control(
            Dynamic.THROTTLE,
            targets=Dynamic.THROTTLE, units='unitless',
            opt=False
        )

        phase.add_parameter('angle_of_attack', val=0.0, opt=False, units='deg')

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        transcription = dm.Radau(num_segments=3, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {
            'climbing': False,
            'friction_key': Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT}


TakeoffBrakeToAbort._add_meta_data('max_duration', val=1000.0, units='s')

TakeoffBrakeToAbort._add_meta_data('duration_ref', val=1.0, units='s')

TakeoffBrakeToAbort._add_meta_data('initial_ref', val=10.0, units='s')

TakeoffBrakeToAbort._add_meta_data('max_range', val=1000.0, units='ft')

TakeoffBrakeToAbort._add_meta_data('max_velocity', val=100.0, units='ft/s')

TakeoffBrakeToAbort._add_initial_guess_meta_data(
    InitialGuessParameter('angle_of_attack'))


class TakeoffTrajectory:
    '''
    Define a trajectory builder for detailed takeoff.

    Identify, collect, and call the necessary phase builders to create a typical takeoff
    trajectory.
    '''
    MappedPhase = namedtuple('MappedPhase', ('phase', 'phase_builder'))

    default_name = 'detailed_takeoff'

    def __init__(self, name=None):
        if name is None:
            name = self.default_name

        self.name = name

        self._brake_release_to_decision_speed = None
        self._decision_speed_to_rotate = None
        self._rotate_to_liftoff = None
        self._liftoff_to_obstacle = None
        self._obstacle_to_mic_p2 = None
        self._mic_p2_to_engine_cutback = None
        self._engine_cutback = None
        self._engine_cutback_to_mic_p1 = None
        self._mic_p1_to_climb = None
        self._decision_speed_to_brake = None
        self._brake_to_abort = None

        self._phases = {}
        self._traj = None

    def get_phase_names(self):
        '''
        Return a list of base names for available phases.
        '''
        keys = list(self._phases)

        return keys

    def get_phase(self, key) -> dm.Phase:
        '''
        Return the phase associated with the specified base name.

        Raises
        ------
        KeyError
            if the specified base name is not found
        '''
        mapped_phase: self.MappedPhase = self._phases[key]

        return mapped_phase.phase

    def set_brake_release_to_decision_speed(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the beginning of takeoff to the time when the pilot
        must choose either to liftoff or halt the aircraft.
        '''
        self._brake_release_to_decision_speed = phase_builder

    def set_decision_speed_to_rotate(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the short distance between achieving decision speed
        and beginning the rotation phase.
        '''
        self._decision_speed_to_rotate = phase_builder

    def set_rotate_to_liftoff(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the short distance required to rotate the aircraft
        to achieve liftoff.
        '''
        self._rotate_to_liftoff = phase_builder

    def set_liftoff_to_obstacle(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the short period between liftoff and clearing the
        required obstacle.
        '''
        self._liftoff_to_obstacle = phase_builder

    def set_obstacle_to_mic_p2(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the fifth phase of takeoff, from clearing the required
        obstacle to the p2 mic loation. This phase is required for acoustic calculations.
        '''
        self._obstacle_to_mic_p2 = phase_builder

    def set_mic_p2_to_engine_cutback(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the sixth phase of takeoff, from the p2 mic location
        to engine cutback. This phase is required for acoustic calculations.
        '''
        self._mic_p2_to_engine_cutback = phase_builder

    def set_engine_cutback(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the seventh phase of takeoff, from start to
        finish of engine cutback. This phase is required for acoustic calculations.
        '''
        self._engine_cutback = phase_builder

    def set_engine_cutback_to_mic_p1(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the eighth phase of takeoff, engine cutback
        to the P1 mic location. This phase is required for acoustic calculations.
        '''
        self._engine_cutback_to_mic_p1 = phase_builder

    def set_mic_p1_to_climb(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for the ninth phase of takeoff, from P1 mic
        location to climb. This phase is required for acoustic calculations.
        '''
        self._mic_p1_to_climb = phase_builder

    def set_decision_speed_to_brake(self, phase_builder: PhaseBuilderBase):
        '''
        Assign a phase builder for delayed braking when the engine fails.

        Note, this phase is optional. It is only required if balanced field length
        calculations are required.
        '''
        self._decision_speed_to_brake = phase_builder

    def set_brake_to_abort(
        self, phase_builder: PhaseBuilderBase, balanced_field_ref=8_000.
    ):
        '''
        Assign a phase builder for braking to fullstop after engine failure.

        Note, this phase is optional. It is only required if balanced field length
        calculations are required.

        Parameters
        ----------
        phase_builder : PhaseBuilderBase

        balanced_field_ref : float (8_000.0)
            the ref value to use for the linkage constraint for the final range
            between the liftoff-to-obstacle and the decision-speed-to-abort phases;

        Notes
        -----
        The default value for `balanced_field_ref` is appropriate total takeoff distances
        calculated in 'ft' for larger commercial passenger transports traveling in the
        continental United States. International travel of similar aircraft may require a
        larger value, while a smaller aircraft with a shorter range may require a smaller
        value.
        '''
        self._brake_to_abort = phase_builder
        self._balanced_field_ref = balanced_field_ref

    def build_trajectory(
        self, *, aviary_options: AviaryValues, model: om.Group = None,
        traj: dm.Trajectory = None
    ) -> dm.Trajectory:
        '''
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
        '''
        if traj is None:
            traj = dm.Trajectory()

        self._traj = traj

        self._add_phases(aviary_options)
        self._link_phases()

        if model is not None:
            phase_names = self.get_phase_names()

            setup_trajectory_params(model, traj, aviary_options, phase_names)

        return traj

    def apply_initial_guesses(self, prob: om.Problem, traj_name):
        '''
        Call `prob.set_val()` for states/parameters/etc. for each phase in this
        trajectory.

        Call only after `build_trajectory()` and `prob.setup()`.

        Returns
        -------
        not_applied : dict[str, list[str]]
            for any phase with missing initial guesses that cannot be applied, a list of
            those missing initial guesses; if a given phase has no missing initial
            guesses, the returned mapping will not contain the name of that phase
        '''
        not_applied = {}
        phase_builder: PhaseBuilderBase = None  # type hint

        for (phase, phase_builder) in self._phases.values():
            tmp = phase_builder.apply_initial_guesses(prob, traj_name, phase)

            if tmp:
                not_applied[phase_builder.name] = tmp

        return not_applied

    def _add_phases(self, aviary_options: AviaryValues):
        self._phases = {}

        self._add_phase(
            self._brake_release_to_decision_speed, aviary_options)

        self._add_phase(
            self._decision_speed_to_rotate, aviary_options)

        self._add_phase(
            self._rotate_to_liftoff, aviary_options)

        self._add_phase(
            self._liftoff_to_obstacle, aviary_options)

        obstacle_to_mic_p2 = self._obstacle_to_mic_p2

        if obstacle_to_mic_p2 is not None:
            self._add_phase(
                obstacle_to_mic_p2, aviary_options)

            self._add_phase(
                self._mic_p2_to_engine_cutback, aviary_options)

            self._add_phase(
                self._engine_cutback, aviary_options)

            self._add_phase(
                self._engine_cutback_to_mic_p1, aviary_options)

            self._add_phase(
                self._mic_p1_to_climb, aviary_options)

        decision_speed_to_brake = self._decision_speed_to_brake

        if decision_speed_to_brake is not None:
            self._add_phase(
                decision_speed_to_brake, aviary_options)

            self._add_phase(
                self._brake_to_abort, aviary_options)

    def _link_phases(self):
        traj: dm.Trajectory = self._traj

        brake_release_name = self._brake_release_to_decision_speed.name
        decision_speed_name = self._decision_speed_to_rotate.name

        basic_vars = ['time', 'range', 'velocity', 'mass']

        traj.link_phases([brake_release_name, decision_speed_name], vars=basic_vars)

        rotate_name = self._rotate_to_liftoff.name

        ext_vars = basic_vars + ['angle_of_attack']

        traj.link_phases([decision_speed_name, rotate_name], vars=ext_vars)

        liftoff_name = self._liftoff_to_obstacle.name

        traj.link_phases([rotate_name, liftoff_name], vars=ext_vars)

        obstacle_to_mic_p2 = self._obstacle_to_mic_p2

        if obstacle_to_mic_p2 is not None:
            obstacle_to_mic_p2_name = obstacle_to_mic_p2.name
            mic_p2_to_engine_cutback_name = self._mic_p2_to_engine_cutback.name
            engine_cutback_name = self._engine_cutback.name
            engine_cutback_to_mic_p1_name = self._engine_cutback_to_mic_p1.name
            mic_p1_to_climb_name = self._mic_p1_to_climb.name

            acoustics_vars = ext_vars + [Dynamic.FLIGHT_PATH_ANGLE, 'altitude']

            traj.link_phases(
                [liftoff_name, obstacle_to_mic_p2_name],
                vars=acoustics_vars)

            traj.link_phases(
                [obstacle_to_mic_p2_name, mic_p2_to_engine_cutback_name],
                vars=acoustics_vars)

            traj.link_phases(
                [mic_p2_to_engine_cutback_name, engine_cutback_name],
                vars=acoustics_vars)

            traj.link_phases(
                [engine_cutback_name, engine_cutback_to_mic_p1_name],
                vars=acoustics_vars)

            traj.link_phases(
                [engine_cutback_to_mic_p1_name, mic_p1_to_climb_name],
                vars=acoustics_vars)

        decision_speed_to_brake = self._decision_speed_to_brake

        if decision_speed_to_brake is not None:
            brake_name = decision_speed_to_brake.name
            abort_name = self._brake_to_abort.name

            traj.link_phases([brake_release_name, brake_name], vars=basic_vars)
            traj.link_phases([brake_name, abort_name], vars=basic_vars)

            traj.add_linkage_constraint(
                phase_a=abort_name, var_a='range', loc_a='final',
                phase_b=liftoff_name, var_b='range', loc_b='final',
                ref=self._balanced_field_ref)

    def _add_phase(self, phase_builder: PhaseBuilderBase, aviary_options: AviaryValues):
        name = phase_builder.name
        phase = phase_builder.build_phase(aviary_options)

        self._traj.add_phase(name, phase)

        self._phases[name] = self.MappedPhase(phase, phase_builder)
