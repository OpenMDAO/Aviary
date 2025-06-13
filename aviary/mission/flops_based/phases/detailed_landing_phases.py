"""
Define utilities for building detailed landing phases and the typical landing trajectory.

Classes
-------
LandingApproachToMicP3 : a phase builder for moving from descent to the mic location P3;
this phase is required for acoustic calculations

LandingMicP3ToObstacle : a phase builder for moving from the mic location P3 to the start
of the runway, just above the required clearance height; this phase is required for
acoustic calculations

LandingObstacleToFlare : a phase builder for moving from the start of the runway, just
above the required clearance height, to the start of a maneuver to help soften the impact
of touchdown

LandingFlareToTouchdown : a phase builder for moving through a maneuver to help soften
the impact of touchdown

LandingTouchdownToNoseDown : a phase builder for rotating the nose down after touchdown

LandingNoseDownToStop : a phase builder for the final phase of landing, from nose down to
full stop

LandingTrajectory : a trajectory builder for detailed landing
"""

import dymos as dm
import openmdao.api as om

from aviary.mission.flops_based.ode.landing_ode import FlareODE, LandingODE
from aviary.mission.flops_based.phases.detailed_takeoff_phases import (
    TakeoffTrajectory as _TakeoffTrajectory,
)
from aviary.mission.flops_based.phases.detailed_takeoff_phases import _init_initial_guess_meta_data
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
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


class LandingApproachToMicP3Options(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='max_duration',
            default=100.0,
            units='s',
            desc='Upper bound on duration for this phase.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='time_initial_ref',
            default=10.0,
            units='s',
            desc='Scale factor ref for the phase starting time.',
        )

        self.declare(
            name='distance_max', default=1000.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='max_velocity', default=100.0, units='ft/s', desc='Upper bound for velocity.'
        )

        self.declare(
            name='altitude_ref', default=1.0, units='ft', desc='Scale factor ref for altitude.'
        )

        self.declare(
            name='lower_angle_of_attack',
            types=tuple,
            default=-10.0,
            units='deg',
            desc='Lower bound for angle of attack.',
        )

        self.declare(
            name='upper_angle_of_attack',
            default=15.0,
            units='deg',
            desc='Upper bound for angle of attack.',
        )

        self.declare(
            name='angle_of_attack_ref',
            default=10.0,
            units='deg',
            desc='Scale factor ref for angle of attack.',
        )

        self.declare(
            name='initial_height', default=1.0, units='ft', desc='Starting altitude for thie phase.'
        )


@_init_initial_guess_meta_data
class LandingApproachToMicP3(PhaseBuilderBase):
    """
    Define a phase builder for moving from descent to the mic location P3. This phase is
    required for acoustic calculations.

    Attributes
    ----------
    name : str ('landing_approach')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - time_duration_ref (1.0, 's')
            - time_initial_ref (10.0, 's')
            - distance_max (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')
            - initial_height (1.0, 'ft')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - time
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
    """

    __slots__ = ()

    default_name = 'landing_approach'
    default_ode_class = LandingODE
    default_options_class = LandingApproachToMicP3Options

    def build_phase(self, aviary_options: AviaryValues = None):
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
        initial_ref = user_options.get_val('time_initial_ref', units)

        phase.set_time_options(
            duration_bounds=(1, max_duration),
            duration_ref=duration_ref,
            initial_ref=initial_ref,
            units=units,
        )

        distance_max, units = user_options['distance_max']

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=False,
            fix_final=False,
            upper=0,
            ref=distance_max,
            defect_ref=distance_max,
            units=units,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
        )

        altitude_ref, units = user_options['altitude_ref']

        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=False,
            fix_final=False,
            ref=altitude_ref,
            defect_ref=altitude_ref,
            units=units,
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
        )

        max_velocity, units = user_options['max_velocity']

        phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=False,
            fix_final=False,
            lower=0,
            ref=max_velocity,
            defect_ref=max_velocity,
            units=units,
            rate_source=Dynamic.Mission.VELOCITY_RATE,
        )

        phase.add_control(Dynamic.Mission.FLIGHT_PATH_ANGLE, opt=False, fix_initial=True)

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=True,
            fix_final=False,
            lower=0.0,
            ref=5e4,
            defect_ref=5e4,
            units='kg',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Vehicle.MASS,
        )

        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            opt=False,
        )

        units = 'deg'
        lower_angle_of_attack = user_options.get_val('lower_angle_of_attack', units)
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            opt=True,
            units=units,
            upper=upper_angle_of_attack,
            lower=lower_angle_of_attack,
            ref=angle_of_attack_ref,
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        initial_height, units = user_options['initial_height']

        airport_altitude = aviary_options.get_val(Mission.Landing.AIRPORT_ALTITUDE, units)

        h = initial_height + airport_altitude

        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc='initial',
            equals=h,
            ref=h,
            units=units,
            linear=True,
        )

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {'climbing': True, 'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


LandingApproachToMicP3._add_initial_guess_meta_data(
    InitialGuessControl(Dynamic.Vehicle.ANGLE_OF_ATTACK)
)

LandingApproachToMicP3._add_initial_guess_meta_data(InitialGuessState('altitude'))

LandingApproachToMicP3._add_initial_guess_meta_data(
    InitialGuessControl(Dynamic.Mission.FLIGHT_PATH_ANGLE)
)


# @_init_initial_guess_meta_data  # <--- inherited from base class
class LandingMicP3ToObstacle(LandingApproachToMicP3):
    """
    Define a phase builder for moving from the mic location P3 to the start of the
    runway, just above the required clearance height. This phase is required for acoustic
    calculations.

    Attributes
    ----------
    name : str ('landing_approach')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - time_duration_ref (1.0, 's')
            - time_initial_ref (10.0, 's')
            - distance_max (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')
            - initial_height (1.0, 'ft')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - time
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
    """

    __slots__ = ()

    default_name = 'landing_mic_p3'

    def build_phase(self, aviary_options: AviaryValues = None):
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

        # at the moment, these state options are the only differences between phases of
        # this class and phases of its base class
        phase.set_state_options(Dynamic.Mission.DISTANCE, fix_final=True)
        phase.set_state_options(Dynamic.Mission.VELOCITY, fix_final=True)
        phase.set_state_options(Dynamic.Vehicle.MASS, fix_initial=False)

        return phase


class LandingObstacleToFlareOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='max_duration',
            default=100.0,
            units='s',
            desc='Upper bound on duration for this phase.',
        )

        self.declare(
            name='distance_max', default=1000.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='max_velocity', default=100.0, units='ft/s', desc='Upper bound for velocity.'
        )

        self.declare(
            name='altitude_ref', default=1.0, units='ft', desc='Scale factor ref for altitude.'
        )


@_init_initial_guess_meta_data
class LandingObstacleToFlare(PhaseBuilderBase):
    """
    Define a phase builder for moving from the start of the runway, just above the
    required clearance height, to the start of a maneuver to help soften the impact of
    touchdown.

    Attributes
    ----------
    name : str ('landing_obstacle')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - distance_max (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - time
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
    """

    __slots__ = ()

    default_name = 'landing_obstacle'

    default_ode_class = LandingODE
    default_options_class = LandingObstacleToFlareOptions

    def build_phase(self, aviary_options: AviaryValues = None):
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

        phase.set_time_options(fix_initial=True, duration_bounds=(1, max_duration), units=units)

        distance_max, units = user_options['distance_max']

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=True,
            lower=0,
            ref=distance_max,
            defect_ref=distance_max,
            units=units,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
        )

        altitude_ref, units = user_options['altitude_ref']

        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=False,
            lower=0,
            ref=altitude_ref,
            defect_ref=altitude_ref,
            units=units,
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
        )

        max_velocity, units = user_options['max_velocity']

        phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=True,
            lower=0,
            ref=max_velocity,
            defect_ref=max_velocity,
            units=units,
            rate_source=Dynamic.Mission.VELOCITY_RATE,
        )

        phase.add_control(Dynamic.Mission.FLIGHT_PATH_ANGLE, opt=False, fix_initial=False)

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=False,
            fix_final=False,
            lower=0.0,
            ref=5e4,
            defect_ref=5e4,
            units='kg',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Vehicle.MASS,
        )

        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            opt=False,
        )

        phase.add_control(Dynamic.Vehicle.ANGLE_OF_ATTACK, opt=False, units='deg')

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        obstacle_height, units = aviary_options.get_item(Mission.Landing.OBSTACLE_HEIGHT)

        if obstacle_height is None:
            raise TypeError(f'missing required aviary_option: {Mission.Landing.OBSTACLE_HEIGHT}')

        airport_altitude = aviary_options.get_val(Mission.Landing.AIRPORT_ALTITUDE, units)

        h = obstacle_height + airport_altitude

        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc='initial',
            equals=h,
            ref=h,
            units=units,
            linear=True,
        )

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {'climbing': True, 'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


LandingObstacleToFlare._add_initial_guess_meta_data(
    InitialGuessControl(Dynamic.Vehicle.ANGLE_OF_ATTACK)
)

LandingObstacleToFlare._add_initial_guess_meta_data(InitialGuessState('altitude'))

LandingObstacleToFlare._add_initial_guess_meta_data(
    InitialGuessControl(Dynamic.Mission.FLIGHT_PATH_ANGLE)
)


class LandingFlareToTouchdownOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='max_duration',
            default=100.0,
            units='s',
            desc='Upper bound on duration for this phase.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='time_initial_ref',
            default=10.0,
            units='s',
            desc='Scale factor ref for the phase starting time.',
        )

        self.declare(
            name='distance_max', default=1000.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='max_velocity', default=100.0, units='ft/s', desc='Upper bound for velocity.'
        )

        self.declare(
            name='altitude_ref', default=1.0, units='ft', desc='Scale factor ref for altitude.'
        )

        self.declare(
            name='lower_angle_of_attack',
            types=tuple,
            default=-10.0,
            units='deg',
            desc='Lower bound for angle of attack.',
        )

        self.declare(
            name='upper_angle_of_attack',
            default=15.0,
            units='deg',
            desc='Upper bound for angle of attack.',
        )

        self.declare(
            name='angle_of_attack_ref',
            default=10.0,
            units='deg',
            desc='Scale factor ref for angle of attack.',
        )


@_init_initial_guess_meta_data
class LandingFlareToTouchdown(PhaseBuilderBase):
    """
    Define a phase builder for moving through a maneuver to help soften the impact of
    touchdown.

    Attributes
    ----------
    name : str ('landing_flare')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (100.0, 's')
            - time_duration_ref (1.0, 's')
            - time_initial_ref (10.0, 's')
            - distance_max (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - altitude_ref (1.0, 'ft')
            - lower_angle_of_attack (-10.0, 'deg')
            - upper_angle_of_attack (15.0, 'deg')
            - angle_of_attack_ref (10.0, 'deg')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - time
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
    """

    __slots__ = ()

    default_name = 'landing_flare'

    default_ode_class = FlareODE
    default_options_class = LandingFlareToTouchdownOptions

    def build_phase(self, aviary_options: AviaryValues = None):
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
        initial_ref = user_options.get_val('time_initial_ref', units)

        phase.set_time_options(
            fix_initial=False,
            duration_bounds=(1, max_duration),
            duration_ref=duration_ref,
            initial_ref=initial_ref,
            units=units,
        )

        distance_max, units = user_options['distance_max']

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=False,
            lower=0,
            ref=distance_max,
            defect_ref=distance_max,
            units=units,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
        )

        altitude_ref, units = user_options['altitude_ref']

        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=False,
            fix_final=True,
            lower=0,
            ref=altitude_ref,
            defect_ref=altitude_ref,
            units=units,
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
        )

        max_velocity, units = user_options['max_velocity']

        phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=False,
            lower=0,
            ref=max_velocity,
            defect_ref=max_velocity,
            units=units,
            rate_source=Dynamic.Mission.VELOCITY_RATE,
        )

        phase.add_control(Dynamic.Mission.FLIGHT_PATH_ANGLE, fix_initial=False, opt=False)

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=False,
            fix_final=False,
            lower=0.0,
            ref=5e4,
            defect_ref=5e4,
            units='kg',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Vehicle.MASS,
        )

        # TODO: Upper limit is a bit of a hack. It hopefully won't be needed if we
        # can get some other constraints working.
        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            lower=0.0,
            upper=0.2,
            opt=True,
        )

        units = 'deg'
        lower_angle_of_attack = user_options.get_val('lower_angle_of_attack', units)
        upper_angle_of_attack = user_options.get_val('upper_angle_of_attack', units)
        angle_of_attack_ref = user_options.get_val('angle_of_attack_ref', units)

        phase.add_control(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            control_type='polynomial',
            opt=True,
            units=units,
            order=1,
            lower=lower_angle_of_attack,
            upper=upper_angle_of_attack,
            ref=angle_of_attack_ref,
            rate_targets='angle_of_attack_rate',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        phase.add_timeseries_output('required_thrust', units='lbf')

        phase.add_timeseries_output('forces_perpendicular', units='lbf')

        # Since the control is linear, only need to constrain one point.
        phase.add_boundary_constraint('net_alpha_rate', equals=0.0, loc='final', units='deg/s')

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {}


LandingFlareToTouchdown._add_initial_guess_meta_data(
    InitialGuessPolynomialControl(Dynamic.Vehicle.ANGLE_OF_ATTACK)
)

LandingFlareToTouchdown._add_initial_guess_meta_data(InitialGuessState('altitude'))

LandingFlareToTouchdown._add_initial_guess_meta_data(
    InitialGuessControl(Dynamic.Mission.FLIGHT_PATH_ANGLE)
)


class LandingTouchdownToNoseDownOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='max_duration',
            default=5.0,
            units='s',
            desc='Upper bound on duration for this phase.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='time_initial_ref',
            default=10.0,
            units='s',
            desc='Scale factor ref for the phase starting time.',
        )

        self.declare(
            name='distance_max', default=1000.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='max_velocity', default=100.0, units='ft/s', desc='Upper bound for velocity.'
        )

        self.declare(
            name='altitude_ref', default=1.0, units='ft', desc='Scale factor ref for altitude.'
        )

        self.declare(
            name='max_angle_of_attack',
            default=10.0,
            units='deg',
            desc='Maximum angle of attack in this phase.',
        )


@_init_initial_guess_meta_data
class LandingTouchdownToNoseDown(PhaseBuilderBase):
    """
    Define a phase builder for rotating the nose down after touchdown.

    Attributes
    ----------
    name : str ('landing_touchdown')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration (5.0, 's')
            - time_duration_ref (1.0, 's')
            - time_initial_ref (10.0, 's')
            - distance_max (1000.0, 'ft')
            - max_velocity (100.0, 'ft/s')
            - max_angle_of_attack (10.0, 'deg')

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

    default_name = 'landing_touchdown'

    default_ode_class = LandingODE
    default_options_class = LandingTouchdownToNoseDownOptions

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
        initial_ref = user_options.get_val('time_initial_ref', units)

        phase.set_time_options(
            fix_initial=False,
            duration_bounds=(1, max_duration),
            duration_ref=duration_ref,
            initial_ref=initial_ref,
            units=units,
        )

        distance_max, units = user_options['distance_max']

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=False,
            lower=0,
            ref=distance_max,
            defect_ref=distance_max,
            units=units,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
        )

        max_velocity, units = user_options['max_velocity']

        phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=False,
            lower=0,
            ref=max_velocity,
            defect_ref=max_velocity,
            units=units,
            rate_source=Dynamic.Mission.VELOCITY_RATE,
        )

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=False,
            fix_final=False,
            lower=0.0,
            ref=5e4,
            defect_ref=5e4,
            units='kg',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Vehicle.MASS,
        )

        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            opt=False,
        )

        units = 'deg'
        max_angle_of_attack = user_options.get_val('max_angle_of_attack', units)

        phase.add_control(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            control_type='polynomial',
            opt=True,
            units=units,
            order=1,
            lower=0,
            upper=max_angle_of_attack,
            fix_final=True,
            fix_initial=False,
            ref=max_angle_of_attack,
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        transcription = dm.Radau(num_segments=3, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {'climbing': False, 'friction_key': Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT}


LandingTouchdownToNoseDown._add_initial_guess_meta_data(
    InitialGuessPolynomialControl(Dynamic.Vehicle.ANGLE_OF_ATTACK)
)


class LandingNoseDownToStopOptions(AviaryOptionsDictionary):
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
            name='time_initial_ref',
            default=10.0,
            units='s',
            desc='Scale factor ref for the phase starting time.',
        )

        self.declare(
            name='distance_max', default=1000.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='max_velocity', default=100.0, units='ft/s', desc='Upper bound for velocity.'
        )


@_init_initial_guess_meta_data
class LandingNoseDownToStop(PhaseBuilderBase):
    """
    Define a phase builder for the final phase of landing, from nose down to full stop.

    Attributes
    ----------
    name : str ('landing_stop')
        object label

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - max_duration
            - time_duration_ref
            - distance_max
            - max_velocity

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

    default_name = 'landing_stop'

    default_ode_class = LandingODE
    default_options_class = LandingNoseDownToStopOptions

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
        initial_ref = user_options.get_val('time_initial_ref', units)

        phase.set_time_options(
            fix_initial=False,
            duration_bounds=(1, max_duration),
            duration_ref=duration_ref,
            initial_ref=initial_ref,
            units=units,
        )

        distance_max, units = user_options['distance_max']

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=False,
            fix_final=False,
            lower=0,
            ref=distance_max,
            defect_ref=distance_max,
            units=units,
            rate_source=Dynamic.Mission.DISTANCE_RATE,
        )

        max_velocity, units = user_options['max_velocity']

        phase.add_state(
            Dynamic.Mission.VELOCITY,
            fix_initial=False,
            fix_final=True,
            lower=0,
            ref=max_velocity,
            defect_ref=max_velocity,
            units=units,
            rate_source=Dynamic.Mission.VELOCITY_RATE,
        )

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=False,
            fix_final=False,
            lower=0.0,
            ref=5e4,
            defect_ref=5e4,
            units='kg',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Vehicle.MASS,
        )

        # TODO: Energy phase places this under an if num_engines > 0.
        phase.add_control(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            targets=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
            opt=False,
        )

        phase.add_parameter(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=0.0, opt=False, units='deg')

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        transcription = dm.Radau(num_segments=3, order=3, compressed=True)

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {'climbing': False, 'friction_key': Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT}


LandingNoseDownToStop._add_initial_guess_meta_data(
    InitialGuessParameter(Dynamic.Vehicle.ANGLE_OF_ATTACK)
)


class LandingTrajectory:
    """
    Define a trajectory builder for detailed landing.

    Identify, collect, and call the necessary phase builders to create a typical landing
    trajectory.
    """

    MappedPhase = _TakeoffTrajectory.MappedPhase

    default_name = 'detailed_landing'

    def __init__(self, name=None):
        if name is None:
            name = self.default_name

        self.name = name

        self._approach_to_mic_p3 = None
        self._mic_p3_to_obstacle = None
        self._obstacle_to_flare = None
        self._flare_to_touchdown = None
        self._touchdown_to_nose_down = None
        self._nose_down_to_stop = None

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

    def set_approach_to_mic_p3(self, phase_builder: PhaseBuilderBase):
        """
        Assign a phase builder for moving from descent to the mic location P3. This phase
        is required for acoustic calculations.
        """
        self._approach_to_mic_p3 = phase_builder

    def set_mic_p3_to_obstacle(self, phase_builder: PhaseBuilderBase):
        """
        Assign a phase builder for moving from the mic location P3 to the start of the
        runway, just above the required clearance height. This phase is required for
        acoustic calculations.
        """
        self._mic_p3_to_obstacle = phase_builder

    def set_obstacle_to_flare(self, phase_builder: PhaseBuilderBase):
        """
        Assign a phase builder for moving from the start of the runway, just above the
        required clearance height, to the start of a maneuver to help soften the impact
        of touchdown.
        """
        self._obstacle_to_flare = phase_builder

    def set_flare_to_touchdown(self, phase_builder: PhaseBuilderBase):
        """
        Assign a phase builder for moving through a maneuver to help soften the impact of
        touchdown.
        """
        self._flare_to_touchdown = phase_builder

    def set_touchdown_to_nose_down(self, phase_builder: PhaseBuilderBase):
        """Assign a phase builder for rotating the nose down after touchdown."""
        self._touchdown_to_nose_down = phase_builder

    def set_nose_down_to_stop(self, phase_builder: PhaseBuilderBase):
        """
        Assign a phase builder for the final phase of landing, from nose down to full
        stop.
        """
        self._nose_down_to_stop = phase_builder

    def build_trajectory(
        self, *, aviary_options: AviaryValues, model: om.Group = None, traj: dm.Trajectory = None
    ) -> dm.Trajectory:
        """
        Return a new trajectory for detailed landing analysis.

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

            args = {'method': 'low_speed'}

            params = aero.get_parameters(aviary_options, **args)

            # takeoff introduces this one.
            params[Mission.Landing.LIFT_COEFFICIENT_MAX] = {
                'shape': (1,),
                'static_target': True,
            }

            ext_params = {}
            for phase in self._phases.keys():
                ext_params[phase] = params

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
        phase_builder: PhaseBuilderBase = None  # type hint

        for phase, phase_builder in self._phases.values():
            tmp = phase_builder.apply_initial_guesses(prob, traj_name, phase)

            if tmp:
                not_applied[phase_builder.name] = tmp

        return not_applied

    def _add_phases(self, aviary_options: AviaryValues):
        self._phases = {}

        approach_to_mic_p3 = self._approach_to_mic_p3

        if approach_to_mic_p3 is not None:
            self._add_phase(approach_to_mic_p3, aviary_options)
            self._add_phase(self._mic_p3_to_obstacle, aviary_options)

        self._add_phase(self._obstacle_to_flare, aviary_options)
        self._add_phase(self._flare_to_touchdown, aviary_options)
        self._add_phase(self._touchdown_to_nose_down, aviary_options)
        self._add_phase(self._nose_down_to_stop, aviary_options)

    def _link_phases(self):
        traj: dm.Trajectory = self._traj

        basic_vars = ['time', 'distance', 'velocity', 'mass']
        ext_vars = basic_vars + ['angle_of_attack']
        full_vars = ext_vars + ['altitude']

        obstacle_name = self._obstacle_to_flare.name

        approach_to_mic_p3 = self._approach_to_mic_p3

        if approach_to_mic_p3 is not None:
            approach_p3_name = approach_to_mic_p3.name
            p3_obstacle_name = self._mic_p3_to_obstacle.name

            obstacle_vars = ['mass', 'time', 'altitude', 'angle_of_attack']
            traj.link_phases([p3_obstacle_name, obstacle_name], vars=obstacle_vars)

            pre_obs_vars = obstacle_vars + ['distance', 'velocity']
            traj.link_phases([approach_p3_name, p3_obstacle_name], vars=pre_obs_vars)

        flare_name = self._flare_to_touchdown.name

        traj.link_phases([obstacle_name, flare_name], vars=full_vars)

        touchdown_name = self._touchdown_to_nose_down.name

        traj.link_phases([flare_name, touchdown_name], vars=ext_vars)

        nose_down_name = self._nose_down_to_stop.name

        traj.link_phases([touchdown_name, nose_down_name], vars=basic_vars)

    def _add_phase(self, phase_builder: PhaseBuilderBase, aviary_options: AviaryValues):
        name = phase_builder.name
        phase = phase_builder.build_phase(aviary_options)

        self._traj.add_phase(name, phase)

        self._phases[name] = self.MappedPhase(phase, phase_builder)
