import numpy as np

from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder import PhaseBuilder
from aviary.mission.phase_utils import add_subsystem_variables_to_phase
from aviary.mission.two_dof.ode.takeoff_ode import TakeOffODE
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class TakeoffPhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='num_segments',
            types=int,
            default=1,
            desc='The number of segments in transcription creation in Dymos. '
            'The default value is 1.',
        )

        self.declare(
            name='order',
            types=int,
            default=3,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos. The default value is 3.',
        )

        defaults = {
            'mass_bounds': (0.0, 200_000.0),
            'mass_ref': 100_000.0,
            'mass_defect_ref': 100.0,
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        defaults = {
            'time_duration_bounds': (1.0, 100.0),
        }
        self.add_time_options(units='s', defaults=defaults)

        # NOTE: All GASP phases before accel are in 'ft'.
        defaults = {
            'distance_bounds': (0.0, 10000.0),
            'distance_ref': 3000.0,
        }
        self.add_state_options('distance', units='ft', defaults=defaults)

        defaults = {
            'velocity_bounds': (0.0, 1000.0),
            'velocity_ref': 100.0,
        }
        self.add_state_options('velocity', units='kn', defaults=defaults)

        defaults = {
            'altitude_ref': 100.0,
            'altitude_bounds': (0.0, 700.0),
            'altitude_constraint_ref': 100.0,
        }
        self.add_state_options('altitude', units='ft', defaults=defaults)

        defaults = {
            'flight_path_angle_ref': np.deg2rad(1),
            'flight_path_angle_bounds': (-15 * np.pi / 180, 25.0 * np.pi / 180),
        }
        self.add_state_options('flight_path_angle', units='rad', defaults=defaults)

        defaults = {
            'angle_of_attack_ref': np.deg2rad(5),
            'angle_of_attack_bounds': (np.deg2rad(-30), np.deg2rad(30)),
            'angle_of_attack_optimize': True,
        }
        # NOTE: Sometimes alpha is a control, and sometimes a state. This actually works
        # for making sure both sets of options are in there.
        self.add_state_options('angle_of_attack', units='rad', defaults=defaults)
        self.add_control_options('angle_of_attack', units='rad', defaults=defaults)

        self.declare(
            'ground_roll',
            types=bool,
            default=False,
            desc='True if the aircraft is confined to the ground. Angle of attack is fixed and '
            'removed as an input.',
        )

        self.declare(
            'rotation',
            types=bool,
            default=False,
            desc='True if the aircraft is pitching up, but the rear wheels are still on the '
            'ground.',
        )

        self.declare(
            'reserve',
            types=bool,
            default=False,
            desc='Designate this phase as a reserve phase and contributes its fuel burn '
            'towards the reserve mission fuel requirements. Reserve phases should be '
            'be placed after all non-reserve phases in the phase_info.',
        )

        self.declare(
            name='target_distance',
            default=None,
            units='m',
            desc='The total distance traveled by the aircraft from takeoff to landing '
            'for the primary mission, not including reserve missions. This value must '
            'be positive.',
        )

        # The options below have not yet been revamped.

        self.declare(
            name='normal_ref',
            default=1.0,
            units='lbf',
            desc='Scale factor ref for the normal force constraint.',
        )

        self.declare(
            name='normal_ref0',
            default=0.0,
            units='lbf',
            desc='Scale factor ref0 for the normal force constraint.',
        )

        self.declare(
            name='pitch_constraint_bounds',
            default=(0.0, 15.0),
            types=tuple,
            units='deg',
            allow_none=True,
            desc='Tuple containing the lower and upper bounds of the pitch constraint, '
            'with unit string.',
        )

        self.declare(
            name='pitch_constraint_ref',
            default=1.0,
            units='deg',
            desc='Scale factor ref for the pitch constraint.',
        )


class TakeoffPhase(PhaseBuilder):
    """
    A phase builder for a two DOF takeoff phase.

    This can be used to build takeoff phases and includes support for ground roll and rotation.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilder.

    Methods
    -------
    Inherits all methods from PhaseBuilder.
    """

    default_name = 'takeoff_phase'
    default_ode_class = TakeOffODE
    default_options_class = TakeoffPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        ground_roll = user_options.get_val('ground_roll')
        rotation = user_options.get_val('rotation')
        pitch_constraint_bounds = user_options.get_val('pitch_constraint_bounds', units='deg')
        pitch_constraint_ref = user_options.get_val('pitch_constraint_ref', units='deg')

        # Add states
        if rotation:
            self.add_state(
                'angle_of_attack',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                'angle_of_attack_rate',
            )

        if not (ground_roll or rotation):
            self.add_state(
                'flight_path_angle',
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
            )
            self.add_state('altitude', Dynamic.Mission.ALTITUDE, Dynamic.Mission.ALTITUDE_RATE)

        self.add_state('velocity', Dynamic.Mission.VELOCITY, Dynamic.Mission.VELOCITY_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

        add_subsystem_variables_to_phase(phase, self.name, self.subsystems)

        # Add controls
        if not (ground_roll or rotation):
            self.add_control(
                'angle_of_attack',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            )

        # Add parameters
        # TODO: These are backdoor defaults.
        # Although, I can see how this is useful for keeping flaps/gear from engaging before
        # ascent.
        if ground_roll or rotation:
            t_init_gear = 100.0
            t_init_flaps = 100.0
        else:
            t_init_gear = 38.25
            t_init_flaps = 48.21

        phase.add_parameter(
            't_init_gear',
            units='s',
            static_target=True,
            opt=False,
            val=t_init_gear,
        )
        phase.add_parameter(
            't_init_flaps',
            units='s',
            static_target=True,
            opt=False,
            val=t_init_flaps,
        )

        # Add boundary constraints
        if rotation:
            normal_ref = user_options.get_val('normal_ref', units='lbf')
            normal_ref0 = user_options.get_val('normal_ref0', units='lbf')

            phase.add_boundary_constraint(
                'normal_force',
                loc='final',
                equals=0,
                units='lbf',
                ref=normal_ref,
                ref0=normal_ref0,
            )

        if not (ground_roll or rotation):

            phase.add_path_constraint('load_factor', upper=1.10, lower=0.0)

            phase.add_path_constraint(
                'fuselage_pitch',
                'theta',
                lower=pitch_constraint_bounds[0],
                upper=pitch_constraint_bounds[1],
                units='deg',
                ref=pitch_constraint_ref,
            )

        # Add timeseries outputs
        phase.add_timeseries_output('time', units='s', output_name='time')
        phase.add_timeseries_output(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')

        phase.add_timeseries_output('normal_force')
        phase.add_timeseries_output(Dynamic.Atmosphere.MACH)
        phase.add_timeseries_output('EAS', units='kn')

        phase.add_timeseries_output(Dynamic.Vehicle.LIFT)
        phase.add_timeseries_output('CL')
        phase.add_timeseries_output('CD')
        phase.add_timeseries_output('fuselage_pitch', output_name='theta', units='deg')

        return phase

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'ground_roll': self.user_options.get_val('ground_roll'),
            'rotation': self.user_options.get_val('rotation'),
        }


# Adding initial guess metadata
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(), desc='initial guess for time options'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for altitude state'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'), desc='initial guess for true airspeed state'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass state'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance state'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessState('flight_path_angle'), desc='initial guess for flight path angle state'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessState('angle_of_attack'), desc='initial guess for angle of attack state'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessControl('tau_gear'), desc='when the gear is retracted'
)
TakeoffPhase._add_initial_guess_meta_data(
    InitialGuessControl('tau_flaps'), desc='when the flaps are retracted'
)
