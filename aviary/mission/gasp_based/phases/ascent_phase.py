import numpy as np

from aviary.mission.gasp_based.ode.ascent_ode import AscentODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class AscentPhaseOptions(AviaryOptionsDictionary):
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
            default=None,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos.',
        )

        defaults = {
            'mass_ref': 100_000.0,
            'mass_defect_ref': 1.0e2,
            'mass_bounds': (0.0, 190_000.0),
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        # NOTE: All GASP phases before accel are in 'ft'.
        defaults = {
            'distance_ref': 3000.0,
            'distance_bounds': (0.0, 10.0e3),
        }
        self.add_state_options('distance', units='ft', defaults=defaults)

        defaults = {
            'velocity_ref': 1.0e2,
            'velocity_bounds': (0.0, 1000.0),
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
            'flight_path_angle_defect_ref': 0.01,
            'flight_path_angle_bounds': (-15 * np.pi / 180, 25.0 * np.pi / 180),
        }
        self.add_state_options('flight_path_angle', units='rad', defaults=defaults)

        defaults = {
            'angle_of_attack_ref': np.deg2rad(5),
            'angle_of_attack_bounds': (np.deg2rad(-30), np.deg2rad(30)),
            'angle_of_attack_optimize': True,
        }
        self.add_control_options('angle_of_attack', units='rad', defaults=defaults)

        self.add_time_options(units='s')

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
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
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


class AscentPhase(PhaseBuilderBase):
    """
    A phase builder for an ascent phase in a 2-degree of freedom mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the ascent phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the ascent phase are included.
    """

    default_name = 'ascent_phase'
    default_ode_class = AscentODE
    default_options_class = AscentPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options

        pitch_constraint_bounds = user_options.get_val('pitch_constraint_bounds', units='deg')
        pitch_constraint_ref = user_options.get_val('pitch_constraint_ref', units='deg')

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

        self.add_control(
            'angle_of_attack',
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
        )

        phase.add_path_constraint('load_factor', upper=1.10, lower=0.0)

        phase.add_path_constraint(
            'fuselage_pitch',
            'theta',
            lower=pitch_constraint_bounds[0],
            upper=pitch_constraint_bounds[1],
            units='deg',
            ref=pitch_constraint_ref,
        )

        phase.add_parameter('t_init_gear', units='s', static_target=True, opt=False, val=38.25)

        phase.add_parameter('t_init_flaps', units='s', static_target=True, opt=False, val=48.21)

        phase.add_timeseries_output(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        phase.add_timeseries_output('normal_force')
        phase.add_timeseries_output(Dynamic.Atmosphere.MACH)
        phase.add_timeseries_output('EAS', units='kn')
        phase.add_timeseries_output(Dynamic.Vehicle.LIFT)
        phase.add_timeseries_output('CL')
        phase.add_timeseries_output('CD')

        return phase


# Adding initial guess metadata
AscentPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(), desc='initial guess for time options'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('flight_path_angle'), desc='initial guess for flight path angle state'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for altitude state'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'), desc='initial guess for true airspeed state'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass state'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance state'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('angle_of_attack'),
    desc='initial guess for angle of attack control',
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('tau_gear'), desc='when the gear is retracted'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('tau_flaps'), desc='when the flaps are retracted'
)

AscentPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
