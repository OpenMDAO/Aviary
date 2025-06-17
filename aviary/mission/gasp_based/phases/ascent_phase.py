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
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
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

        self.declare(
            'time_duration',
            default=None,
            units='s',
            desc='The amount of time taken by this phase added as a constraint.',
        )

        self.declare(
            name='fix_initial',
            types=bool,
            default=False,
            desc='Fixes the initial states (mass, distance) and does not allow them to '
            'change during the optimization.',
        )

        self.declare(
            name='angle_lower',
            types=tuple,
            default=-15 * np.pi / 180,
            units='rad',
            desc='Lower bound for angle.',
        )

        self.declare(
            name='angle_upper', default=25 * np.pi / 180, units='rad', desc='Upper bound for angle.'
        )

        self.declare(
            name='angle_ref', default=np.deg2rad(1), units='rad', desc='Scale factor ref for angle.'
        )

        self.declare(
            name='angle_ref0', default=0.0, units='rad', desc='Scale factor ref0 for angle.'
        )

        self.declare(
            name='angle_defect_ref',
            default=0.01,
            units='rad',
            desc='Scale factor ref for angle defect.',
        )

        self.declare(
            name='alt_lower', types=tuple, default=0.0, units='ft', desc='Lower bound for altitude.'
        )

        self.declare(name='alt_upper', default=700.0, units='ft', desc='Upper bound for altitude.')

        self.declare(
            name='alt_ref', default=100.0, units='ft', desc='Scale factor ref for altitude.'
        )

        self.declare(
            name='alt_ref0', default=0.0, units='ft', desc='Scale factor ref0 for altitude.'
        )

        self.declare(
            name='alt_defect_ref',
            default=100.0,
            units='ft',
            desc='Scale factor ref for altitude defect.',
        )

        self.declare(
            name='altitude_final',
            default=500.0,
            units='ft',
            desc='Altitude for final point in the phase.',
        )

        self.declare(
            name='alt_constraint_ref',
            default=100.0,
            units='ft',
            desc='Scaling ref for the final altitude constraint.',
        )

        self.declare(
            name='alt_constraint_ref0',
            default=0.0,
            units='ft',
            desc='Scaling ref0 for the final altitude constraint.',
        )

        self.declare(
            name='velocity_lower', default=0.0, units='kn', desc='Lower bound for velocity.'
        )

        self.declare(
            name='velocity_upper', default=1000.0, units='kn', desc='Upper bound for velocity.'
        )

        self.declare(
            name='velocity_ref', default=1.0e2, units='kn', desc='Scale factor ref for velocity.'
        )

        self.declare(
            name='velocity_ref0', default=0.0, units='kn', desc='Scale factor ref0 for velocity.'
        )

        self.declare(
            name='velocity_defect_ref',
            default=None,
            units='kn',
            desc='Scale factor ref for velocity defect.',
        )

        self.declare(
            name='mass_lower', types=tuple, default=0.0, units='lbm', desc='Lower bound for mass.'
        )

        self.declare(
            name='mass_upper', default=190_000.0, units='lbm', desc='Upper bound for mass.'
        )

        self.declare(
            name='mass_ref', default=100_000.0, units='lbm', desc='Scale factor ref for mass.'
        )

        self.declare(name='mass_ref0', default=0.0, units='lbm', desc='Scale factor ref0 for mass.')

        self.declare(
            name='mass_defect_ref',
            default=1.0e2,
            units='lbm',
            desc='Scale factor ref for mass defect.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='distance_lower', default=0.0, units='ft', desc='Lower bound for distance.'
        )

        self.declare(
            name='distance_upper', default=10.0e3, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='distance_ref', default=3000.0, units='ft', desc='Scale factor ref for distance.'
        )

        self.declare(
            name='distance_ref0', default=0.0, units='ft', desc='Scale factor ref0 for distance.'
        )

        self.declare(
            name='distance_defect_ref',
            default=3000.0,
            units='ft',
            desc='Scale factor ref for distance defect.',
        )

        self.declare(
            name='pitch_constraint_lower',
            default=0.0,
            units='deg',
            desc='Pitch lower bound constraint.',
        )

        self.declare(
            name='pitch_constraint_upper',
            default=15.0,
            units='deg',
            desc='Pitch upper bound constraint.',
        )

        self.declare(
            name='pitch_constraint_ref',
            default=1.0,
            units='deg',
            desc='Scale factor ref for the pitch constraint.',
        )

        self.declare(
            name='alpha_constraint_lower',
            default=np.deg2rad(-30),
            units='rad',
            desc='Angle of attack lower bound constraint.',
        )

        self.declare(
            name='alpha_constraint_upper',
            default=np.deg2rad(30),
            units='rad',
            desc='Angle of attack upper bound constraint.',
        )

        self.declare(
            name='alpha_constraint_ref',
            default=np.deg2rad(5),
            units='rad',
            desc='Scale factor ref for the Angle of attack constraint.',
        )

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

        pitch_constraint_lower = user_options.get_val('pitch_constraint_lower', units='deg')
        pitch_constraint_upper = user_options.get_val('pitch_constraint_upper', units='deg')
        pitch_constraint_ref = user_options.get_val('pitch_constraint_ref', units='deg')
        alpha_constraint_lower = user_options.get_val('alpha_constraint_lower', units='rad')
        alpha_constraint_upper = user_options.get_val('alpha_constraint_upper', units='rad')
        alpha_constraint_ref = user_options.get_val('alpha_constraint_ref', units='rad')

        self.add_flight_path_angle_state(user_options)
        self.add_altitude_state(user_options)
        self.add_velocity_state(user_options)
        self.add_mass_state(user_options)
        self.add_distance_state(user_options, units='ft')

        self.add_altitude_constraint(user_options)

        phase.add_path_constraint('load_factor', upper=1.10, lower=0.0)

        phase.add_path_constraint(
            'fuselage_pitch',
            'theta',
            lower=pitch_constraint_lower,
            upper=pitch_constraint_upper,
            units='deg',
            ref=pitch_constraint_ref,
        )

        phase.add_control(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            val=0,
            lower=alpha_constraint_lower,
            upper=alpha_constraint_upper,
            units='rad',
            ref=alpha_constraint_ref,
            opt=True,
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
