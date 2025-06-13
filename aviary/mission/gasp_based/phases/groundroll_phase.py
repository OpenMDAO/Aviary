from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class GroundrollPhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
        )

        self.declare(
            name='fix_initial',
            types=bool,
            default=True,
            desc='Fixes the initial state (distance only) and does not allow it to '
            'change during the optimization.',
        )

        self.declare(
            name='fix_initial_mass',
            types=bool,
            default=False,
            desc='Fixes the initial state for mass and does not allow it to '
            'change during the optimization.',
        )

        self.declare(
            name='connect_initial_mass',
            types=bool,
            default=True,
            desc='When true, initial mass is connected to an outside value.',
        )

        self.declare(
            name='time_duration_bounds',
            default=(1.0, 100.0),
            units='s',
            desc='Lower and upper bounds on the phase duration, in the form of a nested tuple: '
            'i.e. ((20, 36), "min") This constrains the duration to be between 20 and 36 min.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='velocity_lower', default=0.0, units='kn', desc='Lower bound for velocity.'
        )

        self.declare(
            name='velocity_upper', default=1000.0, units='kn', desc='Upper bound for velocity.'
        )

        self.declare(
            name='velocity_ref', default=100.0, units='kn', desc='Scale factor ref for velocity.'
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

        self.declare(name='mass_lower', default=0.0, units='lbm', desc='Lower bound for mass.')

        self.declare(
            name='mass_upper', default=200_000.0, units='lbm', desc='Upper bound for mass.'
        )

        self.declare(
            name='mass_ref', default=100_000.0, units='lbm', desc='Scale factor ref for mass.'
        )

        self.declare(name='mass_ref0', default=0.0, units='lbm', desc='Scale factor ref0 for mass.')

        self.declare(
            name='mass_defect_ref',
            default=100.0,
            units='lbm',
            desc='Scale factor ref for mass defect.',
        )

        self.declare(
            name='distance_lower', default=0.0, units='ft', desc='Lower bound for distance.'
        )

        self.declare(
            name='distance_upper', default=4000.0, units='ft', desc='Upper bound for distance.'
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
            name='t_init_gear', default=100.0, units='s', desc='Time where landing gear is lifted.'
        )

        self.declare(
            name='t_init_flaps', default=100.0, units='s', desc='Time where flaps are retracted.'
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
            default=3,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos. The default value is 3.',
        )


class GroundrollPhase(PhaseBuilderBase):
    """
    A phase builder for a groundroll phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the groundroll phase of a 2-degree of freedom flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the groundroll phase are included.
    """

    default_name = 'groundroll_phase'
    default_ode_class = GroundrollODE
    default_options_class = GroundrollPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        fix_initial = user_options['fix_initial']
        fix_initial_mass = user_options['fix_initial_mass']
        connect_initial_mass = user_options['connect_initial_mass']
        mass_lower = user_options['mass_lower'][0]
        mass_upper = user_options['mass_upper'][0]
        mass_ref = user_options['mass_ref'][0]
        mass_ref0 = user_options['mass_ref0'][0]
        mass_defect_ref = user_options['mass_defect_ref'][0]
        distance_lower = user_options['distance_lower'][0]
        distance_upper = user_options['distance_upper'][0]
        distance_ref = user_options['distance_ref'][0]
        distance_ref0 = user_options['distance_ref0'][0]
        distance_defect_ref = user_options['distance_defect_ref'][0]

        # Add states
        self.add_velocity_state(user_options)

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=fix_initial_mass,
            input_initial=connect_initial_mass,
            fix_final=False,
            lower=mass_lower,
            upper=mass_upper,
            units='lbm',
            rate_source=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            ref=mass_ref,
            defect_ref=mass_defect_ref,
            ref0=mass_ref0,
        )

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=fix_initial,
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units='ft',
            rate_source='distance_rate',
            ref=distance_ref,
            defect_ref=distance_defect_ref,
            ref0=distance_ref0,
        )

        phase.add_parameter('t_init_gear', units='s', static_target=True, opt=False, val=100)
        phase.add_parameter('t_init_flaps', units='s', static_target=True, opt=False, val=100)

        # boundary/path constraints + controls
        # the final TAS is constrained externally to define the transition to the rotation
        # phase

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


# Adding initial guess metadata
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(), desc='initial guess for time options'
)
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'), desc='initial guess for true airspeed state'
)
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass state'
)
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance state'
)
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
