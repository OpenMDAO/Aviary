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
            'distance_bounds': (0.0, 4000.0),
            'distance_ref': 3000.0,
        }
        self.add_state_options('distance', units='ft', defaults=defaults)

        defaults = {
            'velocity_bounds': (0.0, 1000.0),
            'velocity_ref': 100.0,
        }
        self.add_state_options('velocity', units='kn', defaults=defaults)

        # The options below have not yet been revamped.

        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
        )

        self.declare(
            name='t_init_gear', default=100.0, units='s', desc='Time where landing gear is lifted.'
        )

        self.declare(
            name='t_init_flaps', default=100.0, units='s', desc='Time where flaps are retracted.'
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

        # Add states
        self.add_state('velocity', Dynamic.Mission.VELOCITY, Dynamic.Mission.VELOCITY_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

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
