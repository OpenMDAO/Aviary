from aviary.mission.gasp_based.ode.accel_ode import AccelODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class AccelPhaseOptions(AviaryOptionsDictionary):
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
            'mass_bounds': (0.0, None),
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        defaults = {
            'distance_bounds': (0.0, None),
        }
        self.add_state_options('distance', units='NM', defaults=defaults)

        defaults = {
            'velocity_bounds': (0.0, None),
        }
        self.add_state_options('velocity', units='kn', defaults=defaults)

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

        self.declare(
            name='EAS_constraint_eq',
            default=250.0,
            units='kn',
            desc='Airspeed constraint applied at the end of the phase.',
        )

        # The options below have not yet been revamped.

        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
        )

        self.declare(
            name='alt', default=500.0, units='ft', desc='Constant altitude for this phase.'
        )


@register
class AccelPhase(PhaseBuilderBase):
    """
    A phase builder for an acceleration phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the acceleration phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the acceleration phase are included.
    """

    default_name = 'accel_phase'
    default_ode_class = AccelODE
    default_options_class = AccelPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new acceleration phase for analysis using these constraints.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase = self.phase = super().build_phase(aviary_options)
        user_options = self.user_options

        # Extracting and setting options
        EAS_constraint_eq = user_options.get_val('EAS_constraint_eq', 'kn')
        alt = user_options.get_val('alt', 'ft')

        # States
        self.add_state('velocity', Dynamic.Mission.VELOCITY, Dynamic.Mission.VELOCITY_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

        # Boundary Constraints
        phase.add_boundary_constraint(
            'EAS', loc='final', equals=EAS_constraint_eq, units='kn', ref=EAS_constraint_eq
        )

        phase.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, units='ft', val=alt)

        # Timeseries Outputs
        phase.add_timeseries_output('EAS', output_name='EAS', units='kn')
        phase.add_timeseries_output(
            Dynamic.Atmosphere.MACH,
            output_name=Dynamic.Atmosphere.MACH,
            units='unitless',
        )
        phase.add_timeseries_output(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            output_name=Dynamic.Vehicle.ANGLE_OF_ATTACK,
            units='deg',
        )
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        # TODO: These should be promoted in the 2dof mission outputs.
        phase.add_timeseries_output('core_aerodynamics.CL', output_name='CL', units='unitless')
        phase.add_timeseries_output('core_aerodynamics.CD', output_name='CD', units='unitless')

        return phase


AccelPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple',
)

AccelPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'), desc='initial guess for true airspeed'
)

AccelPhase._add_initial_guess_meta_data(InitialGuessState('mass'), desc='initial guess for mass')

AccelPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for horizontal distance traveled'
)

AccelPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
