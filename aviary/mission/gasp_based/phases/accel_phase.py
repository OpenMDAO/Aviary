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
            name='EAS_constraint_eq',
            default=250.0,
            units='kn',
            desc='Airspeed constraint applied at the end of the phase.',
        )

        self.declare(
            name='time_duration_bounds',
            default=(None, None),
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
            name='velocity_upper', default=0.0, units='kn', desc='Upper bound for velocity.'
        )

        self.declare(
            name='velocity_ref', default=1.0, units='kn', desc='Scale factor ref for velocity.'
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

        self.declare(name='mass_upper', default=0.0, units='lbm', desc='Upper bound for mass.')

        self.declare(name='mass_ref', default=1.0, units='lbm', desc='Scale factor ref for mass.')

        self.declare(name='mass_ref0', default=0.0, units='lbm', desc='Scale factor ref0 for mass.')

        self.declare(
            name='mass_defect_ref',
            default=0.0,
            units='lbm',
            desc='Scale factor ref for mass defect.',
        )

        self.declare(
            name='distance_lower', default=0.0, units='NM', desc='Lower bound for distance.'
        )

        self.declare(
            name='distance_upper', default=0.0, units='ft', desc='Upper bound for distance.'
        )

        self.declare(
            name='distance_ref', default=1.0, units='ft', desc='Scale factor ref for distance.'
        )

        self.declare(
            name='distance_ref0', default=0.0, units='ft', desc='Scale factor ref0 for distance.'
        )

        self.declare(
            name='distance_defect_ref',
            default=None,
            units='ft',
            desc='Scale factor ref for distance defect.',
        )

        self.declare(
            name='alt', default=500.0, units='ft', desc='Constant altitude for this phase.'
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
        self.add_velocity_state(user_options)

        self.add_mass_state(user_options)

        self.add_distance_state(user_options)

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
