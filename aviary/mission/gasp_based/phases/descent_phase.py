from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class DescentPhaseOptions(AviaryOptionsDictionary):
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
            name='input_initial',
            types=bool,
            default=False,
            desc='Links all states to a calculation external to this phase.',
        )

        self.declare(
            name='EAS_limit',
            default=0.0,
            units='kn',
            desc='Value for maximum speed constraint in this phase.',
        )

        self.declare(
            name='mach_cruise', default=0.0, desc='Defines the mach constraint in this phase.'
        )

        self.declare(
            name='input_speed_type',
            default=SpeedType.MACH,
            values=[SpeedType.MACH, SpeedType.EAS, SpeedType.TAS],
            desc='Determines which speed variable is independent. The other two will be .'
            'computed from it.',
        )

        self.declare(
            name='altitude_final',
            default=0.0,
            units='ft',
            desc='Altitude for final point in the phase.',
        )

        self.declare(
            name='time_duration_bounds',
            default=(0, 0),
            units='s',
            desc='Lower and upper bounds on the phase duration, in the form of a nested tuple: '
            'i.e. ((20, 36), "min") This constrains the duration to be between 20 and 36 min.',
        )

        self.declare(
            name='time_duration_ref', default=1.0, units='s', desc='Scale factor ref for duration.'
        )

        self.declare(
            name='alt_lower', types=tuple, default=0.0, units='ft', desc='Lower bound for altitude.'
        )

        self.declare(name='alt_upper', default=0.0, units='ft', desc='Upper bound for altitude.')

        self.declare(name='alt_ref', default=1.0, units='ft', desc='Scale factor ref for altitude.')

        self.declare(
            name='alt_ref0', default=0.0, units='ft', desc='Scale factor ref0 for altitude.'
        )

        self.declare(
            name='alt_defect_ref',
            default=None,
            units='ft',
            desc='Scale factor ref for altitude defect.',
        )

        self.declare(
            name='alt_constraint_ref',
            default=None,
            units='ft',
            desc='Scale factor ref for altitude defect.',
        )

        self.declare(
            name='alt_constraint_ref',
            default=100.0,
            units='ft',
            desc='Scaling ref for the final altitude constraint.',
        )

        self.declare(
            name='mass_lower', types=tuple, default=0.0, units='lbm', desc='Lower bound for mass.'
        )

        self.declare(name='mass_upper', default=0.0, units='lbm', desc='Upper bound for mass.')

        self.declare(name='mass_ref', default=1.0, units='lbm', desc='Scale factor ref for mass.')

        self.declare(name='mass_ref0', default=0.0, units='lbm', desc='Scale factor ref0 for mass.')

        self.declare(
            name='mass_defect_ref',
            default=None,
            units='lbm',
            desc='Scale factor ref for mass defect.',
        )

        self.declare(
            name='distance_lower', default=0.0, units='NM', desc='Lower bound for distance.'
        )

        self.declare(
            name='distance_upper', default=0.0, units='NM', desc='Upper bound for distance.'
        )

        self.declare(
            name='distance_ref', default=1.0, units='NM', desc='Scale factor ref for distance.'
        )

        self.declare(
            name='distance_ref0', default=0.0, units='NM', desc='Scale factor ref0 for distance.'
        )

        self.declare(
            name='distance_defect_ref',
            default=None,
            units='NM',
            desc='Scale factor ref for distance defect.',
        )

        self.declare(
            name='num_segments',
            types=int,
            default=None,
            desc='The number of segments in transcription creation in Dymos. ',
        )

        self.declare(
            name='order',
            types=int,
            default=None,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos.',
        )


class DescentPhase(PhaseBuilderBase):
    """
    A phase builder for an descent phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the descent phase of a 2-degree of freedom flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the descent phase are included.
    """

    default_name = 'descent_phase'
    default_ode_class = DescentODE
    default_options_class = DescentPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        input_speed_type = user_options.get_val('input_speed_type')
        EAS_limit = user_options.get_val('EAS_limit', units='kn')

        # Add states
        self.add_altitude_state(user_options)

        self.add_mass_state(user_options)

        self.add_distance_state(user_options)

        # Add boundary constraint
        self.add_altitude_constraint(user_options)

        # Add parameter if necessary
        if input_speed_type == SpeedType.EAS:
            phase.add_parameter('EAS', opt=False, units='kn', val=EAS_limit)

        # Add timeseries outputs
        phase.add_timeseries_output(
            Dynamic.Atmosphere.MACH,
            output_name=Dynamic.Atmosphere.MACH,
            units='unitless',
        )
        phase.add_timeseries_output('EAS', output_name='EAS', units='kn')
        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY,
            units='kn',
        )
        phase.add_timeseries_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE,
            units='deg',
        )
        phase.add_timeseries_output(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            output_name=Dynamic.Vehicle.ANGLE_OF_ATTACK,
            units='deg',
        )
        phase.add_timeseries_output('theta', output_name='theta', units='deg')
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )
        # TODO: These should be promoted in the 2dof mission outputs.
        phase.add_timeseries_output('core_aerodynamics.CL', output_name='CL', units='unitless')
        phase.add_timeseries_output('core_aerodynamics.CD', output_name='CD', units='unitless')

        return phase

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'input_speed_type': self.user_options.get_val('input_speed_type'),
            'mach_cruise': self.user_options.get_val('mach_cruise'),
            'EAS_limit': self.user_options.get_val('EAS_limit', 'kn'),
        }


# Adding initial guess metadata
DescentPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(), desc='initial guess for time options'
)
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for altitude state'
)
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass state'
)
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance state'
)
DescentPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
