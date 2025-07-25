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

        defaults = {
            'mass_bounds': (0.0, None),
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        defaults = {
            'distance_bounds': (0.0, None),
        }
        self.add_state_options('distance', units='NM', defaults=defaults)

        defaults = {
            'altitude_bounds': (0.0, None),
            'altitude_constraint_ref': 100.0,
        }
        self.add_state_options('altitude', units='ft', defaults=defaults)

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
        self.add_state('altitude', Dynamic.Mission.ALTITUDE, Dynamic.Mission.ALTITUDE_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

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
