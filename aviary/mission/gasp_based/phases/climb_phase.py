from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class ClimbPhaseOptions(AviaryOptionsDictionary):
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

        self.declare(
            name='required_available_climb_rate',
            default=None,
            units='ft/min',
            desc='Adds a constraint requiring Dynamic.Mission.ALTITUDE_RATE_MAX to be no '
            'smaller than required_available_climb_rate. This helps to ensure that the '
            'propulsion system is large enough to handle emergency maneuvers at all points '
            'throughout the flight envelope. Default value is None for no constraint.',
        )

        # The options below have not yet been revamped.

        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
        )

        self.declare(
            name='EAS_target',
            default=0.0,
            units='kn',
            desc='Target airspeed for the balance in this phase.',
        )

        self.declare(
            name='mach_cruise',
            default=0.0,
            desc='Defines the mach constraint at the end of the phase. '
            'Only valid when target_mach=True.',
        )

        self.declare(
            'target_mach',
            types=bool,
            default=False,
            desc='Set to true to enforce a mach_constraint at the phase endpoint. '
            'The mach value is set in "mach_cruise".',
        )


class ClimbPhase(PhaseBuilderBase):
    """
    A phase builder for a climb phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the climb phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the climb phase are included.
    """

    default_name = 'climb_phase'
    default_ode_class = ClimbODE
    default_options_class = ClimbPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new climb phase for analysis using these constraints.

        If ode_class is None, ClimbODE is used as the default.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase = self.phase = super().build_phase(aviary_options)

        # Custom configurations for the climb phase
        user_options = self.user_options

        mach_cruise = user_options.get_val('mach_cruise')
        target_mach = user_options.get_val('target_mach')
        altitude_final = user_options.get_val('altitude_final', units='ft')
        required_available_climb_rate = user_options.get_val(
            'required_available_climb_rate', units='ft/min'
        )

        # States
        self.add_state('altitude', Dynamic.Mission.ALTITUDE, Dynamic.Mission.ALTITUDE_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

        if required_available_climb_rate is not None:
            # TODO: this should be altitude rate max
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                loc='final',
                lower=required_available_climb_rate,
                units='ft/min',
                ref=1,
            )

        if target_mach:
            phase.add_boundary_constraint(
                Dynamic.Atmosphere.MACH,
                loc='final',
                equals=mach_cruise,
            )

        # Timeseries Outputs
        phase.add_timeseries_output(
            Dynamic.Atmosphere.MACH,
            output_name=Dynamic.Atmosphere.MACH,
            units='unitless',
        )
        phase.add_timeseries_output('EAS', output_name='EAS', units='kn')
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/s'
        )
        phase.add_timeseries_output('theta', output_name='theta', units='deg')
        phase.add_timeseries_output(
            Dynamic.Vehicle.ANGLE_OF_ATTACK,
            output_name=Dynamic.Vehicle.ANGLE_OF_ATTACK,
            units='deg',
        )
        phase.add_timeseries_output(
            Dynamic.Mission.FLIGHT_PATH_ANGLE,
            output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE,
            units='deg',
        )
        phase.add_timeseries_output('TAS_violation', output_name='TAS_violation', units='kn')
        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY,
            units='kn',
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

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'EAS_target': self.user_options.get_val('EAS_target', units='kn'),
            'mach_cruise': self.user_options.get_val('mach_cruise'),
        }


ClimbPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple',
)

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for horizontal distance traveled'
)

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for vertical distances'
)

ClimbPhase._add_initial_guess_meta_data(InitialGuessState('mass'), desc='initial guess for mass')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
