from aviary.mission.two_dof.ode.flight_ode import FlightODE
from aviary.mission.initial_guess_builders import (
    InitialGuessControl,
    InitialGuessIntegrationVariable,
    InitialGuessState,
)
from aviary.mission.phase_builder import PhaseBuilder
from aviary.mission.phase_utils import add_subsystem_variables_to_phase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class FlightPhaseOptions(AviaryOptionsDictionary):
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
            desc='Value for maximum speed constraint for this phase.',
        )

        self.declare(
            name='mach_target',
            default=0.0,
            desc='Defines the maximum mach constraint for this phase.',
        )

        self.declare(
            name='input_speed_type',
            default=SpeedType.MACH,
            values=[SpeedType.MACH, SpeedType.EAS, SpeedType.TAS],
            desc='Determines which speed variable is independent. The other two will be .'
            'computed from it.',
        )

        self.declare(
            name='constraints',
            types=dict,
            default={},
            desc="Add in custom constraints i.e. 'flight_path_angle': {'equals': -3., "
            "'loc': 'initial', 'units': 'deg', 'type': 'boundary',}. For more details see "
            '_add_user_defined_constraints().',
        )


class FlightPhase(PhaseBuilder):
    """
    A phase builder for a 2DOF flight phase.

    This class extends the PhaseBuilder class, providing specific implementations for
    the descent phase of a 2-degree of freedom flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilder.

    Methods
    -------
    Inherits all methods from PhaseBuilder.
    Additional method overrides and new methods specific to this phase are included.
    """

    default_name = 'flight_phase'
    default_ode_class = FlightODE
    default_options_class = FlightPhaseOptions

    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        input_speed_type = user_options.get_val('input_speed_type')
        EAS_target = user_options.get_val('EAS_target', units='kn')
        required_available_climb_rate = user_options.get_val(
            'required_available_climb_rate', units='ft/min'
        )

        # Add states
        self.add_state('altitude', Dynamic.Mission.ALTITUDE, Dynamic.Mission.ALTITUDE_RATE)
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

        add_subsystem_variables_to_phase(phase, self.name, self.subsystems)

        # Add constraints
        if required_available_climb_rate is not None:
            # TODO: this should be altitude rate max
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                loc='final',
                lower=required_available_climb_rate,
                units='ft/min',
                ref=1,
            )

        constraints = user_options['constraints']
        self._add_user_defined_constraints(phase, constraints)

        # Add parameter if necessary
        if input_speed_type == SpeedType.EAS:
            phase.add_parameter('EAS', opt=False, units='kn', val=EAS_target)

        # Add timeseries outputs
        phase.add_timeseries_output(
            Dynamic.Atmosphere.MACH,
            output_name=Dynamic.Atmosphere.MACH,
            units='unitless',
        )
        phase.add_timeseries_output('EAS', output_name='EAS', units='kn')
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/s'
        )
        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY,
            units='kn',
        )
        phase.add_timeseries_output('TAS_violation', output_name='TAS_violation', units='kn')
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
        phase.add_timeseries_output('aerodynamics.CL', output_name='CL', units='unitless')
        phase.add_timeseries_output('aerodynamics.CD', output_name='CD', units='unitless')

        return phase

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'input_speed_type': self.user_options.get_val('input_speed_type'),
            'mach_target': self.user_options.get_val('mach_target'),
            'EAS_target': self.user_options.get_val('EAS_target', 'kn'),
        }


# Adding initial guess metadata
FlightPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(), desc='initial guess for time options'
)
FlightPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for altitude state'
)
FlightPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass state'
)
FlightPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance state'
)
FlightPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'), desc='initial guess for throttle'
)
