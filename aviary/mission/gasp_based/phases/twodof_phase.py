import dymos as dm

from aviary.mission.flight_phase_builder import FlightPhaseBase, register
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import UnsteadySolvedODE
from aviary.mission.initial_guess_builders import (
    InitialGuessIntegrationVariable,
    InitialGuessPolynomialControl,
    InitialGuessState,
)
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.enums import EquationsOfMotion, SpeedType, ThrottleAllocation
from aviary.variable_info.variables import Dynamic

# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point


class TwoDOFPhaseOptions(AviaryOptionsDictionary):
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

        # TODO: These defaults aren't great, but need to keep things the same for now.
        defaults = {
            'mass_ref': 1e4,
            'mass_defect_ref': 1e6,
            'mass_bounds': (0.0, None),
        }
        self.add_state_options('mass', units='kg', defaults=defaults)

        # TODO: These defaults aren't great, but need to keep things the same for now.
        defaults = {
            'distance_ref': 1e6,
            'distance_defect_ref': 1e8,
            'distance_bounds': (0.0, None),
            'mass_bounds': (0.0, None),
        }
        self.add_state_options('distance', units='m', defaults=defaults)

        self.add_control_options('altitude', units='ft')

        # TODO: These defaults aren't great, but need to keep things the same for now.
        defaults = {
            'mach_ref': 0.5,
        }
        self.add_control_options('mach', units='unitless', defaults=defaults)

        defaults = {
            'angle_of_attack_polynomial_order': 1,
            'angle_of_attack_optimize': True,
            'angle_of_attack_ref': 10.0,
            'angle_of_attack_bounds': (0.0, 15.0),
            'angle_of_attack_optimize': True,
            'angle_of_attack_initial': 0.0,
        }
        self.add_control_options('angle_of_attack', units='deg', defaults=defaults)

        self.add_time_options(units='ft')

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
            name='ground_roll',
            types=bool,
            default=False,
            desc='Set to True only for phases where the aircraft is rolling on the ground. '
            'All other phases of flight (climb, cruise, descent) this must be set to False.',
        )

        # The options below have not yet been revamped.

        self.declare(
            name='required_available_climb_rate',
            default=None,
            units='ft/s',
            desc='Adds a constraint requiring Dynamic.Mission.ALTITUDE_RATE_MAX to be no '
            'smaller than required_available_climb_rate. This helps to ensure that the '
            'propulsion system is large enough to handle emergency maneuvers at all points '
            'throughout the flight envelope. Default value is None for no constraint.',
        )

        self.declare(
            name='no_climb',
            types=bool,
            default=False,
            desc='Set to True to prevent the aircraft from climbing during the phase. This option '
            'can be used to prevent unexpected climb during a descent phase.',
        )

        self.declare(
            name='no_descent',
            types=bool,
            default=False,
            desc='Set to True to prevent the aircraft from descending during the phase. This '
            'can be used to prevent unexpected descent during a climb phase.',
        )

        self.declare(
            name='throttle_enforcement',
            default='path_constraint',
            values=['path_constraint', 'boundary_constraint', 'bounded', None],
            desc='Flag to enforce engine throttle constraints on the path or at the segment '
            'boundaries or using solver bounds.',
        )

        self.declare(
            name='throttle_allocation',
            default=ThrottleAllocation.FIXED,
            values=[
                ThrottleAllocation.FIXED,
                ThrottleAllocation.STATIC,
                ThrottleAllocation.DYNAMIC,
            ],
            desc='Specifies how to handle the throttles for multiple engines. FIXED is a '
            'user-specified value. STATIC is specified by the optimizer as one value for the '
            'whole phase. DYNAMIC is specified by the optimizer at each point in the phase.',
        )

        self.declare(
            name='constraints',
            types=dict,
            default={},
            desc="Add in custom constraints i.e. 'flight_path_angle': {'equals': -3., "
            "'loc': 'initial', 'units': 'deg', 'type': 'boundary',}. For more details see "
            '_add_user_defined_constraints().',
        )

        self.declare(
            name='rotation',
            types=bool,
            default=False,
            desc='Set to true if this is a rotation phase.',
        )

        self.declare(
            name='clean',
            types=bool,
            default=False,
            desc='Set to true to use clean aero with no ground effects.',
        )


@register
class TwoDOFPhase(FlightPhaseBase):
    """A phase builder for a two degree of freedom (2DOF) phase."""

    default_options_class = TwoDOFPhaseOptions

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new 2dof phase for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (<empty>)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        self.ode_class = UnsteadySolvedODE
        phase: dm.Phase = super().build_phase(
            aviary_options, phase_type=EquationsOfMotion.SOLVED_2DOF
        )

        user_options = self.user_options

        time_units = 'ft'
        initial = wrapped_convert_units(user_options['time_initial'], time_units)
        duration = wrapped_convert_units(user_options['time_duration'], time_units)
        initial_bounds = wrapped_convert_units(user_options['time_initial_bounds'], time_units)
        duration_bounds = wrapped_convert_units(user_options['time_duration_bounds'], time_units)
        initial_ref = wrapped_convert_units(user_options['time_initial_ref'], time_units)
        duration_ref = wrapped_convert_units(user_options['time_duration_ref'], time_units)
        rotation = user_options.get_val('rotation')

        fix_duration = duration is not None
        fix_initial = initial is not None

        extra_options = {}
        if not fix_initial:
            extra_options = {
                'initial_bounds': initial_bounds,
                'initial_ref': initial_ref,
            }

        if not fix_duration:
            extra_options = {
                'duration_bounds': duration_bounds,
                'duration_ref': duration_ref,
            }

        phase.set_time_options(
            fix_initial=fix_initial,
            fix_duration=fix_duration,
            units=time_units,
            name=Dynamic.Mission.DISTANCE,
            **extra_options,
        )

        phase.set_state_options(
            'time',
            rate_source='dt_dr',
            fix_initial=fix_initial,
            fix_final=False,
            ref=100.0,
            defect_ref=100.0,
        )

        if rotation:
            self.add_control(
                'angle_of_attack',
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            )

        phase.add_timeseries_output('EAS', units='kn')
        phase.add_timeseries_output(Dynamic.Mission.VELOCITY, units='kn')
        phase.add_timeseries_output(Dynamic.Vehicle.LIFT)
        phase.add_timeseries_output('thrust_req', units='lbf')

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        user_options = self.user_options

        num_segments = user_options['num_segments']
        order = user_options['order']

        seg_ends, _ = dm.utils.lgl.lgl(num_segments + 1)

        transcription = dm.Radau(
            num_segments=num_segments, order=order, compressed=True, segment_ends=seg_ends
        )

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'external_subsystems': self.external_subsystems,
            'meta_data': self.meta_data,
            'subsystem_options': self.subsystem_options,
            'input_speed_type': SpeedType.MACH,
            'clean': self.user_options.get_val('clean'),
            'ground_roll': self.user_options.get_val('ground_roll'),
            'throttle_enforcement': self.user_options.get_val('throttle_enforcement'),
        }


TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(key='distance'),
    desc='initial guess for initial distance and duration specified as a tuple',
)

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('angle_of_attack'),
    desc='initial guess for angle of attack',
)

TwoDOFPhase._add_initial_guess_meta_data(InitialGuessState('time'), desc='initial guess for time')
