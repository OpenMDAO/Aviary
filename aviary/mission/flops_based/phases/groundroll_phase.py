import dymos as dm

from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.initial_guess_builders import (
    InitialGuessIntegrationVariable,
    InitialGuessPolynomialControl,
    InitialGuessState,
)
from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic

# Solved 2DOF uses this builder.
#
# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.meta_data, with cls.default_meta_data customization point


class GroundrollPhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='num_segments',
            types=int,
            default=5,
            desc='The number of segments in transcription creation in Dymos. '
            'The default value is 5.',
        )

        self.declare(
            name='order',
            types=int,
            default=3,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos. The default value is 3.',
        )

        defaults = {
            'time_initial_bounds': (0.0, 100.0),
            'time_duration_bounds': (0.0, 3600.0),
            'time_initial_ref': 100.0,
            'time_duration_ref': 100.0,
        }
        self.add_time_options(units='kn', defaults=defaults)

        # The options below have not yet been revamped.

        self.declare(
            name='constraints',
            types=dict,
            default={},
            desc="Add in custom constraints i.e. 'flight_path_angle': {'equals': -3., "
            "'loc': 'initial', 'units': 'deg', 'type': 'boundary',}. For more details see "
            '_add_user_defined_constraints().',
        )

        self.declare(
            name='ground_roll',
            types=bool,
            default=False,
            desc='Set to True only for phases where the aircraft is rolling on the ground. '
            'All other phases of flight (climb, cruise, descent) this must be set to False.',
        )


@register
class GroundrollPhase(PhaseBuilderBase):
    """A phase builder for a two degree of freedom (2DOF) phase."""

    __slots__ = ('external_subsystems', 'meta_data')

    _initial_guesses_meta_data_ = {}

    default_name = 'groundroll'
    default_ode_class = GroundrollODE
    default_options_class = GroundrollPhaseOptions
    default_meta_data = _MetaData

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
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        duration_bounds = user_options.get_val('time_duration_bounds', units='kn')
        duration_ref = user_options.get_val('time_duration_ref', units='kn')
        constraints = user_options.get_val('constraints')

        phase.set_time_options(
            fix_initial=True,
            fix_duration=False,
            units='kn',
            name=Dynamic.Mission.VELOCITY,
            duration_bounds=duration_bounds,
            duration_ref=duration_ref,
        )

        phase.set_state_options(
            'time',
            rate_source='dt_dv',
            units='s',
            fix_initial=True,
            fix_final=False,
            ref=1.0,
            defect_ref=1.0,
            solve_segments='forward',
        )

        phase.set_state_options(
            'mass',
            rate_source='dmass_dv',
            fix_initial=True,
            fix_final=False,
            lower=1,
            upper=500.0e3,
            ref=100.0e3,
            defect_ref=100.0e3,
            units='lbm',
        )

        phase.set_state_options(
            Dynamic.Mission.DISTANCE,
            rate_source='over_a',
            fix_initial=True,
            fix_final=False,
            lower=0,
            upper=8000.0,
            ref=1.0e2,
            defect_ref=1.0e2,
            units='ft',
        )

        phase.add_parameter('t_init_gear', units='s', static_target=True, opt=False, val=32.3)
        phase.add_parameter('t_init_flaps', units='s', static_target=True, opt=False, val=44.0)
        phase.add_parameter('wing_area', units='ft**2', static_target=True, opt=False, val=1370)

        self._add_user_defined_constraints(phase, constraints)

        phase.add_timeseries_output(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        phase.add_timeseries_output('normal_force')
        phase.add_timeseries_output(Dynamic.Atmosphere.MACH)
        phase.add_timeseries_output('EAS', units='kn')
        phase.add_timeseries_output(Dynamic.Mission.VELOCITY, units='kn')
        phase.add_timeseries_output(Dynamic.Vehicle.LIFT)
        phase.add_timeseries_output(Dynamic.Vehicle.DRAG)
        phase.add_timeseries_output('time')
        phase.add_timeseries_output('mass')
        phase.add_timeseries_output(Dynamic.Vehicle.ANGLE_OF_ATTACK)

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
            'set_input_defaults': False,
        }


GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(key='velocity'),
    desc='initial guess for initial velocity and final specified as a tuple',
)

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('altitude'), desc='initial guess for vertical distances'
)

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass'
)

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for distance'
)

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('time'), desc='initial guess for time'
)
