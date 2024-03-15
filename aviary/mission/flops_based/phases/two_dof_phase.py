import dymos as dm

from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl, InitialGuessPolynomialControl

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.mission.flops_based.phases.phase_utils import add_subsystem_variables_to_phase, get_initial
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import UnsteadySolvedODE
from aviary.variable_info.enums import SpeedType

# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point


@register
class TwoDOFPhase(PhaseBuilderBase):
    '''
    A phase builder for a two degree of freedom (2DOF) phase.
    '''
    __slots__ = ('external_subsystems', 'meta_data')

    # region : derived type customization points
    _meta_data_ = {}

    _initial_guesses_meta_data_ = {}

    default_name = 'cruise'

    default_ode_class = UnsteadySolvedODE

    default_meta_data = _MetaData
    # endregion : derived type customization points

    def __init__(
        self, name=None, subsystem_options=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, core_subsystems=None,
        external_subsystems=None, meta_data=None
    ):
        super().__init__(
            name=name, core_subsystems=core_subsystems, subsystem_options=subsystem_options, user_options=user_options, initial_guesses=initial_guesses, ode_class=ode_class, transcription=transcription)

        # TODO: support external_subsystems and meta_data in the base class
        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

    def build_phase(self, aviary_options: AviaryValues = None):
        '''
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
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        control_order = user_options.get_val('control_order')

        fix_initial = user_options.get_val('fix_initial')
        duration_bounds = user_options.get_val('duration_bounds', units='ft')
        duration_ref = user_options.get_val('duration_ref', units='ft')
        ground_roll = user_options.get_val('ground_roll')
        throttle_enforcement = user_options.get_val('throttle_enforcement')
        rotation = user_options.get_val('rotation')
        constraints = user_options.get_val('constraints')
        optimize_mach = user_options.get_val('optimize_mach')
        optimize_altitude = user_options.get_val('optimize_altitude')
        mach_bounds = user_options.get_item('mach_bounds')
        altitude_bounds = user_options.get_item('altitude_bounds')
        use_polynomial_control = user_options.get_val('use_polynomial_control')

        initial_kwargs = {}
        if not fix_initial:
            initial_kwargs = {
                'initial_bounds': user_options.get_val('initial_bounds', units='ft'),
                'initial_ref': user_options.get_val('initial_ref', units='ft'),
            }

        phase.set_time_options(fix_initial=fix_initial, fix_duration=False,
                               units='ft', name=Dynamic.Mission.DISTANCE,
                               duration_bounds=duration_bounds, duration_ref=duration_ref,
                               **initial_kwargs)

        phase.set_state_options("time", rate_source="dt_dr",
                                fix_initial=fix_initial, fix_final=False, ref=100., defect_ref=100.)

        phase.set_state_options("mass", rate_source="dmass_dr",
                                fix_initial=fix_initial, fix_final=False, ref=170.e3, defect_ref=170.e5,
                                val=170.e3, units='lbm', lower=10.e3)

        if use_polynomial_control:
            phase.add_polynomial_control(Dynamic.Mission.MACH,
                                         order=control_order,
                                         val=0.4, units=mach_bounds[1],
                                         opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                                         fix_initial=fix_initial,
                                         rate_targets=['dmach_dr'],
                                         )
        else:
            phase.add_control(Dynamic.Mission.MACH,
                              fix_initial=fix_initial,
                              val=0.4, units=mach_bounds[1],
                              opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                              rate_targets=['dmach_dr'],
                              )

        if rotation:
            phase.add_polynomial_control("alpha",
                                         order=control_order,
                                         fix_initial=True,
                                         lower=0, upper=15,
                                         units='deg', ref=10.,
                                         val=0.,
                                         opt=True)

        if ground_roll:
            phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                         order=1, val=0, opt=False,
                                         fix_initial=fix_initial,
                                         rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'])
        else:
            if use_polynomial_control:
                phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                             order=control_order,
                                             fix_initial=fix_initial,
                                             units=altitude_bounds[1],
                                             rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                             opt=optimize_altitude,
                                             lower=altitude_bounds[0][0],
                                             upper=altitude_bounds[0][1],
                                             ref=(altitude_bounds[0][0] +
                                                  altitude_bounds[0][1]) / 2,
                                             )
            else:
                phase.add_control(Dynamic.Mission.ALTITUDE,
                                  fix_initial=fix_initial,
                                  units=altitude_bounds[1],
                                  rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                  opt=optimize_altitude,
                                  lower=altitude_bounds[0][0],
                                  upper=altitude_bounds[0][1],
                                  ref=(altitude_bounds[0][0] +
                                       altitude_bounds[0][1]) / 2,
                                  )

        if not Dynamic.Mission.THROTTLE in constraints:
            if throttle_enforcement == 'boundary_constraint':
                phase.add_boundary_constraint(
                    Dynamic.Mission.THROTTLE, loc='initial', lower=0.0, upper=1.0, units='unitless',
                )
                phase.add_boundary_constraint(
                    Dynamic.Mission.THROTTLE, loc='final', lower=0.0, upper=1.0, units='unitless',
                )
            elif throttle_enforcement == 'path_constraint':
                phase.add_path_constraint(
                    Dynamic.Mission.THROTTLE, lower=0.0, upper=1.0, units='unitless',
                )

        self._add_user_defined_constraints(phase, constraints)

        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("thrust_req", units="lbf")
        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output("TAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output(Dynamic.Mission.DRAG)
        phase.add_timeseries_output("time")
        phase.add_timeseries_output("mass")
        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)
        phase.add_timeseries_output("alpha")
        phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE)
        phase.add_timeseries_output(Dynamic.Mission.THROTTLE)
        phase.add_timeseries_output(
            "fuselage_pitch", output_name="theta", units="deg")

        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        user_options = self.user_options

        num_segments, _ = user_options.get_item('num_segments')
        order, _ = user_options.get_item('order')

        seg_ends, _ = dm.utils.lgl.lgl(num_segments + 1)

        transcription = dm.Radau(
            num_segments=num_segments, order=order, compressed=True,
            segment_ends=seg_ends)

        return transcription

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
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


TwoDOFPhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')
TwoDOFPhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')
TwoDOFPhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')

TwoDOFPhase._add_meta_data(
    'num_segments', val=5, desc='transcription: number of segments')

TwoDOFPhase._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

TwoDOFPhase._add_meta_data('fix_initial', val=False)

TwoDOFPhase._add_meta_data('fix_duration', val=False)

TwoDOFPhase._add_meta_data('optimize_mach', val=False)

TwoDOFPhase._add_meta_data('optimize_altitude', val=False)

TwoDOFPhase._add_meta_data('initial_bounds', val=(0., 100.),
                           units='s', desc='initial bounds')
TwoDOFPhase._add_meta_data('duration_bounds', val=(
    0., 3600.), units='s', desc='duration bounds')
TwoDOFPhase._add_meta_data('initial_ref', val=100., units='s', desc='initial reference')
TwoDOFPhase._add_meta_data('duration_ref', val=1000.,
                           units='s', desc='duration reference')
TwoDOFPhase._add_meta_data('control_order', val=1, desc='control order')
TwoDOFPhase._add_meta_data('ground_roll', val=False)
TwoDOFPhase._add_meta_data('rotation', val=False)
TwoDOFPhase._add_meta_data('clean', val=False)
TwoDOFPhase._add_meta_data('throttle_enforcement', val=None)
TwoDOFPhase._add_meta_data('constraints', val={})
TwoDOFPhase._add_meta_data('mach_bounds', val=(0., 2.), units='unitless')
TwoDOFPhase._add_meta_data('altitude_bounds', val=(0., 60.e3), units='ft')
TwoDOFPhase._add_meta_data('use_polynomial_control', val=True)

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(key='distance'),
    desc='initial guess for initial distance and duration specified as a tuple')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('altitude'),
    desc='initial guess for vertical distances')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('mach'),
    desc='initial guess for speed')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('alpha'),
    desc='initial guess for alpha')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('time'),
    desc='initial guess for time')
