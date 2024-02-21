import dymos as dm

from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessTime, InitialGuessControl, InitialGuessPolynomialControl

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
        Return a new energy phase for analysis using these constraints.

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

        throttle_setting = user_options.get_val('throttle_setting')
        control_order = user_options.get_val('control_order')
        opt = user_options.get_val('opt')

        fix_initial = user_options.get_val('fix_initial')
        initial_bounds = user_options.get_val('initial_bounds', units='ft')
        initial_ref = user_options.get_val('initial_ref', units='ft')
        duration_bounds = user_options.get_val('duration_bounds', units='ft')
        duration_ref = user_options.get_val('duration_ref', units='ft')
        ground_roll = user_options.get_val('ground_roll')
        balance_throttle = user_options.get_val('balance_throttle')
        rotation = user_options.get_val('rotation')
        constraints = user_options.get_val('constraints')
        optimize_mach = user_options.get_val('optimize_mach')
        optimize_altitude = user_options.get_val('optimize_altitude')
        mach_bounds = user_options.get_item('mach_bounds')
        altitude_bounds = user_options.get_item('altitude_bounds')

        if not balance_throttle:
            phase.add_parameter(
                Dynamic.Mission.THROTTLE,
                opt=False,
                units="unitless",
                val=throttle_setting,
                static_target=False)

        if fix_initial:
            phase.set_time_options(fix_initial=fix_initial, fix_duration=False,
                                   units='ft', name=Dynamic.Mission.DISTANCE,
                                   duration_bounds=duration_bounds, duration_ref=duration_ref)
        else:
            phase.set_time_options(fix_initial=fix_initial, fix_duration=False,
                                   units='ft', name=Dynamic.Mission.DISTANCE,
                                   initial_bounds=initial_bounds, initial_ref=initial_ref,
                                   duration_bounds=duration_bounds, duration_ref=duration_ref)

        phase.set_state_options("time", rate_source="dt_dr",
                                fix_initial=fix_initial, fix_final=False, ref=100., defect_ref=100. * 1.e2, solve_segments='forward')

        phase.set_state_options("mass", rate_source="dmass_dr",
                                fix_initial=fix_initial, fix_final=False, ref=170.e3, defect_ref=170.e5,
                                val=170.e3, units='lbm', lower=10.e3)

        phase.add_parameter("wing_area", units="ft**2",
                            static_target=True, opt=False, val=1370)

        phase.add_polynomial_control(Dynamic.Mission.MACH,
                                     order=control_order,
                                     val=0.4, units=mach_bounds[1],
                                     opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                                     fix_initial=fix_initial,
                                     rate_targets=['dmach_dr'],
                                     )

        if rotation:
            phase.add_polynomial_control(Dynamic.Mission.ANGLE_OF_ATTACK,
                                         order=control_order,
                                         fix_initial=True,
                                         lower=0, upper=15,
                                         units='deg', ref=10.,
                                         val=0.,
                                         opt=opt)

        if ground_roll:
            phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                         order=1, val=0, opt=False,
                                         fix_initial=fix_initial,
                                         rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'])
        else:
            phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                         order=control_order,
                                         fix_initial=fix_initial,
                                         units=altitude_bounds[1],
                                         rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                         opt=optimize_altitude, lower=altitude_bounds[
                                             0][0], upper=altitude_bounds[0][1],
                                         ref=(altitude_bounds[0][0] +
                                              altitude_bounds[0][1]) / 2,
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
        phase.add_timeseries_output(Dynamic.Mission.ANGLE_OF_ATTACK)
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
            'balance_throttle': self.user_options.get_val('balance_throttle'),
        }


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

TwoDOFPhase._add_meta_data('throttle_setting', val=None, desc='throttle setting')
TwoDOFPhase._add_meta_data('initial_bounds', val=(0., 100.),
                           units='s', desc='initial bounds')
TwoDOFPhase._add_meta_data('duration_bounds', val=(
    0., 3600.), units='s', desc='duration bounds')
TwoDOFPhase._add_meta_data('initial_ref', val=100., units='s', desc='initial reference')
TwoDOFPhase._add_meta_data('duration_ref', val=1000.,
                           units='s', desc='duration reference')
TwoDOFPhase._add_meta_data('control_order', val=1, desc='control order')
TwoDOFPhase._add_meta_data('opt', val=True, desc='opt')
TwoDOFPhase._add_meta_data('ground_roll', val=True)
TwoDOFPhase._add_meta_data('rotation', val=False)
TwoDOFPhase._add_meta_data('clean', val=False)
TwoDOFPhase._add_meta_data('balance_throttle', val=False)
TwoDOFPhase._add_meta_data('constraints', val={})
TwoDOFPhase._add_meta_data('mach_bounds', val=(0., 2.), units='unitless')
TwoDOFPhase._add_meta_data('altitude_bounds', val=(0., 60.e3), units='ft')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessTime(key='distance'),
    desc='initial guess for initial time and duration specified as a tuple')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('altitude'),
    desc='initial guess for vertical distances')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('mach'),
    desc='initial guess for speed')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('time'),
    desc='initial guess for time')
