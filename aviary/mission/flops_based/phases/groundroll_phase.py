import dymos as dm

from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessPolynomialControl

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE


# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point
@register
class GroundrollPhase(PhaseBuilderBase):
    '''
    A phase builder for a two degree of freedom (2DOF) phase.
    '''
    __slots__ = ('external_subsystems', 'meta_data')

    # region : derived type customization points
    _meta_data_ = {}

    _initial_guesses_meta_data_ = {}

    default_name = 'groundroll'

    default_ode_class = GroundrollODE

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

        duration_bounds = user_options.get_val('duration_bounds', units='kn')
        duration_ref = user_options.get_val('duration_ref', units='kn')
        constraints = user_options.get_val('constraints')

        phase.set_time_options(fix_initial=True, fix_duration=False,
                               units="kn", name=Dynamic.Mission.VELOCITY,
                               duration_bounds=duration_bounds, duration_ref=duration_ref)

        phase.set_state_options("time", rate_source="dt_dv", units="s",
                                fix_initial=True, fix_final=False, ref=1., defect_ref=1., solve_segments='forward')

        phase.set_state_options("mass", rate_source="dmass_dv",
                                fix_initial=True, fix_final=False, lower=1, upper=500.e3, ref=100.e3, defect_ref=100.e3, units='lbm')

        phase.set_state_options(Dynamic.Mission.DISTANCE, rate_source="over_a",
                                fix_initial=True, fix_final=False, lower=0, upper=8000., ref=1.e2, defect_ref=1.e2, units='ft')

        phase.add_parameter("t_init_gear", units="s",
                            static_target=True, opt=False, val=32.3)
        phase.add_parameter("t_init_flaps", units="s",
                            static_target=True, opt=False, val=44.0)
        phase.add_parameter("wing_area", units="ft**2",
                            static_target=True, opt=False, val=1370)

        self._add_user_defined_constraints(phase, constraints)

        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("thrust_req", units="lbf")
        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.VELOCITY, units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output(Dynamic.Mission.DRAG)
        phase.add_timeseries_output("time")
        phase.add_timeseries_output("mass")
        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)
        phase.add_timeseries_output("alpha")
        phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE)
        phase.add_timeseries_output(Dynamic.Mission.THROTTLE)

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
            'set_input_defaults': False,
        }


GroundrollPhase._add_meta_data(
    'num_segments', val=5, desc='transcription: number of segments')

GroundrollPhase._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

GroundrollPhase._add_meta_data('fix_initial', val=False)

GroundrollPhase._add_meta_data('fix_duration', val=False)

GroundrollPhase._add_meta_data('optimize_mach', val=False)

GroundrollPhase._add_meta_data('optimize_altitude', val=False)

GroundrollPhase._add_meta_data('initial_bounds', val=(0., 100.),
                               units='s', desc='initial bounds')
GroundrollPhase._add_meta_data('duration_bounds', val=(
    0., 3600.), units='s', desc='duration bounds')
GroundrollPhase._add_meta_data('initial_ref', val=100.,
                               units='s', desc='initial reference')
GroundrollPhase._add_meta_data('duration_ref', val=1000.,
                               units='s', desc='duration reference')
GroundrollPhase._add_meta_data('control_order', val=1, desc='control order')
GroundrollPhase._add_meta_data('opt', val=True, desc='opt')
GroundrollPhase._add_meta_data('input_speed_type', val='TAS', desc='input speed type')
GroundrollPhase._add_meta_data('ground_roll', val=True)
GroundrollPhase._add_meta_data('rotation', val=False)
GroundrollPhase._add_meta_data('clean', val=False)
GroundrollPhase._add_meta_data('constraints', val={})

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(key='velocity'),
    desc='initial guess for initial velocity and final specified as a tuple')

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('altitude'),
    desc='initial guess for vertical distances')

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for distance')

GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('time'),
    desc='initial guess for time')
