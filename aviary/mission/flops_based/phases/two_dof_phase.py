import dymos as dm

from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessTime, InitialGuessControl

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.mission.flops_based.phases.phase_utils import add_subsystem_variables_to_phase, get_initial
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import UnsteadySolvedODE
from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE


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
        aviary_options : AviaryValues (<emtpy>)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        phase: dm.Phase = super().build_phase(aviary_options)

        user_options: AviaryValues = self.user_options

        throttle_setting = user_options['throttle_setting']
        initial_bounds = user_options['initial_bounds']
        duration_bounds = user_options['duration_bounds']
        initial_ref = user_options['initial_ref']
        duration_ref = user_options['duration_ref']
        control_order = user_options['control_order']
        opt = user_options['opt']

        phase_name = 'rotation'

        phase.add_parameter(
            Dynamic.Mission.THROTTLE,
            opt=False,
            units="unitless",
            val=throttle_setting,
            static_target=False)

        phase.set_time_options(fix_initial=False, fix_duration=False,
                               units="distance_units", name=Dynamic.Mission.DISTANCE,
                               duration_bounds=duration_bounds, duration_ref=duration_ref,
                               initial_bounds=initial_bounds, initial_ref=initial_ref)

        if phase_name == "cruise" or phase_name == "descent":
            time_ref = 1.e4
        else:
            time_ref = 100.

        phase.set_state_options("time", rate_source="dt_dr", targets=['t_curr'] if 'retract' in phase_name else [],
                                fix_initial=False, fix_final=False, ref=time_ref, defect_ref=time_ref * 1.e2)

        phase.set_state_options("mass", rate_source="dmass_dr",
                                fix_initial=False, fix_final=False, ref=170.e3, defect_ref=170.e5,
                                val=170.e3, units='lbm', lower=10.e3)

        phase.add_parameter("wing_area", units="ft**2",
                            static_target=True, opt=False, val=1370)

        if Dynamic.Mission.VELOCITY_RATE in phase_name or 'ascent' in phase_name:
            phase.add_parameter(
                "t_init_gear", units="s", static_target=True, opt=False, val=100)
            phase.add_parameter(
                "t_init_flaps", units="s", static_target=True, opt=False, val=100)

        if 'rotation' in phase_name:
            phase.add_polynomial_control("TAS",
                                         order=control_order,
                                         units="kn", val=200.0,
                                         opt=opt, lower=1, upper=500, ref=250)

            phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                         order=control_order,
                                         fix_initial=False,
                                         rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                         opt=False, upper=40.e3, ref=30.e3, lower=-1.)

            phase.add_polynomial_control("alpha",
                                         order=control_order,
                                         lower=-4, upper=15,
                                         units='deg', ref=10.,
                                         val=[0., 5.],
                                         opt=opt)
        # else:
        #     if 'constant_EAS' in phase_name:
        #         fixed_EAS = phase_info[phase_name]['fixed_EAS']
        #         phase.add_parameter("EAS", units="kn", val=fixed_EAS)
        #     elif 'constant_mach' in phase_name:
        #         phase.add_parameter(
        #             Dynamic.Mission.MACH,
        #             units="unitless",
        #             val=climb_mach)
        #     elif 'cruise' in phase_name:
        #         phase.add_parameter(
        #             Dynamic.Mission.MACH, units="unitless", val=cruise_mach)
        #     else:
        #         phase.add_polynomial_control("TAS",
        #                                         order=control_order,
        #                                         fix_initial=False,
        #                                         units="kn", val=200.0,
        #                                         opt=True, lower=1, upper=500, ref=250)

        #     phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
        #                                     order=control_order,
        #                                     units="ft", val=0.,
        #                                     fix_initial=False,
        #                                     rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
        #                                     opt=opt, upper=40.e3, ref=30.e3, lower=-1.)

        # if phase_name in phases[1:3]:
        #     phase.add_path_constraint(
        #         "fuselage_pitch", upper=15., units='deg', ref=15)
        # if phase_name == "rotation":
        #     phase.add_boundary_constraint(
        #         "TAS", loc="final", upper=200., units="kn", ref=200.)
        #     phase.add_boundary_constraint(
        #         "normal_force", loc="final", equals=0., units="lbf", ref=10000.0)
        # elif phase_name == "ascent_to_gear_retract":
        #     phase.add_path_constraint("load_factor", lower=0.0, upper=1.10)
        # elif phase_name == "ascent_to_flap_retract":
        #     phase.add_path_constraint("load_factor", lower=0.0, upper=1.10)
        # elif phase_name == "ascent":
        #     phase.add_path_constraint("EAS", upper=250., units="kn", ref=250.)
        # elif phase_name == Dynamic.Mission.VELOCITY_RATE:
        #     phase.add_boundary_constraint(
        #         "EAS", loc="final", equals=250., units="kn", ref=250.)
        # elif phase_name == "climb_at_constant_EAS":
        #     pass
        # elif phase_name == "climb_at_constant_EAS_to_mach":
        #     phase.add_boundary_constraint(
        #         Dynamic.Mission.MACH, loc="final", equals=climb_mach, units="unitless")
        # elif phase_name == "climb_at_constant_mach":
        #     pass
        # elif phase_name == "descent":
        #     phase.add_boundary_constraint(
        #         Dynamic.Mission.DISTANCE,
        #         loc="final",
        #         equals=target_range,
        #         units="NM",
        #         ref=1.e3)
        #     phase.add_boundary_constraint(
        #         Dynamic.Mission.ALTITUDE,
        #         loc="final",
        #         equals=10.e3,
        #         units="ft",
        #         ref=10e3)
        #     phase.add_boundary_constraint(
        #         "TAS", loc="final", equals=250., units="kn", ref=250.)

        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("thrust_req", units="lbf")
        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output("TAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output("CL")
        phase.add_timeseries_output("CD")
        phase.add_timeseries_output("time")
        phase.add_timeseries_output("mass")
        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)
        phase.add_timeseries_output("gear_factor")
        phase.add_timeseries_output("flap_factor")
        phase.add_timeseries_output("alpha")
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
            'input_speed_type': self.user_options.get_val('input_speed_type'),
            'clean': self.user_options.get_val('clean'),
            'ground_roll': self.user_options.get_val('ground_roll'),
        }


TwoDOFPhase._add_meta_data(
    'num_segments', val=5, desc='transcription: number of segments')

TwoDOFPhase._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

TwoDOFPhase._add_meta_data('polynomial_control_order', val=3)

TwoDOFPhase._add_meta_data('use_polynomial_control', val=True)

TwoDOFPhase._add_meta_data('add_initial_mass_constraint', val=False)

TwoDOFPhase._add_meta_data('fix_initial', val=True)

TwoDOFPhase._add_meta_data('fix_duration', val=False)

TwoDOFPhase._add_meta_data('optimize_mach', val=False)

TwoDOFPhase._add_meta_data('optimize_altitude', val=False)

TwoDOFPhase._add_meta_data('initial_ref', val=100., units='s')

TwoDOFPhase._add_meta_data('duration_ref', val=100., units='s')

TwoDOFPhase._add_meta_data('initial_bounds', val=(0., 100.), units='s')

TwoDOFPhase._add_meta_data('duration_bounds', val=(0., 100.), units='s')

TwoDOFPhase._add_meta_data(
    'required_available_climb_rate', val=None, units='m/s',
    desc='minimum avaliable climb rate')

TwoDOFPhase._add_meta_data(
    'no_climb', val=False, desc='aircraft is not allowed to climb during phase')

TwoDOFPhase._add_meta_data(
    'no_descent', val=False, desc='aircraft is not allowed to descend during phase')

TwoDOFPhase._add_meta_data('constrain_final', val=False)

TwoDOFPhase._add_meta_data('input_initial', val=False)

TwoDOFPhase._add_meta_data('initial_mach', val=None, units='unitless')

TwoDOFPhase._add_meta_data('final_mach', val=None, units='unitless')

TwoDOFPhase._add_meta_data('initial_altitude', val=None, units='ft')

TwoDOFPhase._add_meta_data('final_altitude', val=None, units='ft')

TwoDOFPhase._add_meta_data('throttle_enforcement', val=None)

TwoDOFPhase._add_meta_data('throttle_setting', val=None)

TwoDOFPhase._add_meta_data('input_speed_type', val=None)

TwoDOFPhase._add_meta_data('clean', val=False)

TwoDOFPhase._add_meta_data('ground_roll', val=None)

TwoDOFPhase._add_meta_data('mach_bounds', val=(0., 2.), units='unitless')

TwoDOFPhase._add_meta_data('altitude_bounds', val=(0., 60.e3), units='ft')

TwoDOFPhase._add_meta_data('solve_for_distance', val=False)

TwoDOFPhase._add_meta_data('constraints', val={})

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for initial time and duration specified as a tuple')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for horizontal distance traveled')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('mach'),
    desc='initial guess for speed')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')
