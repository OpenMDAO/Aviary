import dymos as dm

from aviary.mission.flight_phase_builder import FlightPhaseBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl, InitialGuessPolynomialControl

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import UnsteadySolvedODE
from aviary.variable_info.enums import SpeedType, EquationsOfMotion

# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point


@register
class TwoDOFPhase(FlightPhaseBase):
    '''
    A phase builder for a two degree of freedom (2DOF) phase.
    '''

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
        self.ode_class = UnsteadySolvedODE
        phase: dm.Phase = super().build_phase(
            aviary_options, phase_type=EquationsOfMotion.SOLVED_2DOF)

        user_options: AviaryValues = self.user_options

        control_order = user_options.get_val('control_order')

        fix_initial = user_options.get_val('fix_initial')
        duration_bounds = user_options.get_val('duration_bounds', units='ft')
        duration_ref = user_options.get_val('duration_ref', units='ft')
        rotation = user_options.get_val('rotation')

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

        if rotation:
            phase.add_polynomial_control("alpha",
                                         order=control_order,
                                         fix_initial=True,
                                         lower=0, upper=15,
                                         units='deg', ref=10.,
                                         val=0.,
                                         opt=True)

        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.VELOCITY, units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)

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


TwoDOFPhase._add_meta_data('initial_ref', val=100., units='s', desc='initial reference')
TwoDOFPhase._add_meta_data('duration_ref', val=1000.,
                           units='s', desc='duration reference')
TwoDOFPhase._add_meta_data('control_order', val=1, desc='control order')
TwoDOFPhase._add_meta_data('rotation', val=False)
TwoDOFPhase._add_meta_data('clean', val=False)

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(key='distance'),
    desc='initial guess for initial distance and duration specified as a tuple')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessPolynomialControl('alpha'),
    desc='initial guess for alpha')

TwoDOFPhase._add_initial_guess_meta_data(
    InitialGuessState('time'),
    desc='initial guess for time')
