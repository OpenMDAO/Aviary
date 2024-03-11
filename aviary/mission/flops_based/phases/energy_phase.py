import dymos as dm

from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.mission.flops_based.phases.phase_utils import add_subsystem_variables_to_phase, get_initial
from aviary.variable_info.variables import Dynamic
from aviary.mission.flops_based.ode.mission_ODE import MissionODE


# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point
@register
class EnergyPhase(PhaseBuilderBase):
    '''
    A phase builder for a simple energy phase.

    Attributes
    ----------
    name : str ('energy_phase')
        object label

    subsystem_options (None)
        dictionary of parameters to be passed to the subsystem builders

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - num_segments : int (5)
                transcription: number of segments
            - order : int (3)
                transcription: order of the state transcription; the order of the control
                transcription is `order - 1`
            - fix_initial : bool (True)
            - initial_ref : float (1.0, 's')
            - initial_bounds : pair ((0.0, 100.0) 's')
            - duration_ref : float (1.0, 's')
            - duration_bounds : pair ((0.0, 100.0) 's')
            - required_available_climb_rate : float (None)
                minimum avaliable climb rate
            - constrain_final : bool (False)
            - input_initial : bool (False)
            - reserve : bool (False)
                Determines if the phase is handled as a reserve phase or a regular phase.
            - target_duration : pair (float, str) (None)
                Adds a duration constraint to the phase. Can be used with reserve and non-reserve phases.
                Example: "target_duration": (60, 'min')
            - target_distance : pair (float, str) (None)
                Adds a distance constraint on the phase. Can be used with reserve and non-reserve phases.
                Example: "target_distance": (200, 'km')

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - distance
            - altitude
            - velocity
            - velocity_rate
            - mass
            - throttle

    ode_class : type (None)
        advanced: the type of system defining the ODE

    transcription : "Dymos transcription object" (None)
        advanced: an object providing the transcription technique of the
        optimal control problem

    external_subsystems : Sequence["subsystem builder"] (<empty>)
        advanced

    meta_data : dict (<"builtin" meta data>)
        advanced: meta data associated with variables in the Aviary data hierarchy

    default_name : str
        class attribute: derived type customization point; the default value
        for name

    default_ode_class : type
        class attribute: derived type customization point; the default value
        for ode_class used by build_phase

    default_meta_data : dict
        class attribute: derived type customization point; the default value for
        meta_data

    Methods
    -------
    build_phase
    make_default_transcription
    validate_options
    assign_default_options
    '''
    __slots__ = ('external_subsystems', 'meta_data')

    # region : derived type customization points
    _meta_data_ = {}

    _initial_guesses_meta_data_ = {}

    default_name = 'cruise'

    default_ode_class = MissionODE

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

        fix_initial = user_options.get_val('fix_initial')
        constrain_final = user_options.get_val('constrain_final')
        optimize_mach = user_options.get_val('optimize_mach')
        optimize_altitude = user_options.get_val('optimize_altitude')
        input_initial = user_options.get_val('input_initial')
        polynomial_control_order = user_options.get_item('polynomial_control_order')[0]
        use_polynomial_control = user_options.get_val('use_polynomial_control')
        throttle_enforcement = user_options.get_val('throttle_enforcement')
        mach_bounds = user_options.get_item('mach_bounds')
        altitude_bounds = user_options.get_item('altitude_bounds')
        initial_mach = user_options.get_item('initial_mach')[0]
        final_mach = user_options.get_item('final_mach')[0]
        initial_altitude = user_options.get_item('initial_altitude')[0]
        final_altitude = user_options.get_item('final_altitude')[0]
        solve_for_distance = user_options.get_val('solve_for_distance')
        no_descent = user_options.get_val('no_descent')
        no_climb = user_options.get_val('no_climb')
        constraints = user_options.get_val('constraints')

        ##############
        # Add States #
        ##############
        # TODO: critically think about how we should handle fix_initial and input_initial defaults.
        # In keeping with Dymos standards, the default should be False instead of True.
        input_initial_mass = get_initial(input_initial, Dynamic.Mission.MASS)
        fix_initial_mass = get_initial(fix_initial, Dynamic.Mission.MASS, True)

        # Experiment: use a constraint for mass instead of connected initial.
        # This is due to some problems in mpi.
        # This is needed for the cutting edge full subsystem integration.
        # TODO: when a Dymos fix is in and we've verified that full case works with the fix,
        # remove this argument.
        if user_options.get_val('add_initial_mass_constraint'):
            phase.add_constraint('rhs_all.initial_mass_residual', equals=0.0, ref=1e4)
            input_initial_mass = False

        phase.add_state(
            Dynamic.Mission.MASS, fix_initial=fix_initial_mass, fix_final=False,
            lower=0.0, ref=1e4, defect_ref=1e6, units='kg',
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            input_initial=input_initial_mass,
            # solve_segments='forward',
        )

        input_initial_distance = get_initial(input_initial, Dynamic.Mission.DISTANCE)
        fix_initial_distance = get_initial(fix_initial, Dynamic.Mission.DISTANCE, True)
        phase.add_state(
            Dynamic.Mission.DISTANCE, fix_initial=fix_initial_distance, fix_final=False,
            lower=0.0, ref=1e6, defect_ref=1e8, units='m',
            rate_source=Dynamic.Mission.DISTANCE_RATE,
            input_initial=input_initial_distance,
            solve_segments='forward' if solve_for_distance else None,
        )

        phase = add_subsystem_variables_to_phase(
            phase, self.name, self.external_subsystems)

        ################
        # Add Controls #
        ################
        if use_polynomial_control:
            phase.add_polynomial_control(
                Dynamic.Mission.MACH,
                targets=Dynamic.Mission.MACH, units=mach_bounds[1],
                opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                rate_targets=[Dynamic.Mission.MACH_RATE],
                order=polynomial_control_order, ref=0.5,
            )
        else:
            phase.add_control(
                Dynamic.Mission.MACH,
                targets=Dynamic.Mission.MACH, units=mach_bounds[1],
                opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                rate_targets=[Dynamic.Mission.MACH_RATE],
                ref=0.5,
            )

        # Add altitude rate as a control
        if use_polynomial_control:
            phase.add_polynomial_control(
                Dynamic.Mission.ALTITUDE,
                targets=Dynamic.Mission.ALTITUDE, units=altitude_bounds[1],
                opt=optimize_altitude, lower=altitude_bounds[0][0], upper=altitude_bounds[0][1],
                rate_targets=[Dynamic.Mission.ALTITUDE_RATE],
                order=polynomial_control_order, ref=altitude_bounds[0][1],
            )
        else:
            phase.add_control(
                Dynamic.Mission.ALTITUDE,
                targets=Dynamic.Mission.ALTITUDE, units=altitude_bounds[1],
                opt=optimize_altitude, lower=altitude_bounds[0][0], upper=altitude_bounds[0][1],
                rate_targets=[Dynamic.Mission.ALTITUDE_RATE],
                ref=altitude_bounds[0][1],
            )

        ##################
        # Add Timeseries #
        ##################
        phase.add_timeseries_output(
            Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units='unitless'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.THRUST_TOTAL,
            output_name=Dynamic.Mission.THRUST_TOTAL, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.DRAG, output_name=Dynamic.Mission.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
            output_name=Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS, units='m/s'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            output_name=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/h'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.ALTITUDE_RATE,
            output_name=Dynamic.Mission.ALTITUDE_RATE, units='ft/s'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.THROTTLE,
            output_name=Dynamic.Mission.THROTTLE, units='unitless'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY, units='m/s'
        )

        ###################
        # Add Constraints #
        ###################
        if optimize_mach and fix_initial and not Dynamic.Mission.MACH in constraints:
            phase.add_boundary_constraint(
                Dynamic.Mission.MACH, loc='initial', equals=initial_mach,
            )

        if optimize_mach and constrain_final and not Dynamic.Mission.MACH in constraints:
            phase.add_boundary_constraint(
                Dynamic.Mission.MACH, loc='final', equals=final_mach,
            )

        if optimize_altitude and fix_initial and not Dynamic.Mission.ALTITUDE in constraints:
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE, loc='initial', equals=initial_altitude, units=altitude_bounds[1], ref=1.e4,
            )

        if optimize_altitude and constrain_final and not Dynamic.Mission.ALTITUDE in constraints:
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE, loc='final', equals=final_altitude, units=altitude_bounds[1], ref=1.e4,
            )

        if no_descent and not Dynamic.Mission.ALTITUDE_RATE in constraints:
            phase.add_path_constraint(Dynamic.Mission.ALTITUDE_RATE, lower=0.0)

        if no_climb and not Dynamic.Mission.ALTITUDE_RATE in constraints:
            phase.add_path_constraint(Dynamic.Mission.ALTITUDE_RATE, upper=0.0)

        required_available_climb_rate, units = user_options.get_item(
            'required_available_climb_rate')

        if required_available_climb_rate is not None and not Dynamic.Mission.ALTITUDE_RATE_MAX in constraints:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE_MAX,
                lower=required_available_climb_rate, units=units
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
            'throttle_enforcement': self.user_options.get_val('throttle_enforcement'),
        }


EnergyPhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')

EnergyPhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')

EnergyPhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')

EnergyPhase._add_meta_data(
    'num_segments', val=5, desc='transcription: number of segments')

EnergyPhase._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

EnergyPhase._add_meta_data('polynomial_control_order', val=3)

EnergyPhase._add_meta_data('use_polynomial_control', val=True)

EnergyPhase._add_meta_data('add_initial_mass_constraint', val=False)

EnergyPhase._add_meta_data('fix_initial', val=True)

EnergyPhase._add_meta_data('fix_duration', val=False)

EnergyPhase._add_meta_data('optimize_mach', val=False)

EnergyPhase._add_meta_data('optimize_altitude', val=False)

EnergyPhase._add_meta_data('initial_bounds', val=(0., 100.), units='s')

EnergyPhase._add_meta_data('duration_bounds', val=(0., 100.), units='s')

EnergyPhase._add_meta_data(
    'required_available_climb_rate', val=None, units='m/s',
    desc='minimum avaliable climb rate')

EnergyPhase._add_meta_data(
    'no_climb', val=False, desc='aircraft is not allowed to climb during phase')

EnergyPhase._add_meta_data(
    'no_descent', val=False, desc='aircraft is not allowed to descend during phase')

EnergyPhase._add_meta_data('constrain_final', val=False)

EnergyPhase._add_meta_data('input_initial', val=False)

EnergyPhase._add_meta_data('initial_mach', val=None, units='unitless')

EnergyPhase._add_meta_data('final_mach', val=None, units='unitless')

EnergyPhase._add_meta_data('initial_altitude', val=None, units='ft')

EnergyPhase._add_meta_data('final_altitude', val=None, units='ft')

EnergyPhase._add_meta_data('throttle_enforcement', val=None)

EnergyPhase._add_meta_data('mach_bounds', val=(0., 2.), units='unitless')

EnergyPhase._add_meta_data('altitude_bounds', val=(0., 60.e3), units='ft')

EnergyPhase._add_meta_data('solve_for_distance', val=False)

EnergyPhase._add_meta_data('constraints', val={})

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for horizontal distance traveled')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('mach'),
    desc='initial guess for speed')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')
