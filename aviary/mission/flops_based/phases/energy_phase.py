from math import isclose

import dymos as dm

from aviary.mission.flops_based.phases.phase_builder_base import (
    register, PhaseBuilderBase, InitialGuessControl, InitialGuessParameter,
    InitialGuessPolynomialControl, InitialGuessState, InitialGuessTime)

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.mission.flops_based.phases.phase_utils import add_subsystem_variables_to_phase
from aviary.variable_info.variables import Dynamic


# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point
class EnergyPhase(PhaseBuilderBase):
    '''
    Define a phase builder base class for typical energy phases such as climb and
    descent.

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
            - fix_initial_time : bool (None)
            - initial_ref : float (1.0, 's')
                note: applied if, and only if, "fix_initial_time" is unspecified
            - initial_bounds : pair ((0.0, 100.0) 's')
                note: applied if, and only if, "fix_initial_time" is unspecified
            - duration_ref : float (1.0, 's')
            - duration_bounds : pair ((0.0, 100.0) 's')
            - initial_altitude : float (0.0, 'ft)
                starting true altitude from mean sea level
            - final_altitude : float
                ending true altitude from mean sea level
            - initial_mach : float (0.0, 'ft)
                starting Mach number
            - final_mach : float
                ending Mach number
            - required_altitude_rate : float (None)
                minimum avaliable climb rate
            - no_climb : bool (False)
                aircraft is not allowed to climb during phase
            - no_descent : bool (False)
                aircraft is not allowed to descend during phase
            - fix_range : bool (False)
            - input_initial : bool (False)
            - polynomial_control_order : None or int
                When set to an integer value, replace controls with
                polynomial controls of that specified order.


    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
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

    default_name = 'energy_phase'

    # default_ode_class = MissionODE

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

        fix_initial = user_options.get_val('fix_initial')
        initial_mach = user_options.get_val('initial_mach')
        final_mach = user_options.get_val('final_mach')
        initial_altitude = user_options.get_val('initial_altitude', 'm')
        final_altitude = user_options.get_val('final_altitude', 'm')
        no_descent = user_options.get_val('no_descent')
        no_climb = user_options.get_val('no_climb')
        fix_range = user_options.get_val('fix_range')
        input_initial = user_options.get_val('input_initial')
        polynomial_control_order = user_options.get_item('polynomial_control_order')[0]
        num_engines = len(aviary_options.get_val('engine_models'))

        # NOTE currently defaulting to climb if altitudes match. Review at future date
        climb = final_altitude >= initial_altitude
        descent = initial_altitude > final_altitude

        max_altitude = max(initial_altitude, final_altitude)
        min_altitude = min(initial_altitude, final_altitude)
        max_mach = max(initial_mach, final_mach)
        min_mach = min(initial_mach, final_mach)

        if climb:
            lower_accel = 0.0
        else:
            lower_accel = -2.12

        if descent:
            upper_accel = 0.0
        else:
            upper_accel = 2.12

        ##############
        # Add States #
        ##############
        # TODO: critically think about how we should handle fix_initial and input_initial defaults.
        # In keeping with Dymos standards, the default should be False instead of True.
        def get_initial(input_initial, key, input_initial_for_this_variable=False):
            # Check if input_initial is a dictionary.
            # If so, return the value corresponding to the key or False if the key is not found.
            # If not, return the value of input_initial.
            if isinstance(input_initial, dict):
                if key in input_initial:
                    input_initial_for_this_variable = input_initial[key]
            elif isinstance(input_initial, bool):
                input_initial_for_this_variable = input_initial
            return input_initial_for_this_variable

        input_initial_alt = get_initial(input_initial, Dynamic.Mission.ALTITUDE)
        fix_initial_alt = get_initial(fix_initial, Dynamic.Mission.ALTITUDE)
        phase.add_state(
            Dynamic.Mission.ALTITUDE, fix_initial=fix_initial_alt, fix_final=False,
            lower=0.0, ref=1e4, defect_ref=1e4, units='m',
            rate_source=Dynamic.Mission.ALTITUDE_RATE, targets=Dynamic.Mission.ALTITUDE,
            input_initial=input_initial_alt
        )

        input_initial_vel = get_initial(input_initial, Dynamic.Mission.VELOCITY)
        fix_initial_vel = get_initial(fix_initial, Dynamic.Mission.VELOCITY)
        phase.add_state(
            Dynamic.Mission.VELOCITY, fix_initial=fix_initial_vel, fix_final=False,
            lower=0.0, ref=1e2, defect_ref=1e2, units='m/s',
            rate_source=Dynamic.Mission.VELOCITY_RATE, targets=Dynamic.Mission.VELOCITY,
            input_initial=input_initial_vel
        )

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
            lower=0.0, ref=1e4, defect_ref=1e4, units='kg',
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            input_initial=input_initial_mass
        )

        input_initial_range = get_initial(input_initial, Dynamic.Mission.RANGE)
        fix_initial_range = get_initial(fix_initial, Dynamic.Mission.RANGE, True)
        phase.add_state(
            Dynamic.Mission.RANGE, fix_initial=fix_initial_range, fix_final=fix_range,
            lower=0.0, ref=1e6, defect_ref=1e6, units='m',
            rate_source=Dynamic.Mission.RANGE_RATE,
            input_initial=input_initial_range
        )

        phase = add_subsystem_variables_to_phase(
            phase, self.name, self.external_subsystems)

        ################
        # Add Controls #
        ################
        if polynomial_control_order is not None:
            phase.add_polynomial_control(
                Dynamic.Mission.VELOCITY_RATE,
                targets=Dynamic.Mission.VELOCITY_RATE, units='m/s**2',
                opt=True, lower=lower_accel, upper=upper_accel,
                order=polynomial_control_order,
            )
        else:
            phase.add_control(
                Dynamic.Mission.VELOCITY_RATE,
                targets=Dynamic.Mission.VELOCITY_RATE, units='m/s**2',
                opt=True, lower=lower_accel, upper=upper_accel
            )

        if num_engines > 0:
            if polynomial_control_order is not None:
                phase.add_polynomial_control(
                    Dynamic.Mission.THROTTLE,
                    targets=Dynamic.Mission.THROTTLE, units='unitless',
                    opt=True, lower=0.0, upper=1.0, scaler=1,
                    order=polynomial_control_order,
                )
            else:
                phase.add_control(
                    Dynamic.Mission.THROTTLE,
                    targets=Dynamic.Mission.THROTTLE, units='unitless',
                    opt=True, lower=0.0, upper=1.0, scaler=1,
                )
            if climb:
                phase.add_path_constraint(
                    Dynamic.Mission.THROTTLE,
                    upper=1.0,
                )

            if descent:
                phase.add_path_constraint(
                    Dynamic.Mission.THROTTLE,
                    lower=0.0,
                )

        # check if engine has use_hybrid_throttle
        engine_models = aviary_options.get_val('engine_models')
        if any([engine.use_hybrid_throttle for engine in engine_models]):
            if polynomial_control_order is not None:
                phase.add_polynomial_control(
                    Dynamic.Mission.HYBRID_THROTTLE,
                    targets=Dynamic.Mission.HYBRID_THROTTLE, units='unitless',
                    opt=True, lower=0.0, upper=1.0, scaler=1,
                    order=polynomial_control_order,
                )
            else:
                phase.add_control(
                    Dynamic.Mission.HYBRID_THROTTLE,
                    targets=Dynamic.Mission.HYBRID_THROTTLE, units='unitless',
                    opt=True, lower=0.0, upper=1.0, scaler=1,
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
            Dynamic.Mission.THRUST_MAX_TOTAL,
            output_name=Dynamic.Mission.THRUST_MAX_TOTAL, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.DRAG, output_name=Dynamic.Mission.DRAG, units='lbf'
        )

        phase.add_timeseries_output(
            Dynamic.Mission.SPECIFIC_ENERGY_RATE,
            output_name=Dynamic.Mission.SPECIFIC_ENERGY_RATE, units='m/s'
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
            Dynamic.Mission.ALTITUDE_RATE_MAX,
            output_name=Dynamic.Mission.ALTITUDE_RATE_MAX, units='ft/min'
        )

        ###################
        # Add Constraints #
        ###################
        if climb and no_descent:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE, constraint_name='no_descent',
                lower=0.0
            )

        if descent and no_climb:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE, constraint_name='no_climb',
                upper=0.0
            )

        if descent and num_engines > 0:
            phase.add_boundary_constraint(
                Dynamic.Mission.THROTTLE,
                loc='final',
                upper=1.0,
                lower=0.0,
                units='unitless')

        required_altitude_rate, units = user_options.get_item('required_altitude_rate')

        if required_altitude_rate is not None:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE_MAX,
                lower=required_altitude_rate, units=units
            )

        phase.add_boundary_constraint(
            Dynamic.Mission.MACH, loc='final', equals=final_mach
        )

        if isclose(initial_mach, final_mach):
            phase.add_path_constraint(
                Dynamic.Mission.MACH,
                equals=min_mach, units='unitless',
                # ref=1.e4,
            )

        else:
            phase.add_path_constraint(
                Dynamic.Mission.MACH,
                lower=min_mach, upper=max_mach, units='unitless',
                # ref=1.e4,
            )

        # avoid scaling constraints by zero
        ref = final_altitude
        if ref == 0:
            ref = None

        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE, loc='final', equals=final_altitude, ref=ref
        )

        # avoid scaling constraints by zero
        ref = max_altitude
        if ref == 0:
            ref = None

        phase.add_path_constraint(
            Dynamic.Mission.ALTITUDE,
            lower=min_altitude, upper=max_altitude,
            units='m', ref=ref
        )

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
        }


EnergyPhase._add_meta_data(
    'num_segments', val=5, desc='transcription: number of segments')

EnergyPhase._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

EnergyPhase._add_meta_data('add_initial_mass_constraint', val=False)

EnergyPhase._add_meta_data('fix_initial', val=True)

EnergyPhase._add_meta_data('fix_initial_time', val=None)

EnergyPhase._add_meta_data('initial_ref', val=1., units='s')

EnergyPhase._add_meta_data('initial_bounds', val=(0., 100.), units='s')

EnergyPhase._add_meta_data('duration_ref', val=1., units='s')

EnergyPhase._add_meta_data('duration_bounds', val=(0., 100.), units='s')

EnergyPhase._add_meta_data('cruise_alt', val=11.e3, units='m')

EnergyPhase._add_meta_data(
    'initial_altitude', val=0., units='m',
    desc='starting true altitude from mean sea level')

EnergyPhase._add_meta_data(
    'final_altitude', val=None, units='m',
    desc='ending true altitude from mean sea level')

EnergyPhase._add_meta_data('initial_mach', val=0., desc='starting Mach number')

EnergyPhase._add_meta_data('final_mach', val=None, desc='ending Mach number')

EnergyPhase._add_meta_data(
    'required_altitude_rate', val=None, units='m/s',
    desc='minimum avaliable climb rate')

EnergyPhase._add_meta_data(
    'no_climb', val=False, desc='aircraft is not allowed to climb during phase')

EnergyPhase._add_meta_data(
    'no_descent', val=False, desc='aircraft is not allowed to descend during phase')

EnergyPhase._add_meta_data('fix_range', val=False)

EnergyPhase._add_meta_data('input_initial', val=False)

EnergyPhase._add_meta_data('polynomial_control_order', val=None)

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for initial time and duration specified as a tuple')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('range'),
    desc='initial guess for horizontal distance traveled')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'),
    desc='initial guess for speed')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessControl('velocity_rate'),
    desc='initial guess for acceleration')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
