from math import isclose

import dymos as dm

from aviary.mission.flops_based.phases.phase_builder_base import (
    register, PhaseBuilderBase, InitialGuessControl, InitialGuessParameter,
    InitialGuessPolynomialControl, InitialGuessState, InitialGuessTime)

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase

from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic
from aviary.mission.flops_based.phases.phase_utils import add_subsystem_variables_to_phase


Cruise = None  # forward declaration for type hints


# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point
class Cruise(PhaseBuilderBase):
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
            - min_altitude : float (0.0, 'ft)
                starting true altitude from mean sea level
            - max_altitude : float
                ending true altitude from mean sea level
            - min_mach : float (0.0, 'ft)
                starting Mach number
            - max_mach : float
                ending Mach number
            - required_available_climb_rate : float (None)
                minimum avaliable climb rate
            - fix_final : bool (False)
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

    default_name = 'cruise_phase'

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
        Return a new cruise phase for analysis using these constraints.

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
        min_mach = user_options.get_val('min_mach')
        max_mach = user_options.get_val('max_mach')
        min_altitude = user_options.get_val('min_altitude', units='m')
        max_altitude = user_options.get_val('max_altitude', units='m')
        fix_final = user_options.get_val('fix_final')
        input_initial = user_options.get_val('input_initial')
        num_engines = len(aviary_options.get_val('engine_models'))
        input_initial = user_options.get_val('input_initial')
        mass_f_cruise = user_options.get_val('mass_f_cruise', units='kg')
        range_f_cruise = user_options.get_val('range_f_cruise', units='m')
        polynomial_control_order = user_options.get_item('polynomial_control_order')[0]

        ##############
        # Add States #
        ##############
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
            lower=0.0, ref=max_altitude, defect_ref=max_altitude, units='m',
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
        phase.add_state(
            Dynamic.Mission.MASS, fix_initial=fix_initial_mass, fix_final=False,
            lower=0.0, ref=mass_f_cruise, defect_ref=mass_f_cruise, units='kg',
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            input_initial=input_initial_mass
        )

        input_initial_range = get_initial(input_initial, Dynamic.Mission.RANGE)
        fix_initial_range = get_initial(fix_initial, Dynamic.Mission.RANGE, True)
        phase.add_state(
            Dynamic.Mission.RANGE, fix_initial=fix_initial_range, fix_final=fix_final,
            lower=0.0, ref=range_f_cruise, defect_ref=range_f_cruise, units='m',
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
                opt=True, lower=-2.12, upper=2.12,
                order=polynomial_control_order,
            )
        else:
            phase.add_control(
                Dynamic.Mission.VELOCITY_RATE,
                targets=Dynamic.Mission.VELOCITY_RATE, units='m/s**2',
                opt=True, lower=-2.12, upper=2.12,
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
        required_available_climb_rate = user_options.get_val(
            'required_available_climb_rate', 'm/s')

        if required_available_climb_rate is not None:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE_MAX,
                ref=required_available_climb_rate,
                constraint_name='altitude_rate_minimum',
                lower=required_available_climb_rate, units='m/s'
            )

        if isclose(min_mach, max_mach):
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
        ref = max_altitude
        if ref == 0:
            ref = None

        if isclose(min_altitude, max_altitude):
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE,
                equals=min_altitude, units='m', ref=ref,
            )

        else:
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE,
                lower=min_altitude, upper=max_altitude,
                units='m', ref=ref,
            )

        return phase

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'external_subsystems': self.external_subsystems,
            'meta_data': self.meta_data}


Cruise._add_meta_data(
    'num_segments', val=1, desc='transcription: number of segments')

Cruise._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

Cruise._add_meta_data('fix_initial', val=True)

Cruise._add_meta_data('fix_initial_time', val=None)

Cruise._add_meta_data('initial_ref', val=1., units='s')

Cruise._add_meta_data('initial_bounds', val=(0., 100.), units='s')

Cruise._add_meta_data('duration_ref', val=1., units='s')

Cruise._add_meta_data('duration_bounds', val=(0., 100.), units='s')

Cruise._add_meta_data(
    'min_altitude', val=0., units='m',
    desc='starting true altitude from mean sea level')

Cruise._add_meta_data(
    'max_altitude', val=None, units='m',
    desc='ending true altitude from mean sea level')

Cruise._add_meta_data('min_mach', val=0., desc='starting Mach number')

Cruise._add_meta_data('max_mach', val=None, desc='ending Mach number')

Cruise._add_meta_data('mass_i_cruise', val=1.e4, units='kg')

Cruise._add_meta_data('mass_f_cruise', val=1.e4, units='kg')

Cruise._add_meta_data('range_f_cruise', val=1.e6, units='m')

Cruise._add_meta_data(
    'required_available_climb_rate', val=None, units='m/s',
    desc='minimum avaliable climb rate')

Cruise._add_meta_data('fix_final', val=True)

Cruise._add_meta_data('input_initial', val=False)

Cruise._add_meta_data('polynomial_control_order', val=None)

Cruise._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for initial time and duration specified as a tuple')

Cruise._add_initial_guess_meta_data(
    InitialGuessState('range'),
    desc='initial guess for horizontal distance traveled')

Cruise._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

Cruise._add_initial_guess_meta_data(
    InitialGuessState('velocity'),
    desc='initial guess for speed')

Cruise._add_initial_guess_meta_data(
    InitialGuessControl('velocity_rate'),
    desc='initial guess for acceleration')

Cruise._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

Cruise._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
