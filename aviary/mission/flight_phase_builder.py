import numpy as np

import dymos as dm

from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.mission.flops_based.phases.phase_utils import add_subsystem_variables_to_phase, get_initial
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.variable_info.enums import EquationsOfMotion, ThrottleAllocation
from aviary.variable_info.variables import Aircraft


# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point
@register
class FlightPhaseBase(PhaseBuilderBase):
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

    def build_phase(self, aviary_options: AviaryValues = None, phase_type=EquationsOfMotion.HEIGHT_ENERGY):
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

        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))

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

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_source = Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL
        else:
            rate_source = "dmass_dr"

        phase.add_state(
            Dynamic.Mission.MASS, fix_initial=fix_initial_mass, fix_final=False,
            lower=0.0, ref=1e4, defect_ref=1e6, units='kg',
            rate_source=rate_source,
            targets=Dynamic.Mission.MASS,
            input_initial=input_initial_mass,
        )

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            input_initial_distance = get_initial(input_initial, Dynamic.Mission.DISTANCE)
            fix_initial_distance = get_initial(
                fix_initial, Dynamic.Mission.DISTANCE, True)
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
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_targets = [Dynamic.Mission.MACH_RATE]
        else:
            rate_targets = ['dmach_dr']

        if use_polynomial_control:
            phase.add_polynomial_control(
                Dynamic.Mission.MACH,
                targets=Dynamic.Mission.MACH, units=mach_bounds[1],
                opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                rate_targets=rate_targets,
                order=polynomial_control_order, ref=0.5,
            )
        else:
            phase.add_control(
                Dynamic.Mission.MACH,
                targets=Dynamic.Mission.MACH, units=mach_bounds[1],
                opt=optimize_mach, lower=mach_bounds[0][0], upper=mach_bounds[0][1],
                rate_targets=rate_targets,
                ref=0.5,
            )

        # Add altitude rate as a control
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_targets = [Dynamic.Mission.ALTITUDE_RATE]
            rate2_targets = []
        else:
            rate_targets = ['dh_dr']
            rate2_targets = ['d2h_dr2']

        # For multi-engine cases, we may have throttle allocation control.
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY and num_engine_type > 1:
            allocation = user_options.get_val('throttle_allocation')

            # Allocation should default to an even split so that we don't start
            # with an allocation that might not produce enough thrust.
            val = np.ones(num_engine_type - 1) * (1.0 / num_engine_type)

            if allocation == ThrottleAllocation.DYNAMIC:
                phase.add_control(
                    "throttle_allocations",
                    shape=(num_engine_type - 1, ),
                    val=val,
                    targets="throttle_allocations", units="unitless",
                    opt=True, lower=0.0, upper=1.0,
                )

            else:
                if allocation == ThrottleAllocation.STATIC:
                    opt = True
                else:
                    opt = False

                phase.add_parameter(
                    "throttle_allocations",
                    units="unitless",
                    val=val,
                    shape=(num_engine_type - 1, ),
                    opt=opt, lower=0.0, upper=1.0,
                )

        ground_roll = user_options.get_val('ground_roll')
        if ground_roll:
            phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                         order=1, val=0, opt=False,
                                         fix_initial=fix_initial,
                                         rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'])
        else:
            if use_polynomial_control:
                phase.add_polynomial_control(
                    Dynamic.Mission.ALTITUDE,
                    targets=Dynamic.Mission.ALTITUDE, units=altitude_bounds[1],
                    opt=optimize_altitude, lower=altitude_bounds[0][0], upper=altitude_bounds[0][1],
                    rate_targets=rate_targets, rate2_targets=rate2_targets,
                    order=polynomial_control_order, ref=altitude_bounds[0][1],
                )
            else:
                phase.add_control(
                    Dynamic.Mission.ALTITUDE,
                    targets=Dynamic.Mission.ALTITUDE, units=altitude_bounds[1],
                    opt=optimize_altitude, lower=altitude_bounds[0][0], upper=altitude_bounds[0][1],
                    rate_targets=rate_targets, rate2_targets=rate2_targets,
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
            Dynamic.Mission.ELECTRIC_POWER_IN_TOTAL,
            output_name=Dynamic.Mission.ELECTRIC_POWER_IN_TOTAL, units='kW'
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

        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)

        if phase_type is EquationsOfMotion.SOLVED_2DOF:
            phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE)
            phase.add_timeseries_output("alpha")
            phase.add_timeseries_output(
                "fuselage_pitch", output_name="theta", units="deg")
            phase.add_timeseries_output("thrust_req", units="lbf")
            phase.add_timeseries_output("normal_force")
            phase.add_timeseries_output("time")

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
            'throttle_allocation': self.user_options.get_val('throttle_allocation')
        }


FlightPhaseBase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')

FlightPhaseBase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')

FlightPhaseBase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')

FlightPhaseBase._add_meta_data(
    'num_segments', val=5, desc='transcription: number of segments')

FlightPhaseBase._add_meta_data(
    'order', val=3,
    desc='transcription: order of the state transcription; the order of the control'
    ' transcription is `order - 1`')

FlightPhaseBase._add_meta_data('polynomial_control_order', val=3)

FlightPhaseBase._add_meta_data('use_polynomial_control', val=True)

FlightPhaseBase._add_meta_data('ground_roll', val=False)

FlightPhaseBase._add_meta_data('add_initial_mass_constraint', val=False)

FlightPhaseBase._add_meta_data('fix_initial', val=True)

FlightPhaseBase._add_meta_data('fix_duration', val=False)

FlightPhaseBase._add_meta_data('optimize_mach', val=False)

FlightPhaseBase._add_meta_data('optimize_altitude', val=False)

FlightPhaseBase._add_meta_data('initial_bounds', val=(0., 100.), units='s')

FlightPhaseBase._add_meta_data('duration_bounds', val=(0., 100.), units='s')

FlightPhaseBase._add_meta_data(
    'required_available_climb_rate', val=None, units='m/s',
    desc='minimum avaliable climb rate')

FlightPhaseBase._add_meta_data(
    'no_climb', val=False, desc='aircraft is not allowed to climb during phase')

FlightPhaseBase._add_meta_data(
    'no_descent', val=False, desc='aircraft is not allowed to descend during phase')

FlightPhaseBase._add_meta_data('constrain_final', val=False)

FlightPhaseBase._add_meta_data('input_initial', val=False)

FlightPhaseBase._add_meta_data('initial_mach', val=None, units='unitless')

FlightPhaseBase._add_meta_data('final_mach', val=None, units='unitless')

FlightPhaseBase._add_meta_data('initial_altitude', val=None, units='ft')

FlightPhaseBase._add_meta_data('final_altitude', val=None, units='ft')

FlightPhaseBase._add_meta_data('throttle_enforcement', val=None)

FlightPhaseBase._add_meta_data('throttle_allocation', val=ThrottleAllocation.FIXED)

FlightPhaseBase._add_meta_data('mach_bounds', val=(0., 2.), units='unitless')

FlightPhaseBase._add_meta_data('altitude_bounds', val=(0., 60.e3), units='ft')

FlightPhaseBase._add_meta_data('solve_for_distance', val=False)

FlightPhaseBase._add_meta_data('constraints', val={})

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('mach'),
    desc='initial guess for speed')

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')
