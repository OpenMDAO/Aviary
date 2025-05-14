import dymos as dm
import numpy as np

from aviary.mission.flops_based.ode.energy_ODE import EnergyODE
from aviary.mission.flops_based.phases.phase_utils import (
    add_subsystem_variables_to_phase,
    get_initial,
)
from aviary.mission.initial_guess_builders import InitialGuessState
from aviary.mission.phase_builder_base import PhaseBuilderBase, register
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import EquationsOfMotion, ThrottleAllocation
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic

# Height Energy and Solved2DOF use this builder

# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point


class FlightPhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            'reserve',
            types=bool,
            default=False,
            desc='Designate this phase as a reserve phase and contributes its fuel burn '
            'towards the reserve mission fuel requirements. Reserve phases should be '
            'be placed after all non-reserve phases in the phase_info.',
        )

        self.declare(
            name='target_distance',
            default=None,
            units='m',
            desc='The total distance traveled by the aircraft from takeoff to landing '
            'for the primary mission, not including reserve missions. This value must '
            'be positive.',
        )

        self.declare(
            'target_duration',
            types=tuple,
            default=None,
            units='s',
            desc='The amount of time taken by this phase added as a constraint.',
        )

        self.declare(
            name='num_segments',
            types=int,
            default=1,
            desc='The number of segments in transcription creation in Dymos. '
            'The default value is 1.',
        )

        self.declare(
            name='order',
            types=int,
            default=3,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos. The default value is 3.',
        )

        self.declare(
            name='polynomial_control_order',
            types=int,
            default=3,
            desc='The order of the polynomial fit to control values. '
            'Only used if polynomial_control = True',
        )

        self.declare(
            name='use_polynomial_control',
            types=bool,
            default=True,
            desc='Set fo True to use polynomial controls in this phase, which smooths the '
            'control inputs.',
        )

        self.declare(
            name='ground_roll',
            types=bool,
            default=False,
            desc='Set to True only for phases where the aircraft is rolling on the ground. '
            'All other phases of flight (climb, cruise, descent) this must be set to False.',
        )

        self.declare(
            name='add_initial_mass_constraint',
            types=bool,
            default=False,
            desc='Use a constraint for mass instead of connected initial mass for this phase. '
            'Overwrites input_initial=True and sets it to False.',
        )

        self.declare(
            name='input_initial',
            types=bool,
            default=False,
            desc='Links all states (mass, distance) to a calculation external to this phase.',
        )

        self.declare(
            name='fix_initial',
            types=bool,
            default=True,
            desc='Fixes the initial states (mass, distance) and does not allow them to '
            'change during the optimization.',
        )

        self.declare(
            name='fix_duration',
            types=bool,
            default=True,
            desc='If True, the time duration of the phase is not treated as a design '
            'variable for the optimization problem.',
        )

        self.declare(
            name='optimize_mach',
            types=bool,
            default=False,
            desc='Adds the Mach number as a design variable controlled by the optimizer.',
        )

        self.declare(
            name='optimize_altitude',
            types=bool,
            default=False,
            desc='Adds the Altitude as a design variable controlled by the optimizer.',
        )

        self.declare(
            'initial_bounds',
            types=tuple,
            default=(None, None),
            units='min',
            desc='Lower and upper bounds on the starting time for this phase relative to the '
            'starting time of the mission, i.e., ((25, 45), "min") constrians this phase to '
            'start between 25 and 45 minutes after the start of the mission.',
        )

        self.declare(
            name='duration_bounds',
            types=tuple,
            default=(None, None),
            units='min',
            desc='Lower and upper bounds on the phase duration, in the form of a nested tuple: '
            'i.e. ((20, 36), "min") This constrains the duration to be between 20 and 36 min.',
        )

        self.declare(
            name='required_available_climb_rate',
            default=None,
            units='ft/s',
            desc='Adds a constraint requiring Dynamic.Mission.ALTITUDE_RATE_MAX to be no '
            'smaller than required_available_climb_rate. This helps to ensure that the '
            'propulsion system is large enough to handle emergency maneuvers at all points '
            'throughout the flight envelope. Default value is None for no constraint.',
        )

        self.declare(
            name='no_climb',
            types=bool,
            default=False,
            desc='Set to True to prevent the aircraft from climbing during the phase. This option '
            'can be used to prevent unexpected climb during a descent phase.',
        )

        self.declare(
            name='no_descent',
            types=bool,
            default=False,
            desc='Set to True to prevent the aircraft from descending during the phase. This '
            'can be used to prevent unexpected descent during a climb phase.',
        )

        self.declare(
            name='constrain_final',
            types=bool,
            default=False,
            desc='Fixes the final states (mach and altitude) to the values of final_altitude '
            'and final_mach. These values will be unable to change during the optimization.',
        )

        self.declare(
            name='initial_mach',
            types=float,
            allow_none=True,
            default=None,
            desc='The initial Mach number at the start of the phase. This option is only valid '
            'when fix_initial is True.',
        )

        self.declare(
            name='final_mach',
            types=float,
            allow_none=True,
            default=None,
            desc='The final Mach number at the end of the phase. This option is only valid '
            'when fix_initial is True.',
        )

        self.declare(
            name='initial_altitude',
            types=tuple,
            default=None,
            units='ft',
            desc='The initial altitude at the start of the phase. This option is only valid '
            'when fix_initial is True.',
        )

        self.declare(
            name='final_altitude',
            types=tuple,
            default=None,
            units='ft',
            desc='The final altitude at the end of the phase. This option is only valid '
            'when fix_initial is True.',
        )

        self.declare(
            name='throttle_enforcement',
            default='path_constraint',
            values=['path_constraint', 'boundary_constraint', 'bounded', None],
            desc='Flag to enforce engine throttle constraints on the path or at the segment '
            'boundaries or using solver bounds.',
        )

        self.declare(
            name='throttle_allocation',
            default=ThrottleAllocation.FIXED,
            values=[
                ThrottleAllocation.FIXED,
                ThrottleAllocation.STATIC,
                ThrottleAllocation.DYNAMIC,
            ],
            desc='Specifies how to handle the throttles for multiple engines. FIXED is a '
            'user-specified value. STATIC is specified by the optimizer as one value for the '
            'whole phase. DYNAMIC is specified by the optimizer at each point in the phase.',
        )

        self.declare(
            name='mach_bounds',
            types=tuple,
            default=(None, None),
            desc='The lower and upper constraints on mach during this phase i.e., '
            '((0.18, 0.74), "unitless"). The optimizer is never allowed choose mach values '
            'outside of these bounds constraints.',
        )

        self.declare(
            name='altitude_bounds',
            types=tuple,
            default=(None, None),
            units='ft',
            desc='The lower and upper constraints on altitude during this phase i.e., '
            '((0.0, 34000.0), "ft"). The optimizer is never allowed choose mach values '
            'outside of these bounds constraints.',
        )

        self.declare(
            name='solve_for_distance',
            types=bool,
            default=False,
            desc='if True, use a nonlinear solver to converge the distance state variable to '
            'the desired value. Otherwise uses the optimizer to converge the distance state.',
        )

        self.declare(
            name='constraints',
            types=dict,
            default={},
            desc="Add in custom constraints i.e. 'flight_path_angle': {'equals': -3., "
            "'loc': 'initial', 'units': 'deg', 'type': 'boundary',}. For more details see "
            '_add_user_defined_constraints().',
        )


@register
class FlightPhaseBase(PhaseBuilderBase):
    """The base class for flight phase."""

    __slots__ = ('external_subsystems', 'meta_data')

    _initial_guesses_meta_data_ = {}
    default_name = 'cruise'
    default_ode_class = EnergyODE
    default_options_class = FlightPhaseOptions

    default_meta_data = _MetaData

    def build_phase(
        self,
        aviary_options: AviaryValues = None,
        phase_type=EquationsOfMotion.HEIGHT_ENERGY,
    ):
        """
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
        """
        phase: dm.Phase = super().build_phase(aviary_options)

        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))

        user_options = self.user_options

        fix_initial = user_options['fix_initial']
        constrain_final = user_options['constrain_final']
        optimize_mach = user_options['optimize_mach']
        optimize_altitude = user_options['optimize_altitude']
        input_initial = user_options['input_initial']
        polynomial_control_order = user_options['polynomial_control_order']
        use_polynomial_control = user_options['use_polynomial_control']
        throttle_enforcement = user_options['throttle_enforcement']
        mach_bounds = user_options['mach_bounds']
        altitude_bounds = user_options['altitude_bounds']
        initial_mach = user_options['initial_mach']
        final_mach = user_options['final_mach']
        initial_altitude = user_options['initial_altitude'][0]
        final_altitude = user_options['final_altitude'][0]
        solve_for_distance = user_options['solve_for_distance']
        no_descent = user_options['no_descent']
        no_climb = user_options['no_climb']
        constraints = user_options['constraints']

        ##############
        # Add States #
        ##############
        # TODO: critically think about how we should handle fix_initial and input_initial defaults.
        # In keeping with Dymos standards, the default should be False instead of True.
        input_initial_mass = get_initial(input_initial, Dynamic.Vehicle.MASS)
        fix_initial_mass = get_initial(fix_initial, Dynamic.Vehicle.MASS, True)

        # Experiment: use a constraint for mass instead of connected initial.
        # This is due to some problems in mpi.
        # This is needed for the cutting edge full subsystem integration.
        # TODO: when a Dymos fix is in and we've verified that full case works with the fix,
        # remove this argument.
        if user_options['add_initial_mass_constraint']:
            phase.add_constraint('rhs_all.initial_mass_residual', equals=0.0, ref=1e4)
            input_initial_mass = False

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_source = Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL
        else:
            rate_source = 'dmass_dr'

        phase.add_state(
            Dynamic.Vehicle.MASS,
            fix_initial=fix_initial_mass,
            fix_final=False,
            lower=0.0,
            ref=1e4,
            defect_ref=1e6,
            units='kg',
            rate_source=rate_source,
            targets=Dynamic.Vehicle.MASS,
            input_initial=input_initial_mass,
        )

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            input_initial_distance = get_initial(input_initial, Dynamic.Mission.DISTANCE)
            fix_initial_distance = get_initial(fix_initial, Dynamic.Mission.DISTANCE, True)
            phase.add_state(
                Dynamic.Mission.DISTANCE,
                fix_initial=fix_initial_distance,
                fix_final=False,
                lower=0.0,
                ref=1e6,
                defect_ref=1e8,
                units='m',
                rate_source=Dynamic.Mission.DISTANCE_RATE,
                input_initial=input_initial_distance,
                solve_segments='forward' if solve_for_distance else None,
            )

        phase = add_subsystem_variables_to_phase(phase, self.name, self.external_subsystems)

        ################
        # Add Controls #
        ################
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_targets = [Dynamic.Atmosphere.MACH_RATE]
        else:
            rate_targets = ['dmach_dr']

        # dictionary of options for Mach control
        control_dict = {
            'name': Dynamic.Atmosphere.MACH,
            'targets': Dynamic.Atmosphere.MACH,
            # 'units': mach_bounds[1],
            'rate_targets': rate_targets,
            'opt': optimize_mach,
        }

        if optimize_mach:
            control_dict['lower'] = mach_bounds[0]
            control_dict['upper'] = mach_bounds[1]
            control_dict['ref'] = 0.5

        if use_polynomial_control:
            control_dict['control_type'] = 'polynomial'
            control_dict['order'] = polynomial_control_order

        phase.add_control(**control_dict)

        # Add altitude rate as a control
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_targets = [Dynamic.Mission.ALTITUDE_RATE]
            rate2_targets = []
        else:
            rate_targets = ['dh_dr']
            rate2_targets = ['d2h_dr2']

        # For heterogeneous-engine cases, we may have throttle allocation control.
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY and num_engine_type > 1:
            allocation = user_options['throttle_allocation']

            # Allocation should default to an even split so that we don't start
            # with an allocation that might not produce enough thrust.
            val = np.ones(num_engine_type - 1) * (1.0 / num_engine_type)

            if allocation == ThrottleAllocation.DYNAMIC:
                phase.add_control(
                    'throttle_allocations',
                    shape=(num_engine_type - 1,),
                    val=val,
                    targets='throttle_allocations',
                    units='unitless',
                    opt=True,
                    lower=0.0,
                    upper=1.0,
                )

            else:
                if allocation == ThrottleAllocation.STATIC:
                    opt = True
                else:
                    opt = False

                phase.add_parameter(
                    'throttle_allocations',
                    units='unitless',
                    val=val,
                    shape=(num_engine_type - 1,),
                    opt=opt,
                    lower=0.0,
                    upper=1.0,
                )

        # dictionary of options for altitude control
        control_dict = {
            'name': Dynamic.Mission.ALTITUDE,
            'targets': Dynamic.Mission.ALTITUDE,
            'units': altitude_bounds[1],
            'rate_targets': rate_targets,
            'rate2_targets': rate2_targets,
            'opt': optimize_altitude,
        }

        if optimize_altitude:
            control_dict['lower'] = altitude_bounds[0][0]
            control_dict['upper'] = altitude_bounds[0][1]
            control_dict['ref'] = altitude_bounds[0][1]

        if use_polynomial_control:
            control_dict['control_type'] = 'polynomial'
            control_dict['order'] = polynomial_control_order

        # ground_roll uses some hardcoded settings that overwrite user-provided ones
        ground_roll = user_options['ground_roll']
        if ground_roll:
            control_dict['control_type'] = 'polynomial'
            control_dict['order'] = 1
            control_dict['val'] = 0
            control_dict['opt'] = False
            control_dict['fix_initial'] = fix_initial
            control_dict['rate_targets'] = ['dh_dr']
            control_dict['rate2_targets'] = ['d2h_dr2']

        phase.add_control(**control_dict)

        ##################
        # Add Timeseries #
        ##################
        phase.add_timeseries_output(
            Dynamic.Atmosphere.MACH,
            output_name=Dynamic.Atmosphere.MACH,
            units='unitless',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
            units='lbf',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.DRAG, output_name=Dynamic.Vehicle.DRAG, units='lbf'
        )

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            phase.add_timeseries_output(
                Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                output_name=Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                units='m/s',
            )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            units='lbm/h',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
            output_name=Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL,
            units='kW',
        )

        phase.add_timeseries_output(
            Dynamic.Mission.ALTITUDE_RATE,
            output_name=Dynamic.Mission.ALTITUDE_RATE,
            units='ft/s',
        )

        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            output_name=Dynamic.Vehicle.Propulsion.THROTTLE,
            units='unitless',
        )

        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY,
            output_name=Dynamic.Mission.VELOCITY,
            units='m/s',
        )

        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)

        if phase_type is EquationsOfMotion.SOLVED_2DOF:
            phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE)
            phase.add_timeseries_output(Dynamic.Vehicle.ANGLE_OF_ATTACK)
            phase.add_timeseries_output('fuselage_pitch', output_name='theta', units='deg')
            phase.add_timeseries_output('thrust_req', units='lbf')
            phase.add_timeseries_output('normal_force')
            phase.add_timeseries_output('time')

        ###################
        # Add Constraints #
        ###################
        if optimize_mach and fix_initial and Dynamic.Atmosphere.MACH not in constraints:
            phase.add_boundary_constraint(
                Dynamic.Atmosphere.MACH,
                loc='initial',
                equals=initial_mach,
            )

        if optimize_mach and constrain_final and Dynamic.Atmosphere.MACH not in constraints:
            phase.add_boundary_constraint(
                Dynamic.Atmosphere.MACH,
                loc='final',
                equals=final_mach,
            )

        if optimize_altitude and fix_initial and Dynamic.Mission.ALTITUDE not in constraints:
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE,
                loc='initial',
                equals=initial_altitude,
                units=altitude_bounds[1],
                ref=1.0e4,
            )

        if optimize_altitude and constrain_final and Dynamic.Mission.ALTITUDE not in constraints:
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE,
                loc='final',
                equals=final_altitude,
                units=altitude_bounds[1],
                ref=1.0e4,
            )

        if no_descent and Dynamic.Mission.ALTITUDE_RATE not in constraints:
            phase.add_path_constraint(Dynamic.Mission.ALTITUDE_RATE, lower=0.0)

        if no_climb and Dynamic.Mission.ALTITUDE_RATE not in constraints:
            phase.add_path_constraint(Dynamic.Mission.ALTITUDE_RATE, upper=0.0)

        required_available_climb_rate, units = user_options['required_available_climb_rate']

        if (
            required_available_climb_rate is not None
            and Dynamic.Mission.ALTITUDE_RATE_MAX not in constraints
        ):
            phase.add_path_constraint(
                Dynamic.Mission.ALTITUDE_RATE_MAX,
                lower=required_available_climb_rate,
                units=units,
            )

        if Dynamic.Vehicle.Propulsion.THROTTLE not in constraints:
            if throttle_enforcement == 'boundary_constraint':
                phase.add_boundary_constraint(
                    Dynamic.Vehicle.Propulsion.THROTTLE,
                    loc='initial',
                    lower=0.0,
                    upper=1.0,
                    units='unitless',
                )
                phase.add_boundary_constraint(
                    Dynamic.Vehicle.Propulsion.THROTTLE,
                    loc='final',
                    lower=0.0,
                    upper=1.0,
                    units='unitless',
                )
            elif throttle_enforcement == 'path_constraint':
                phase.add_path_constraint(
                    Dynamic.Vehicle.Propulsion.THROTTLE,
                    lower=0.0,
                    upper=1.0,
                    units='unitless',
                )

        self._add_user_defined_constraints(phase, constraints)

        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        user_options = self.user_options

        num_segments = user_options['num_segments']
        order = user_options['order']

        seg_ends, _ = dm.utils.lgl.lgl(num_segments + 1)

        transcription = dm.Radau(
            num_segments=num_segments,
            order=order,
            compressed=True,
            segment_ends=seg_ends,
        )

        return transcription

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'external_subsystems': self.external_subsystems,
            'meta_data': self.meta_data,
            'subsystem_options': self.subsystem_options,
            'throttle_enforcement': self.user_options['throttle_enforcement'],
            'throttle_allocation': self.user_options['throttle_allocation'],
        }


# Previous implementation:


FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for vertical distances'
)

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('mach'), desc='initial guess for speed'
)

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass'
)
