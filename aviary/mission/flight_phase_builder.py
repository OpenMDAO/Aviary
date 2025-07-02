import dymos as dm
import numpy as np

from aviary.mission.flops_based.ode.energy_ODE import EnergyODE
from aviary.mission.flops_based.phases.phase_utils import (
    add_subsystem_variables_to_phase,
    get_initial,
)
from aviary.mission.initial_guess_builders import (
    InitialGuessState,
    InitialGuessControl,
)
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

        # TODO: These defaults aren't great, but need to keep things the same for now.
        defaults = {
            'mass_ref': 1e4,
            'mass_defect_ref': 1e6,
            'mass_bounds': (0.0, None),
        }
        self.add_state_options('mass', units='kg', defaults=defaults)

        # TODO: These defaults aren't great, but need to keep things the same for now.
        defaults = {
            'distance_ref': 1e6,
            'distance_defect_ref': 1e8,
            'distance_bounds': (0.0, None),
        }
        self.add_state_options('distance', units='m', defaults=defaults)

        self.add_control_options('altitude', units='ft')

        # TODO: These defaults aren't great, but need to keep things the same for now.
        defaults = {
            'mach_ref': 0.5,
        }
        self.add_control_options('mach', units='unitless', defaults=defaults)

        self.add_time_options(units='s')

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
            name='required_available_climb_rate',
            default=None,
            units='ft/s',
            desc='Adds a constraint requiring Dynamic.Mission.ALTITUDE_RATE_MAX to be no '
            'smaller than required_available_climb_rate. This helps to ensure that the '
            'propulsion system is large enough to handle emergency maneuvers at all points '
            'throughout the flight envelope. Default value is None for no constraint.',
        )

        self.declare(
            name='ground_roll',
            types=bool,
            default=False,
            desc='Set to True only for phases where the aircraft is rolling on the ground. '
            'All other phases of flight (climb, cruise, descent) this must be set to False.',
        )

        self.declare(
            name='constraints',
            types=dict,
            default={},
            desc="Add in custom constraints i.e. 'flight_path_angle': {'equals': -3., "
            "'loc': 'initial', 'units': 'deg', 'type': 'boundary',}. For more details see "
            '_add_user_defined_constraints().',
        )

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


@register
class FlightPhaseBase(PhaseBuilderBase):
    """
    The base class for flight phase.

    This houses parts of the build_phase process that are commmon to EnergyPhase and TwoDOFPhase.
    """

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
        self.phase = phase

        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))

        user_options = self.user_options

        throttle_enforcement = user_options['throttle_enforcement']
        no_descent = user_options['no_descent']
        no_climb = user_options['no_climb']
        constraints = user_options['constraints']
        ground_roll = user_options['ground_roll']

        ##############
        # Add States #
        ##############
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_source = Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL
        else:
            rate_source = 'dmass_dr'

        self.add_state('mass', Dynamic.Vehicle.MASS, rate_source)

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            self.add_state('distance', Dynamic.Mission.DISTANCE, Dynamic.Mission.DISTANCE_RATE)

        phase = add_subsystem_variables_to_phase(phase, self.name, self.external_subsystems)

        ################
        # Add Controls #
        ################
        if phase_type is EquationsOfMotion.HEIGHT_ENERGY:
            rate_targets = [Dynamic.Atmosphere.MACH_RATE]
        else:
            rate_targets = ['dmach_dr']

        self.add_control(
            'mach',
            Dynamic.Atmosphere.MACH,
            rate_targets,
            add_constraints=Dynamic.Atmosphere.MACH not in constraints,
        )

        if phase_type is EquationsOfMotion.HEIGHT_ENERGY and not ground_roll:
            rate_targets = [Dynamic.Mission.ALTITUDE_RATE]
            rate2_targets = None
        else:
            rate_targets = ['dh_dr']
            rate2_targets = ['d2h_dr2']

        self.add_control(
            'altitude',
            Dynamic.Mission.ALTITUDE,
            rate_targets,
            rate2_targets=rate2_targets,
            add_constraints=Dynamic.Mission.ALTITUDE not in constraints,
        )

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
                phase.add_parameter(
                    'throttle_allocations',
                    units='unitless',
                    val=val,
                    shape=(num_engine_type - 1,),
                    opt=allocation == ThrottleAllocation.STATIC,
                    lower=0.0,
                    upper=1.0,
                )

        ##################
        # Add Timeseries #
        ##################
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
    InitialGuessControl('altitude'), desc='initial guess for vertical distances'
)

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessControl('mach'), desc='initial guess for speed'
)

FlightPhaseBase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass'
)
