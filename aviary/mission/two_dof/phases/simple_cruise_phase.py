import numpy as np

from aviary.mission.initial_guess_builders import InitialGuessIntegrationVariable, InitialGuessState
from aviary.mission.phase_builder import PhaseBuilder
from aviary.mission.two_dof.ode.simple_cruise_ode import SimpleCruiseODE
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import ThrottleAllocation
from aviary.variable_info.variables import Aircraft, Dynamic


class SimpleCruisePhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(
            name='num_segments',
            types=int,
            default=5,
            desc='The number of segments in transcription creation in Dymos. '
            'While this phase is usually an analytic phase, this option is '
            'needed if an external subsystem requires a dynamic transcription.',
        )

        self.declare(
            name='order',
            types=int,
            default=3,
            desc='The order of polynomials for interpolation in the transcription '
            'created in Dymos. While this phase is usually an analytic phase, this option is '
            'needed if an external subsystem requires a dynamic transcription.',
        )

        defaults = {
            'mass_bounds': (0.0, None),
            'mass_direct_link': False,
        }
        self.add_state_options('mass', units='lbm', defaults=defaults)

        defaults = {
            'time_duration_bounds': (0, 3600),
            'time_initial_bounds': (0.0, 100.0),
            'initial_time_direct_link': True,
        }
        self.add_time_options(units='s', defaults=defaults)

        self.declare(name='alt_cruise', default=0.0, units='ft', desc='Cruise altitude.')

        self.declare(name='mach_cruise', default=0.0, desc='Cruise Mach number.')

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
            name='altitude_direct_link',
            default=False,
            types=bool,
            desc='When True, directly link the initial altitude parameter to the previous '
            'phase. When False, use a constraint.',
        )

        self.declare(
            name='distance_direct_link',
            default=False,
            types=bool,
            desc='When True, directly link the initial distance parameter to the previous '
            'phase. When False, use a constraint.',
        )

        self.declare(
            name='mach_direct_link',
            default=True,
            types=bool,
            desc='When True, directly link the initial mach parameter to the previous '
            'phase. When False, use a constraint.',
        )

        self.declare(
            name='throttle_enforcement',
            default='path_constraint',
            values=['path_constraint', 'boundary_constraint', 'bounded', 'control', None],
            desc='Flag to enforce engine throttle bounds as path constraints, boundary '
            'constraints, solver bounds. You can also select "control" to turn throttle into a '
            'control, which allows you to assign a value or let the optimizer choose it.',
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


class SimpleCruisePhase(PhaseBuilder):
    """
    A phase builder for a cruise phase in a mission simulation.

    This class extends the PhaseBuilder class, providing specific implementations for
    the cruise phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilder.

    Methods
    -------
    Inherits all methods from PhaseBuilder.
    Additional method overrides and new methods specific to the cruise phase are included.
    """

    default_name = 'simple_cruise_phase'
    default_ode_class = SimpleCruiseODE
    default_options_class = SimpleCruisePhaseOptions

    _initial_guesses_meta_data_ = {}

    def __init__(
        self,
        name=None,
        subsystem_options=None,
        user_options=None,
        initial_guesses=None,
        ode_class=None,
        transcription=None,
        subsystems=None,
        meta_data=None,
    ):
        super().__init__(
            name=name,
            subsystem_options=subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
            ode_class=ode_class,
            transcription=transcription,
            subsystems=subsystems,
            meta_data=meta_data,
        )

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new cruise phase for analysis using these constraints.

        If ode_class is None, SimpleCruiseODE is used as the default.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase = self.phase = super().build_phase(aviary_options)

        # Custom configurations for the climb phase
        user_options = self.user_options

        # Add states
        self.add_state(
            'mass',
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
        )

        # These are constant.
        mach_cruise = user_options.get_val('mach_cruise')
        alt_cruise, alt_units = user_options['alt_cruise']

        phase = self.add_subsystem_variables_to_phase(phase, aviary_options)

        phase.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, val=alt_cruise, units=alt_units)
        phase.add_parameter(Dynamic.Atmosphere.MACH, opt=False, val=mach_cruise)
        phase.add_parameter(
            'initial_distance',
            opt=True,
            val=0.0,
            units='nmi',
            static_target=True,
        )

        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))
        if num_engine_type > 1:
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
                opt = allocation == ThrottleAllocation.STATIC
                kwargs = {}
                if opt:
                    kwargs['lower'] = 0.0
                    kwargs['upper'] = 1.0

                phase.add_parameter(
                    'throttle_allocations',
                    units='unitless',
                    val=val,
                    shape=(num_engine_type - 1,),
                    opt=opt,
                    **kwargs,
                )

        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE, units=alt_units)
        phase.add_timeseries_output(Dynamic.Vehicle.ANGLE_OF_ATTACK, units='deg')
        phase.add_timeseries_output(Dynamic.Mission.DISTANCE, units='nmi')
        phase.add_timeseries_output(Dynamic.Vehicle.DRAG, units='lbf')
        phase.add_timeseries_output('EAS', units='kn')
        phase.add_timeseries_output(
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/s'
        )
        phase.add_timeseries_output(Dynamic.Vehicle.LIFT, units='lbf')
        phase.add_timeseries_output(Dynamic.Atmosphere.MACH, units='unitless')
        phase.add_timeseries_output(Dynamic.Vehicle.MASS, units='lbm')
        phase.add_timeseries_output(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        phase.add_timeseries_output(Dynamic.Mission.VELOCITY, units='kn')

        if user_options['throttle_enforcement'] != 'control':
            phase.add_timeseries_output(Dynamic.Vehicle.Propulsion.THROTTLE, units='unitless')

        return phase

    def get_linked_variables(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        linked_vars = [
            'time',
            'initial_distance',
            Dynamic.Atmosphere.MACH,
            Dynamic.Mission.ALTITUDE,
            Dynamic.Vehicle.MASS,
        ]
        return linked_vars

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {
            'throttle_enforcement': self.user_options['throttle_enforcement'],
            'throttle_allocation': self.user_options['throttle_allocation'],
        }


SimpleCruisePhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple',
)

SimpleCruisePhase._add_initial_guess_meta_data(
    InitialGuessState('mass'), desc='initial guess for mass'
)

SimpleCruisePhase._add_initial_guess_meta_data(
    InitialGuessState('initial_distance'), desc='initial guess for initial_distance'
)

SimpleCruisePhase._add_initial_guess_meta_data(
    InitialGuessState('initial_time'), desc='initial guess for initial_time'
)
