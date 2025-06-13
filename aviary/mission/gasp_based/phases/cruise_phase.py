from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution
from aviary.mission.initial_guess_builders import InitialGuessIntegrationVariable, InitialGuessState
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_options_dict import AviaryOptionsDictionary
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class CruisePhaseOptions(AviaryOptionsDictionary):
    def declare_options(self):
        self.declare(name='alt_cruise', default=0.0, units='ft', desc='Cruise altitude.')

        self.declare(name='mach_cruise', default=0.0, desc='Cruise Mach number.')

        self.declare(
            'analytic',
            types=bool,
            default=False,
            desc='When set to True, this is an analytic phase.',
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
            'time_duration',
            default=None,
            units='s',
            desc='The amount of time taken by this phase added as a constraint.',
        )

        self.declare(
            name='time_duration_bounds',
            default=(0, 3600),
            units='s',
            desc='Lower and upper bounds on the phase duration, in the form of a nested tuple: '
            'i.e. ((20, 36), "min") This constrains the duration to be between 20 and 36 min.',
        )

        self.declare(
            name='fix_duration',
            types=bool,
            default=False,
            desc='If True, the time duration of the phase is not treated as a design '
            'variable for the optimization problem.',
        )

        self.declare(
            'time_initial_bounds',
            types=tuple,
            default=(0.0, 100.0),
            units='s',
            desc='Lower and upper bounds on the starting time for this phase relative to the '
            'starting time of the mission, i.e., ((25, 45), "min") constrians this phase to '
            'start between 25 and 45 minutes after the start of the mission.',
        )


class CruisePhase(PhaseBuilderBase):
    """
    A phase builder for a climb phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the cruise phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the cruise phase are included.
    """

    default_name = 'cruise_phase'
    default_ode_class = BreguetCruiseODESolution
    default_options_class = CruisePhaseOptions

    _initial_guesses_meta_data_ = {}

    def __init__(
        self,
        name=None,
        subsystem_options=None,
        user_options=None,
        initial_guesses=None,
        ode_class=None,
        transcription=None,
        core_subsystems=None,
        external_subsystems=None,
        meta_data=None,
    ):
        super().__init__(
            name=name,
            subsystem_options=subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
            ode_class=ode_class,
            transcription=transcription,
            core_subsystems=core_subsystems,
            external_subsystems=external_subsystems,
            meta_data=meta_data,
            is_analytic_phase=True,
        )

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new cruise phase for analysis using these constraints.

        If ode_class is None, BreguetCruiseODESolution is used as the default.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase = super().build_phase(aviary_options)

        # Custom configurations for the climb phase
        user_options = self.user_options

        mach_cruise = user_options.get_val('mach_cruise')
        alt_cruise, alt_units = user_options['alt_cruise']

        phase.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, val=alt_cruise, units=alt_units)
        phase.add_parameter(Dynamic.Atmosphere.MACH, opt=False, val=mach_cruise)
        phase.add_parameter('initial_distance', opt=False, val=0.0, units='NM', static_target=True)
        phase.add_parameter('initial_time', opt=False, val=0.0, units='s', static_target=True)

        phase.add_timeseries_output('time', units='s', output_name='time')
        phase.add_timeseries_output(Dynamic.Vehicle.MASS, units='lbm')
        phase.add_timeseries_output(Dynamic.Mission.DISTANCE, units='nmi')

        return phase


CruisePhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple',
)

CruisePhase._add_initial_guess_meta_data(InitialGuessState('mass'), desc='initial guess for mass')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('initial_distance'), desc='initial guess for initial_distance'
)

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('initial_time'), desc='initial guess for initial_time'
)

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'), desc='initial guess for altitude'
)

CruisePhase._add_initial_guess_meta_data(InitialGuessState('mach'), desc='initial guess for mach')
