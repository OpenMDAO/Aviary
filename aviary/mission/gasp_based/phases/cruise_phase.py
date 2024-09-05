from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution


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

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def __init__(
        self, name=None, subsystem_options=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, core_subsystems=None,
        external_subsystems=None, meta_data=None
    ):
        super().__init__(
            name=name, subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses, ode_class=ode_class, transcription=transcription,
            core_subsystems=core_subsystems, is_analytic_phase=True,
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
        alt_cruise, alt_units = user_options.get_item('alt_cruise')

        phase.add_parameter(
            Dynamic.Atmosphere.ALTITUDE, opt=False, val=alt_cruise, units=alt_units
        )
        phase.add_parameter(Dynamic.Mission.MACH, opt=False,
                            val=mach_cruise)
        phase.add_parameter("initial_distance", opt=False, val=0.0,
                            units="NM", static_target=True)
        phase.add_parameter("initial_time", opt=False, val=0.0,
                            units="s", static_target=True)

        phase.add_timeseries_output("time", units="s", output_name="time")
        phase.add_timeseries_output(Dynamic.Vehicle.MASS, units="lbm")
        phase.add_timeseries_output(Dynamic.Mission.DISTANCE, units="nmi")

        return phase


# Adding metadata for the CruisePhase
CruisePhase._add_meta_data('alt_cruise', val=0)
CruisePhase._add_meta_data('mach_cruise', val=0)
CruisePhase._add_meta_data(
    'analytic', val=False, desc='this is an analytic phase (no states).')
CruisePhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')
CruisePhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')
CruisePhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')
CruisePhase._add_meta_data('duration_bounds', val=(
    0., 3600.), units='s', desc='duration bounds')
CruisePhase._add_meta_data('fix_duration', val=False)
CruisePhase._add_meta_data('initial_bounds', val=(0., 100.), units='s')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('initial_distance'),
    desc='initial guess for initial_distance')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('initial_time'),
    desc='initial guess for initial_time')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for altitude')

CruisePhase._add_initial_guess_meta_data(
    InitialGuessState('mach'),
    desc='initial guess for mach')
