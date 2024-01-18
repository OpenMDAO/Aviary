from aviary.mission.flops_based.phases.phase_builder_base import (
    PhaseBuilderBase, InitialGuessState, InitialGuessTime)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution
from aviary.variable_info.variable_meta_data import _MetaData


class CruisePhase(PhaseBuilderBase):
    """
    A phase builder for a climb phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the climb phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the climb phase are included.
    """
    default_name = 'cruise_phase'
    default_ode_class = BreguetCruiseODESolution

    __slots__ = ('external_subsystems', 'meta_data')

    # region : derived type customization points
    _meta_data_ = {}

    _initial_guesses_meta_data_ = {}

    default_meta_data = _MetaData

    def __init__(
        self, name=None, subsystem_options=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, core_subsystems=None,
        external_subsystems=None, meta_data=None
    ):
        super().__init__(
            name=name, subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses, ode_class=ode_class, transcription=transcription,
            core_subsystems=core_subsystems,
        )

        # TODO: support external_subsystems and meta_data in the base class
        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data
        self.is_analytic_phase = True

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new climb phase for analysis using these constraints.

        If ode_class is None, ClimbODE is used as the default.

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

        # Time here is really the independent variable through which we are integrating.
        # In the case of the Breguet Range ODE, it's mass.
        # We rely on mass being monotonically non-increasing across the phase.
        phase.set_time_options(
            name='mass',
            fix_initial=False,
            fix_duration=False,
            units="lbm",
            targets="mass",
            initial_bounds=(10.e3, 500_000),
            initial_ref=100_000,
            duration_bounds=(-50000, -10),
            duration_ref=50000,
        )

        phase.add_parameter(Dynamic.Mission.ALTITUDE, opt=False,
                            val=alt_cruise, units=alt_units)
        phase.add_parameter(Dynamic.Mission.MACH, opt=False,
                            val=mach_cruise)
        phase.add_parameter("initial_distance", opt=False, val=0.0,
                            units="NM", static_target=True)
        phase.add_parameter("initial_time", opt=False, val=0.0,
                            units="s", static_target=True)

        phase.add_timeseries_output("time", units="s")

        return phase


# Adding metadata for the CruisePhase
CruisePhase._add_meta_data('alt_cruise', val=0)
CruisePhase._add_meta_data('mach_cruise', val=0)

CruisePhase._add_initial_guess_meta_data(
    InitialGuessTime(),
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
