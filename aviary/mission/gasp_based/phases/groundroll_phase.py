from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessTime, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.variable_info.variable_meta_data import _MetaData


class GroundrollPhase(PhaseBuilderBase):
    default_name = 'groundroll_phase'
    default_ode_class = GroundrollODE

    __slots__ = ('external_subsystems', 'meta_data')

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

        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        # Add the necessary get_val calls for each parameter, e.g.,
        fix_initial = user_options.get_val('fix_initial')
        fix_initial_mass = user_options.get_val('fix_initial_mass')
        connect_initial_mass = user_options.get_val('connect_initial_mass')
        duration_bounds = user_options.get_val('duration_bounds', units='s')
        duration_ref = user_options.get_val('duration_ref', units='s')
        TAS_lower = user_options.get_val('TAS_lower', units='kn')
        TAS_upper = user_options.get_val('TAS_upper', units='kn')
        TAS_ref = user_options.get_val('TAS_ref', units='kn')
        TAS_ref0 = user_options.get_val('TAS_ref0', units='kn')
        TAS_defect_ref = user_options.get_val('TAS_defect_ref', units='kn')
        mass_lower = user_options.get_val('mass_lower', units='lbm')
        mass_upper = user_options.get_val('mass_upper', units='lbm')
        mass_ref = user_options.get_val('mass_ref', units='lbm')
        mass_ref0 = user_options.get_val('mass_ref0', units='lbm')
        mass_defect_ref = user_options.get_val('mass_defect_ref', units='lbm')
        distance_lower = user_options.get_val('distance_lower', units='ft')
        distance_upper = user_options.get_val('distance_upper', units='ft')
        distance_ref = user_options.get_val('distance_ref', units='ft')
        distance_ref0 = user_options.get_val('distance_ref0', units='ft')
        distance_defect_ref = user_options.get_val('distance_defect_ref', units='ft')

        # Set time options
        phase.set_time_options(
            fix_initial=fix_initial,
            fix_duration=False,
            units="s",
            targets="t_curr",
            duration_bounds=duration_bounds,
            duration_ref=duration_ref,
        )

        # Add states
        phase.add_state(
            "TAS",
            fix_initial=fix_initial,
            fix_final=False,
            lower=TAS_lower,
            upper=TAS_upper,
            units="kn",
            rate_source="TAS_rate",
            ref=TAS_ref,
            defect_ref=TAS_defect_ref,
            ref0=TAS_ref0,
        )

        phase.add_state(
            Dynamic.Mission.MASS,
            fix_initial=fix_initial_mass,
            input_initial=connect_initial_mass,
            fix_final=False,
            lower=mass_lower,
            upper=mass_upper,
            units="lbm",
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            ref=mass_ref,
            defect_ref=mass_defect_ref,
            ref0=mass_ref0,
        )

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=fix_initial,
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units="ft",
            rate_source="distance_rate",
            ref=distance_ref,
            defect_ref=distance_defect_ref,
            ref0=distance_ref0,
        )

        phase.add_parameter("t_init_gear", units="s",
                            static_target=True, opt=False, val=100)
        phase.add_parameter("t_init_flaps", units="s",
                            static_target=True, opt=False, val=100)

        # boundary/path constraints + controls
        # the final TAS is constrained externally to define the transition to the rotation
        # phase

        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")

        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")

        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output("CL")
        phase.add_timeseries_output("CD")
        phase.add_timeseries_output("fuselage_pitch", output_name="theta", units="deg")

        return phase


# Adding metadata for the GroundrollPhase
GroundrollPhase._add_meta_data('fix_initial', val=True)
GroundrollPhase._add_meta_data('fix_initial_mass', val=False)
GroundrollPhase._add_meta_data('connect_initial_mass', val=True)
GroundrollPhase._add_meta_data('duration_bounds', val=(1, 100), units='s')
GroundrollPhase._add_meta_data('duration_ref', val=1, units='s')
GroundrollPhase._add_meta_data('TAS_lower', val=0, units='kn')
GroundrollPhase._add_meta_data('TAS_upper', val=1000, units='kn')
GroundrollPhase._add_meta_data('TAS_ref', val=100, units='kn')
GroundrollPhase._add_meta_data('TAS_ref0', val=0, units='kn')
GroundrollPhase._add_meta_data('TAS_defect_ref', val=None, units='kn')
GroundrollPhase._add_meta_data('mass_lower', val=0, units='lbm')
GroundrollPhase._add_meta_data('mass_upper', val=200_000, units='lbm')
GroundrollPhase._add_meta_data('mass_ref', val=100_000, units='lbm')
GroundrollPhase._add_meta_data('mass_ref0', val=0, units='lbm')
GroundrollPhase._add_meta_data('mass_defect_ref', val=100, units='lbm')
GroundrollPhase._add_meta_data('distance_lower', val=0, units='ft')
GroundrollPhase._add_meta_data('distance_upper', val=4000, units='ft')
GroundrollPhase._add_meta_data('distance_ref', val=3000, units='ft')
GroundrollPhase._add_meta_data('distance_ref0', val=0, units='ft')
GroundrollPhase._add_meta_data('distance_defect_ref', val=3000, units='ft')
GroundrollPhase._add_meta_data('t_init_gear', val=100, units='s')
GroundrollPhase._add_meta_data('t_init_flaps', val=100, units='s')
GroundrollPhase._add_meta_data('num_segments', val=None, units='unitless')
GroundrollPhase._add_meta_data('order', val=None, units='unitless')

# Adding initial guess metadata
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for time options')
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('TAS'),
    desc='initial guess for true airspeed state')
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass state')
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for distance state')
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
