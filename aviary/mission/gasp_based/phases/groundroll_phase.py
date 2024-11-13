from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl
from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class GroundrollPhase(PhaseBuilderBase):
    """
    A phase builder for a groundroll phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the groundroll phase of a 2-degree of freedom flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the groundroll phase are included.
    """

    default_name = 'groundroll_phase'
    default_ode_class = GroundrollODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = self.phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        # Add the necessary get_val calls for each parameter, e.g.,
        fix_initial = user_options.get_val('fix_initial')
        fix_initial_mass = user_options.get_val('fix_initial_mass')
        connect_initial_mass = user_options.get_val('connect_initial_mass')
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

        # Add states
        self.add_velocity_state(user_options)

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

        phase.add_timeseries_output("time", units="s", output_name="time")
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
GroundrollPhase._add_meta_data(
    'analytic', val=False, desc='this is an analytic phase (no states).')
GroundrollPhase._add_meta_data('fix_initial', val=True)
GroundrollPhase._add_meta_data('fix_initial_mass', val=False)
GroundrollPhase._add_meta_data('connect_initial_mass', val=True)
GroundrollPhase._add_meta_data('duration_bounds', val=(1, 100), units='s')
GroundrollPhase._add_meta_data('duration_ref', val=1, units='s')
GroundrollPhase._add_meta_data('velocity_lower', val=0, units='kn')
GroundrollPhase._add_meta_data('velocity_upper', val=1000, units='kn')
GroundrollPhase._add_meta_data('velocity_ref', val=100, units='kn')
GroundrollPhase._add_meta_data('velocity_ref0', val=0, units='kn')
GroundrollPhase._add_meta_data('velocity_defect_ref', val=None, units='kn')
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
    InitialGuessIntegrationVariable(),
    desc='initial guess for time options')
GroundrollPhase._add_initial_guess_meta_data(
    InitialGuessState('velocity'),
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
