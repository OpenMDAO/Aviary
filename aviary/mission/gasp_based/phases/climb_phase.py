from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessIntegrationVariable, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.climb_ode import ClimbODE


class ClimbPhase(PhaseBuilderBase):
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
    default_name = 'climb_phase'
    default_ode_class = ClimbODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

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
        phase = self.phase = super().build_phase(aviary_options)

        # Custom configurations for the climb phase
        user_options = self.user_options

        mach_cruise = user_options.get_val('mach_cruise')
        target_mach = user_options.get_val('target_mach')
        final_altitude = user_options.get_val('final_altitude', units='ft')
        required_available_climb_rate = user_options.get_val(
            'required_available_climb_rate', units='ft/min')

        # States
        self.add_altitude_state(user_options)

        self.add_mass_state(user_options)

        self.add_distance_state(user_options)

        # Boundary Constraints
        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=final_altitude,
            units="ft",
            ref=final_altitude,
        )

        if required_available_climb_rate is not None:
            # TODO: this should be altitude rate max
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                loc="final",
                lower=required_available_climb_rate,
                units="ft/min",
                ref=1,
            )

        if target_mach:
            phase.add_boundary_constraint(
                Dynamic.Mission.MACH, loc="final", equals=mach_cruise,
            )

        # Timeseries Outputs
        phase.add_timeseries_output(
            Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
        phase.add_timeseries_output("EAS", output_name="EAS", units="kn")
        phase.add_timeseries_output(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units="lbm/s")
        phase.add_timeseries_output("theta", output_name="theta", units="deg")
        phase.add_timeseries_output("alpha", output_name="alpha", units="deg")
        phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                    output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
        phase.add_timeseries_output(
            "TAS_violation", output_name="TAS_violation", units="kn")
        phase.add_timeseries_output(
            Dynamic.Mission.VELOCITY, output_name=Dynamic.Mission.VELOCITY, units="kn"
        )
        phase.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
        phase.add_timeseries_output(
            Dynamic.Mission.THRUST_TOTAL, output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

        return phase

    def _extra_ode_init_kwargs(self):
        """
        Return extra kwargs required for initializing the ODE.
        """
        # TODO: support external_subsystems and meta_data in the base class
        return {
            'EAS_target': self.user_options.get_val('EAS_target', units='kn'),
            'mach_cruise': self.user_options.get_val('mach_cruise'),
        }


# Adding metadata for the ClimbPhase
ClimbPhase._add_meta_data(
    'analytic', val=False, desc='this is an analytic phase (no states).')
ClimbPhase._add_meta_data(
    'reserve', val=False, desc='this phase is part of the reserve mission.')
ClimbPhase._add_meta_data(
    'target_distance', val={}, desc='the amount of distance traveled in this phase added as a constraint')
ClimbPhase._add_meta_data(
    'target_duration', val={}, desc='the amount of time taken by this phase added as a constraint')
ClimbPhase._add_meta_data('fix_initial', val=False)
ClimbPhase._add_meta_data('EAS_target', val=0)
ClimbPhase._add_meta_data('mach_cruise', val=0)
ClimbPhase._add_meta_data('target_mach', val=False)
ClimbPhase._add_meta_data('final_altitude', val=0)
ClimbPhase._add_meta_data('required_available_climb_rate', val=None, units='ft/min')
ClimbPhase._add_meta_data('duration_bounds', val=(0, 0), units='s')
ClimbPhase._add_meta_data('duration_ref', val=1, units='s')
ClimbPhase._add_meta_data('alt_lower', val=0, units='ft')
ClimbPhase._add_meta_data('alt_upper', val=0, units='ft')
ClimbPhase._add_meta_data('alt_ref', val=1, units='ft')
ClimbPhase._add_meta_data('alt_ref0', val=0, units='ft')
ClimbPhase._add_meta_data('alt_defect_ref', val=None, units='ft')
ClimbPhase._add_meta_data('mass_lower', val=0, units='lbm')
ClimbPhase._add_meta_data('mass_upper', val=0, units='lbm')
ClimbPhase._add_meta_data('mass_ref', val=1, units='lbm')
ClimbPhase._add_meta_data('mass_ref0', val=0, units='lbm')
ClimbPhase._add_meta_data('mass_defect_ref', val=None, units='lbm')
ClimbPhase._add_meta_data('distance_lower', val=0, units='NM')
ClimbPhase._add_meta_data('distance_upper', val=0, units='NM')
ClimbPhase._add_meta_data('distance_ref', val=1, units='NM')
ClimbPhase._add_meta_data('distance_ref0', val=0, units='NM')
ClimbPhase._add_meta_data('distance_defect_ref', val=None, units='NM')
ClimbPhase._add_meta_data('num_segments', val=None, units='unitless')
ClimbPhase._add_meta_data('order', val=None, units='unitless')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for horizontal distance traveled')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for vertical distances')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

ClimbPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
