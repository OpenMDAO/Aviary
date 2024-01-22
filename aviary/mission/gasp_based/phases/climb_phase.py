from aviary.mission.phase_builder_base import (
    PhaseBuilderBase, InitialGuessState, InitialGuessTime, InitialGuessControl)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.variable_info.variable_meta_data import _MetaData


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

        fix_initial = user_options.get_val('fix_initial')
        mach_cruise = user_options.get_val('mach_cruise')
        target_mach = user_options.get_val('target_mach')
        final_alt = user_options.get_val('final_alt', units='ft')
        required_available_climb_rate = user_options.get_val(
            'required_available_climb_rate', units='ft/min')
        time_initial_bounds = user_options.get_val('time_initial_bounds', units='s')
        duration_bounds = user_options.get_val('duration_bounds', units='s')
        duration_ref = user_options.get_val('duration_ref', units='s')
        alt_lower = user_options.get_val('alt_lower', units='ft')
        alt_upper = user_options.get_val('alt_upper', units='ft')
        alt_ref = user_options.get_val('alt_ref', units='ft')
        alt_ref0 = user_options.get_val('alt_ref0', units='ft')
        alt_defect_ref = user_options.get_val('alt_defect_ref', units='ft')
        mass_lower = user_options.get_val('mass_lower', units='lbm')
        mass_upper = user_options.get_val('mass_upper', units='lbm')
        mass_ref = user_options.get_val('mass_ref', units='lbm')
        mass_ref0 = user_options.get_val('mass_ref0', units='lbm')
        mass_defect_ref = user_options.get_val('mass_defect_ref', units='lbm')
        distance_lower = user_options.get_val('distance_lower', units='NM')
        distance_upper = user_options.get_val('distance_upper', units='NM')
        distance_ref = user_options.get_val('distance_ref', units='NM')
        distance_ref0 = user_options.get_val('distance_ref0', units='NM')
        distance_defect_ref = user_options.get_val('distance_defect_ref', units='NM')

        phase.set_time_options(
            fix_initial=fix_initial,
            initial_bounds=time_initial_bounds,
            duration_bounds=duration_bounds,
            duration_ref=duration_ref,
            units="s",
        )

        # States
        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=fix_initial,
            fix_final=False,
            lower=alt_lower,
            upper=alt_upper,
            units="ft",
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
            targets=Dynamic.Mission.ALTITUDE,
            ref=alt_ref,
            ref0=alt_ref0,
            defect_ref=alt_defect_ref,
        )

        phase.add_state(
            Dynamic.Mission.MASS,
            fix_initial=fix_initial,
            fix_final=False,
            lower=mass_lower,
            upper=mass_upper,
            units="lbm",
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            ref=mass_ref,
            ref0=mass_ref0,
            defect_ref=mass_defect_ref,
        )

        phase.add_state(
            Dynamic.Mission.DISTANCE,
            fix_initial=fix_initial,
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units="NM",
            rate_source="distance_rate",
            ref=distance_ref,
            ref0=distance_ref0,
            defect_ref=distance_defect_ref,
        )

        # Boundary Constraints
        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=final_alt,
            units="ft",
            ref=final_alt,
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
        phase.add_timeseries_output("TAS", output_name="TAS", units="kn")
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
ClimbPhase._add_meta_data('fix_initial', val=False)
ClimbPhase._add_meta_data('EAS_target', val=0)
ClimbPhase._add_meta_data('mach_cruise', val=0)
ClimbPhase._add_meta_data('target_mach', val=False)
ClimbPhase._add_meta_data('final_alt', val=0)
ClimbPhase._add_meta_data('required_available_climb_rate', val=None, units='ft/min')
ClimbPhase._add_meta_data('time_initial_bounds', val=(0, 0), units='s')
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
    InitialGuessTime(),
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
