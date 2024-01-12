import dymos as dm
from aviary.mission.flops_based.phases.phase_builder_base import (
    PhaseBuilderBase, InitialGuessState, InitialGuessTime)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


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

    def __init__(
        self, name=None, subsystem_options=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, core_subsystems=None,
        external_subsystems=None, meta_data=None
    ):
        super().__init__(
            name=name, subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses, ode_class=ode_class, transcription=transcription,
            core_subsystems=core_subsystems, external_subsystems=external_subsystems,
            meta_data=meta_data
        )

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

        # States
        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=user_options.get_val('alt_lower'),
            upper=user_options.get_val('alt_upper'),
            units="ft",
            rate_source=Dynamic.Mission.ALTITUDE_RATE,
            targets=Dynamic.Mission.ALTITUDE,
            ref=user_options.get_val('alt_ref'),
            ref0=user_options.get_val('alt_ref0'),
            defect_ref=user_options.get_val('alt_defect_ref'),
        )

        phase.add_state(
            Dynamic.Mission.MASS,
            fix_initial=user_options.get_val('fix_initial'),
            fix_final=False,
            lower=user_options.get_val('mass_lower'),
            upper=user_options.get_val('mass_upper'),
            units="lbm",
            rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
            targets=Dynamic.Mission.MASS,
            ref=user_options.get_val('mass_ref'),
            ref0=user_options.get_val('mass_ref0'),
            defect_ref=user_options.get_val('mass_defect_ref'),
        )

        # Boundary Constraints
        if user_options.get_val('target_mach'):
            phase.add_boundary_constraint(
                Dynamic.Mission.MACH, loc="final", equals=user_options.get_val('mach_cruise'), units="unitless"
            )

        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=user_options.get_val('final_alt'),
            units="ft",
            ref=user_options.get_val('final_alt')
        )

        if user_options.get_val('required_available_climb_rate') is not None:
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE_RATE,
                loc="final",
                lower=user_options.get_val('required_available_climb_rate'),
                units="ft/min",
                ref=1,
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

# Function to create and configure ClimbPhase


def get_climb_phase(user_options):
    # Instantiate ClimbPhase with user options
    climb_phase = ClimbPhase(user_options=user_options)

    # Build the phase
    climb = climb_phase.build_phase()

    return climb


# Example usage
user_options = AviaryValues()
# Set the required user options
climb = get_climb_phase(user_options=user_options)
