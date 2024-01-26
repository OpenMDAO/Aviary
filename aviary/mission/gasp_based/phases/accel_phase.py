from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessTime, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.accel_ode import AccelODE


class AccelPhase(PhaseBuilderBase):
    """
    A phase builder for an acceleration phase in a mission simulation.

    This class extends the PhaseBuilderBase class, providing specific implementations for
    the acceleration phase of a flight mission.

    Attributes
    ----------
    Inherits all attributes from PhaseBuilderBase.

    Methods
    -------
    Inherits all methods from PhaseBuilderBase.
    Additional method overrides and new methods specific to the acceleration phase are included.
    """
    default_name = 'accel_phase'
    default_ode_class = AccelODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        """
        Return a new acceleration phase for analysis using these constraints.

        Parameters
        ----------
        aviary_options : AviaryValues
            Collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        phase = self.phase = super().build_phase(aviary_options)
        user_options = self.user_options

        # Extracting and setting options
        fix_initial = user_options.get_val('fix_initial')
        EAS_constraint_eq = user_options.get_val('EAS_constraint_eq', units='kn')
        duration_bounds = user_options.get_val('duration_bounds', units='s')
        duration_ref = user_options.get_val('duration_ref', units='s')
        alt = user_options.get_val('alt', units='ft')

        phase.set_time_options(
            fix_initial=fix_initial,
            duration_bounds=duration_bounds,
            units="s",
            duration_ref=duration_ref,
        )

        # States
        self.add_TAS_state(user_options)

        self.add_mass_state(user_options)

        self.add_distance_state(user_options)

        # Boundary Constraints
        phase.add_boundary_constraint(
            "EAS", loc="final", equals=EAS_constraint_eq, units="kn", ref=EAS_constraint_eq
        )

        phase.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, units="ft", val=alt)

        # Timeseries Outputs
        phase.add_timeseries_output("EAS", output_name="EAS", units="kn")
        phase.add_timeseries_output(
            Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
        phase.add_timeseries_output("alpha", output_name="alpha", units="deg")
        phase.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
        phase.add_timeseries_output(
            Dynamic.Mission.THRUST_TOTAL, output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

        return phase


# Adding metadata for the AccelPhase
AccelPhase._add_meta_data('fix_initial', val=False)
AccelPhase._add_meta_data('EAS_constraint_eq', val=250, units='kn')
AccelPhase._add_meta_data('duration_bounds', val=(0, 0), units='s')
AccelPhase._add_meta_data('duration_ref', val=1, units='s')
AccelPhase._add_meta_data('TAS_lower', val=0, units='kn')
AccelPhase._add_meta_data('TAS_upper', val=0, units='kn')
AccelPhase._add_meta_data('TAS_ref', val=1, units='kn')
AccelPhase._add_meta_data('TAS_ref0', val=0, units='kn')
AccelPhase._add_meta_data('TAS_defect_ref', val=None, units='kn')
AccelPhase._add_meta_data('mass_lower', val=0, units='lbm')
AccelPhase._add_meta_data('mass_upper', val=0, units='lbm')
AccelPhase._add_meta_data('mass_ref', val=1, units='lbm')
AccelPhase._add_meta_data('mass_ref0', val=0, units='lbm')
AccelPhase._add_meta_data('mass_defect_ref', val=None, units='lbm')
AccelPhase._add_meta_data('distance_lower', val=0, units='NM')
AccelPhase._add_meta_data('distance_upper', val=0, units='NM')
AccelPhase._add_meta_data('distance_ref', val=1, units='NM')
AccelPhase._add_meta_data('distance_ref0', val=0, units='NM')
AccelPhase._add_meta_data('distance_defect_ref', val=None, units='NM')
AccelPhase._add_meta_data('alt', val=500, units='ft')
AccelPhase._add_meta_data('num_segments', val=None, units='unitless')
AccelPhase._add_meta_data('order', val=None, units='unitless')

AccelPhase._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for initial time and duration specified as a tuple')

AccelPhase._add_initial_guess_meta_data(
    InitialGuessState('TAS'),
    desc='initial guess for true airspeed')

AccelPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass')

AccelPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for horizontal distance traveled')

AccelPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
