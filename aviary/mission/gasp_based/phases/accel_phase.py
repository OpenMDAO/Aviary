from aviary.mission.phase_builder_base import (
    PhaseBuilderBase, InitialGuessState, InitialGuessTime, InitialGuessControl)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.mission.gasp_based.ode.accel_ode import AccelODE
from aviary.variable_info.variable_meta_data import _MetaData


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
        phase = super().build_phase(aviary_options)
        user_options = self.user_options

        # Extracting and setting options
        fix_initial = user_options.get_val('fix_initial')
        EAS_constraint_eq = user_options.get_val('EAS_constraint_eq', units='kn')
        time_initial_bounds = user_options.get_val('time_initial_bounds', units='s')
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
        distance_lower = user_options.get_val('distance_lower', units='NM')
        distance_upper = user_options.get_val('distance_upper', units='NM')
        distance_ref = user_options.get_val('distance_ref', units='NM')
        distance_ref0 = user_options.get_val('distance_ref0', units='NM')
        distance_defect_ref = user_options.get_val('distance_defect_ref', units='NM')
        alt = user_options.get_val('alt', units='ft')

        phase.set_time_options(
            fix_initial=fix_initial,
            initial_bounds=time_initial_bounds,
            duration_bounds=duration_bounds,
            units="s",
            duration_ref=duration_ref,
        )

        # States
        phase.add_state(
            "TAS",
            fix_initial=fix_initial,
            fix_final=False,
            lower=TAS_lower,
            upper=TAS_upper,
            units="kn",
            rate_source="TAS_rate",
            targets="TAS",
            ref=TAS_ref,
            ref0=TAS_ref0,
            defect_ref=TAS_defect_ref,
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
AccelPhase._add_meta_data('time_initial_bounds', val=(0, 0), units='s')
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
