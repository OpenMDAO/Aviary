from aviary.mission.phase_builder_base import PhaseBuilderBase
from aviary.mission.initial_guess_builders import InitialGuessState, InitialGuessTime, InitialGuessControl
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
from aviary.variable_info.enums import SpeedType
from aviary.mission.gasp_based.ode.descent_ode import DescentODE


class DescentPhase(PhaseBuilderBase):
    default_name = 'descent_phase'
    default_ode_class = DescentODE

    _meta_data_ = {}
    _initial_guesses_meta_data_ = {}

    def build_phase(self, aviary_options: AviaryValues = None):
        phase = super().build_phase(aviary_options)

        # Retrieve user options values
        user_options = self.user_options
        duration_bounds = user_options.get_val('duration_bounds', units='s')
        fix_initial = user_options.get_val('fix_initial')
        input_initial = user_options.get_val('input_initial')
        duration_ref = user_options.get_val('duration_ref', units='s')
        alt_lower = user_options.get_val('alt_lower', units='ft')
        alt_upper = user_options.get_val('alt_upper', units='ft')
        alt_ref = user_options.get_val('alt_ref', units='ft')
        alt_ref0 = user_options.get_val('alt_ref0', units='ft')
        alt_defect_ref = user_options.get_val('alt_defect_ref', units='ft')
        final_altitude = user_options.get_val('final_altitude', units='ft')
        alt_constraint_ref = user_options.get_val('alt_constraint_ref', units='ft')
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
        input_speed_type = user_options.get_val('input_speed_type')
        EAS_limit = user_options.get_val('EAS_limit', units='kn')

        # Time options
        phase.set_time_options(
            duration_bounds=duration_bounds,
            fix_initial=fix_initial,
            input_initial=input_initial,
            units="s",
            duration_ref=duration_ref,
        )

        # Add states
        phase.add_state(
            Dynamic.Mission.ALTITUDE,
            fix_initial=True,
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
            input_initial=input_initial,
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
            input_initial=input_initial,
            fix_final=False,
            lower=distance_lower,
            upper=distance_upper,
            units="NM",
            rate_source="distance_rate",
            ref=distance_ref,
            ref0=distance_ref0,
            defect_ref=distance_defect_ref,
        )

        # Add boundary constraint
        phase.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE,
            loc="final",
            equals=final_altitude,
            units="ft",
            ref=alt_constraint_ref)

        # Add parameter if necessary
        if input_speed_type == SpeedType.EAS:
            phase.add_parameter("EAS", opt=False, units="kn", val=EAS_limit)

        # Add timeseries outputs
        phase.add_timeseries_output(
            Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
        phase.add_timeseries_output("EAS", output_name="EAS", units="kn")
        phase.add_timeseries_output("TAS", output_name="TAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                    output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
        phase.add_timeseries_output("alpha", output_name="alpha", units="deg")
        phase.add_timeseries_output("theta", output_name="theta", units="deg")
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
            'input_speed_type': self.user_options.get_val('input_speed_type'),
            'mach_cruise': self.user_options.get_val('mach_cruise'),
            'EAS_limit': self.user_options.get_val('EAS_limit', 'kn'),
        }


# Adding metadata for the DescentPhase
DescentPhase._add_meta_data('fix_initial', val=False)
DescentPhase._add_meta_data('input_initial', val=False)
DescentPhase._add_meta_data('EAS_limit', val=0, units='kn')
DescentPhase._add_meta_data('mach_cruise', val=0)
DescentPhase._add_meta_data('input_speed_type', val=SpeedType.MACH)
DescentPhase._add_meta_data('final_altitude', val=0, units='ft')
DescentPhase._add_meta_data('duration_bounds', val=(0, 0), units='s')
DescentPhase._add_meta_data('duration_ref', val=1, units='s')
DescentPhase._add_meta_data('alt_lower', val=0, units='ft')
DescentPhase._add_meta_data('alt_upper', val=0, units='ft')
DescentPhase._add_meta_data('alt_ref', val=1, units='ft')
DescentPhase._add_meta_data('alt_ref0', val=0, units='ft')
DescentPhase._add_meta_data('alt_defect_ref', val=None, units='ft')
DescentPhase._add_meta_data('alt_constraint_ref', val=None, units='ft')
DescentPhase._add_meta_data('mass_lower', val=0, units='lbm')
DescentPhase._add_meta_data('mass_upper', val=0, units='lbm')
DescentPhase._add_meta_data('mass_ref', val=1, units='lbm')
DescentPhase._add_meta_data('mass_ref0', val=0, units='lbm')
DescentPhase._add_meta_data('mass_defect_ref', val=None, units='lbm')
DescentPhase._add_meta_data('distance_lower', val=0, units='NM')
DescentPhase._add_meta_data('distance_upper', val=0, units='NM')
DescentPhase._add_meta_data('distance_ref', val=1, units='NM')
DescentPhase._add_meta_data('distance_ref0', val=0, units='NM')
DescentPhase._add_meta_data('distance_defect_ref', val=None, units='NM')
DescentPhase._add_meta_data('num_segments', val=None, units='unitless')
DescentPhase._add_meta_data('order', val=None, units='unitless')

# Adding initial guess metadata
DescentPhase._add_initial_guess_meta_data(
    InitialGuessTime(),
    desc='initial guess for time options')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('altitude'),
    desc='initial guess for altitude state')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('mass'),
    desc='initial guess for mass state')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'),
    desc='initial guess for distance state')
DescentPhase._add_initial_guess_meta_data(
    InitialGuessControl('throttle'),
    desc='initial guess for throttle')
