import dymos as dm

from aviary.mission.gasp_based.ode.climb_ode import ClimbODE
from aviary.variable_info.variables import Mission, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def get_climb(
    ode_args=None,
    transcription=None,
    fix_initial=False,
    EAS_target=0,
    mach_cruise=0,
    target_mach=False,
    final_alt=0,
    required_available_climb_rate=None,
    time_initial_bounds=(0, 0),
    duration_bounds=(0, 0),
    duration_ref=1,
    alt_lower=0,
    alt_upper=0,
    alt_ref=1,
    alt_ref0=0,
    alt_defect_ref=None,
    mass_lower=0,
    mass_upper=0,
    mass_ref=1,
    mass_ref0=0,
    mass_defect_ref=None,
    distance_lower=0,
    distance_upper=0,
    distance_ref=1,
    distance_ref0=0,
    distance_defect_ref=None,
):
    ode_init_kwargs = dict(
        EAS_target=EAS_target,
        mach_cruise=mach_cruise,
        core_subsystems=default_mission_subsystems
    )
    if ode_args:
        ode_init_kwargs.update(ode_args)

    climb = dm.Phase(
        ode_class=ClimbODE,
        transcription=transcription,
        ode_init_kwargs=ode_init_kwargs,
    )

    climb.set_time_options(
        fix_initial=fix_initial,
        initial_bounds=time_initial_bounds,
        duration_bounds=duration_bounds,
        duration_ref=duration_ref,
        units="s",
    )

    climb.add_state(
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

    climb.add_state(
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

    climb.add_state(
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

    climb.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE,
        loc="final",
        equals=final_alt,
        units="ft",
        ref=final_alt)
    if required_available_climb_rate:
        climb.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE_RATE,
            loc="final",
            lower=required_available_climb_rate,
            units="ft/min",
            ref=1,
        )

    if target_mach is True:
        climb.add_boundary_constraint(
            Dynamic.Mission.MACH, loc="final", equals=mach_cruise, units="unitless"
        )

    climb.add_timeseries_output(
        Dynamic.Mission.MACH,
        output_name=Dynamic.Mission.MACH,
        units="unitless")
    climb.add_timeseries_output("EAS", output_name="EAS", units="kn")
    climb.add_timeseries_output(
        Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units="lbm/s"
    )
    climb.add_timeseries_output("theta", output_name="theta", units="deg")
    climb.add_timeseries_output("alpha", output_name="alpha", units="deg")
    climb.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
    climb.add_timeseries_output(
        "TAS_violation", output_name="TAS_violation", units="kn"
    )
    climb.add_timeseries_output("TAS", output_name="TAS", units="kn")
    climb.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
    climb.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL,
                                output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
    climb.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

    return climb
