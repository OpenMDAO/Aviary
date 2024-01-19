import dymos as dm

from aviary.mission.gasp_based.ode.accel_ode import AccelODE
from aviary.variable_info.variables import Dynamic


def get_accel(
    ode_args=None,
    transcription=None,
    fix_initial=False,
    alt=500,
    EAS_constraint_eq=250,
    time_initial_bounds=(0, 0),
    duration_bounds=(0, 0),
    duration_ref=1,
    TAS_lower=0,
    TAS_upper=0,
    TAS_ref=1,
    TAS_ref0=0,
    TAS_defect_ref=None,
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
    accel = dm.Phase(
        ode_class=AccelODE,
        transcription=transcription,
        ode_init_kwargs=ode_args,
    )

    accel.set_time_options(
        fix_initial=fix_initial,
        initial_bounds=time_initial_bounds,
        duration_bounds=duration_bounds,
        units="s",
        duration_ref=duration_ref,
    )

    accel.add_state(
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

    accel.add_state(
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

    accel.add_state(
        Dynamic.Mission.DISTANCE,
        fix_initial=fix_initial,
        fix_final=False,
        lower=distance_lower,
        upper=distance_upper,
        units="NM",
        rate_source=Dynamic.Mission.DISTANCE_RATE,
        ref=distance_ref,
        ref0=distance_ref0,
        defect_ref=distance_defect_ref,
    )

    accel.add_boundary_constraint(
        "EAS", loc="final", equals=EAS_constraint_eq, units="kn", ref=EAS_constraint_eq
    )

    accel.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, units="ft", val=alt)

    accel.add_timeseries_output("EAS", output_name="EAS", units="kn")
    accel.add_timeseries_output(
        Dynamic.Mission.MACH,
        output_name=Dynamic.Mission.MACH,
        units="unitless")
    accel.add_timeseries_output("alpha", output_name="alpha", units="deg")
    accel.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
    accel.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL,
                                output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
    accel.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

    return accel
