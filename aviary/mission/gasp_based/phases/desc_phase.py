import dymos as dm

from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


def get_descent(
    ode_args=None,
    transcription=None,
    fix_initial=False,
    input_initial=False,
    EAS_limit=0,
    mach_cruise=0,
    input_speed_type=SpeedType.MACH,
    final_altitude=0,
    time_initial_bounds=(0, 0),
    time_initial_ref=1,
    duration_bounds=(0, 0),
    duration_ref=1,
    alt_lower=0,
    alt_upper=0,
    alt_ref=1,
    alt_ref0=0,
    alt_defect_ref=None,
    alt_constraint_ref=None,
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
        EAS_limit=EAS_limit,
        input_speed_type=input_speed_type,
        mach_cruise=mach_cruise,
    )
    if ode_args:
        ode_init_kwargs.update(ode_args)

    desc = dm.Phase(
        ode_class=DescentODE,
        transcription=transcription,
        ode_init_kwargs=ode_init_kwargs,
    )

    desc.set_time_options(
        initial_bounds=time_initial_bounds,
        duration_bounds=duration_bounds,
        fix_initial=fix_initial,
        input_initial=input_initial,
        units="s",
        duration_ref=duration_ref,
        initial_ref=time_initial_ref,
    )

    desc.add_state(
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

    if fix_initial and input_initial:
        raise ValueError(
            "ERROR in desc_phase: fix_initial and input_initial are both True"
        )

    desc.add_state(
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

    desc.add_state(
        Dynamic.Mission.DISTANCE,
        fix_initial=fix_initial,
        input_initial=input_initial,
        fix_final=False,
        lower=distance_lower,
        upper=distance_upper,
        units="NM",
        rate_source=Dynamic.Mission.DISTANCE_RATE,
        ref=distance_ref,
        ref0=distance_ref0,
        defect_ref=distance_defect_ref,
    )

    desc.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE,
        loc="final",
        equals=final_altitude,
        units="ft",
        ref=alt_constraint_ref)

    if input_speed_type is SpeedType.EAS:
        desc.add_parameter("EAS", opt=False, units="kn", val=EAS_limit)

    desc.add_timeseries_output(
        Dynamic.Mission.MACH,
        output_name=Dynamic.Mission.MACH,
        units="unitless")
    desc.add_timeseries_output("EAS", output_name="EAS", units="kn")
    desc.add_timeseries_output("TAS", output_name="TAS", units="kn")
    desc.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                               output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
    desc.add_timeseries_output("alpha", output_name="alpha", units="deg")
    desc.add_timeseries_output("theta", output_name="theta", units="deg")
    desc.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
    desc.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL,
                               output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
    desc.add_timeseries_output("aero.CD", output_name="CD", units="unitless")

    return desc
