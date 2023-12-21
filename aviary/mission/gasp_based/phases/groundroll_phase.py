import dymos as dm

from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.variable_info.variables import Dynamic


def get_groundroll(
    ode_args=None,
    transcription=None,
    fix_initial=True,
    fix_initial_mass=False,
    connect_initial_mass=True,
    duration_bounds=(1, 100),
    duration_ref=1,
    TAS_lower=0,
    TAS_upper=1000,
    TAS_ref=100,
    TAS_ref0=0,
    TAS_defect_ref=None,
    mass_lower=0,
    mass_upper=200_000,
    mass_ref=100_000,
    mass_ref0=0,
    mass_defect_ref=100,
    distance_lower=0,
    distance_upper=4000,
    distance_ref=3000,
    distance_ref0=0,
    distance_defect_ref=3000,
):
    phase = dm.Phase(
        ode_class=GroundrollODE,
        ode_init_kwargs=ode_args,
        transcription=transcription,
    )

    phase.set_time_options(
        fix_initial=fix_initial,
        fix_duration=False,
        units="s",
        targets="t_curr",
        duration_bounds=duration_bounds,
        duration_ref=duration_ref,
    )

    phase.add_state(
        "TAS",
        fix_initial=fix_initial,
        fix_final=False,
        lower=TAS_lower,
        upper=TAS_upper,
        units="kn",
        rate_source="TAS_rate",
        ref=TAS_ref,
        defect_ref=TAS_defect_ref,
        ref0=TAS_ref0,
    )

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

    phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")

    phase.add_timeseries_output("normal_force")
    phase.add_timeseries_output(Dynamic.Mission.MACH)
    phase.add_timeseries_output("EAS", units="kn")

    phase.add_timeseries_output(Dynamic.Mission.LIFT)
    phase.add_timeseries_output("CL")
    phase.add_timeseries_output("CD")
    phase.add_timeseries_output("fuselage_pitch", output_name="theta", units="deg")

    return phase
