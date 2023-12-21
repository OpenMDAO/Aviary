import dymos as dm
import numpy as np

from aviary.mission.gasp_based.ode.ascent_ode import AscentODE
from aviary.variable_info.variables import Dynamic


def get_ascent(
    ode_args=None,
    transcription=None,
    fix_initial=False,
    angle_lower=-15 * np.pi / 180,
    angle_upper=25 * np.pi / 180,
    angle_ref=np.deg2rad(1),
    angle_ref0=0,
    angle_defect_ref=0.01,
    alt_lower=0,
    alt_upper=700,
    alt_ref=100,
    alt_ref0=0,
    alt_defect_ref=100,
    alt_constraint_eq=500,
    alt_constraint_ref=100,
    alt_constraint_ref0=0,
    TAS_lower=0,
    TAS_upper=1000,
    TAS_ref=1e2,
    TAS_ref0=0,
    TAS_defect_ref=None,
    mass_lower=0,
    mass_upper=190_000,
    mass_ref=100_000,
    mass_ref0=0,
    mass_defect_ref=1e2,
    distance_lower=0,
    distance_upper=10.e3,
    distance_ref=3000,
    distance_ref0=0,
    distance_defect_ref=3000,
    pitch_constraint_lower=0,
    pitch_constraint_upper=15,
    pitch_constraint_ref=1,
    alpha_constraint_lower=np.deg2rad(-30),
    alpha_constraint_upper=np.deg2rad(30),
    alpha_constraint_ref=np.deg2rad(5),
):
    phase = dm.Phase(
        ode_class=AscentODE,
        ode_init_kwargs=ode_args,
        transcription=transcription,
    )

    phase.set_time_options(
        units="s",
        targets="t_curr",
        input_initial=True,
        input_duration=True,
    )

    phase.add_state(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=True,
        fix_final=False,
        lower=angle_lower,
        upper=angle_upper,
        units="rad",
        rate_source=Dynamic.Mission.FLIGHT_PATH_ANGLE_RATE,
        ref=angle_ref,
        defect_ref=angle_defect_ref,
        ref0=angle_ref0,
    )

    phase.add_state(
        Dynamic.Mission.ALTITUDE,
        fix_initial=True,
        fix_final=False,
        lower=alt_lower,
        upper=alt_upper,
        units="ft",
        rate_source=Dynamic.Mission.ALTITUDE_RATE,
        ref=alt_ref,
        defect_ref=alt_defect_ref,
        ref0=alt_ref0,
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
        fix_initial=fix_initial,
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

    # boundary/path constraints + controls
    phase.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE,
        loc="final",
        equals=alt_constraint_eq,
        units="ft",
        ref=alt_constraint_ref,
        ref0=alt_constraint_ref0,
    )
    phase.add_path_constraint("load_factor", upper=1.10, lower=0.0)
    phase.add_path_constraint(
        "fuselage_pitch",
        "theta",
        lower=pitch_constraint_lower,
        upper=pitch_constraint_upper,
        units="deg",
        ref=pitch_constraint_ref,
    )

    phase.add_control(
        "alpha",
        val=0,
        lower=alpha_constraint_lower,
        upper=alpha_constraint_upper,
        units="rad",
        ref=alpha_constraint_ref,
        opt=True,
    )
    phase.add_parameter("t_init_gear", units="s",
                        static_target=True, opt=False, val=38.25)
    phase.add_parameter("t_init_flaps", units="s",
                        static_target=True, opt=False, val=48.21)
    phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")

    phase.add_timeseries_output("normal_force")
    phase.add_timeseries_output(Dynamic.Mission.MACH)
    phase.add_timeseries_output("EAS", units="kn")

    phase.add_timeseries_output(Dynamic.Mission.LIFT)
    phase.add_timeseries_output("CL")
    phase.add_timeseries_output("CD")

    return phase
