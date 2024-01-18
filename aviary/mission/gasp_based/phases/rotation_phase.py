import dymos as dm
import numpy as np

from aviary.mission.gasp_based.ode.rotation_ode import RotationODE
from aviary.variable_info.variables import Dynamic


def get_rotation(
    ode_args=None,
    transcription=None,
    fix_initial=False,
    initial_ref=1,
    duration_bounds=(1, 100),
    duration_ref=1,
    angle_lower=0.0,  # rad
    angle_upper=25 * np.pi / 180,  # rad
    angle_ref=1,
    angle_ref0=0,
    angle_defect_ref=0.01,
    TAS_lower=0,
    TAS_upper=1000,
    TAS_ref=100,
    TAS_ref0=0,
    TAS_defect_ref=None,
    mass_lower=0,
    mass_upper=190_000,
    mass_ref=100_000,
    mass_ref0=0,
    mass_defect_ref=None,
    distance_lower=0,
    distance_upper=10.e3,
    distance_ref=3000,
    distance_ref0=0,
    distance_defect_ref=3000,
    normal_ref=1,
    normal_ref0=0,
):
    phase = dm.Phase(
        ode_class=RotationODE,
        ode_init_kwargs=ode_args,
        transcription=transcription,
    )

    phase.set_time_options(
        fix_initial=fix_initial,
        fix_duration=False,
        units="s",
        targets="t_curr",
        initial_ref=initial_ref,
        duration_bounds=duration_bounds,
        duration_ref=duration_ref,
    )

    phase.add_state(
        "alpha",
        fix_initial=True,
        fix_final=False,
        lower=angle_lower,
        upper=angle_upper,
        units="rad",
        rate_source="alpha_rate",
        ref=angle_ref,
        ref0=angle_ref0,
        defect_ref=angle_defect_ref,
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
        units="ft",
        rate_source=Dynamic.Mission.DISTANCE_RATE,
        ref=distance_ref,
        ref0=distance_ref0,
        defect_ref=distance_defect_ref,
    )

    phase.add_parameter("t_init_gear", units="s",
                        static_target=True, opt=False, val=100)
    phase.add_parameter("t_init_flaps", units="s",
                        static_target=True, opt=False, val=100)

    # boundary/path constraints + controls
    phase.add_boundary_constraint(
        "normal_force",
        loc="final",
        equals=0,
        units="lbf",
        ref=normal_ref,
        ref0=normal_ref0,
    )

    phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")

    phase.add_timeseries_output("normal_force")
    phase.add_timeseries_output(Dynamic.Mission.MACH)
    phase.add_timeseries_output("EAS", units="kn")

    phase.add_timeseries_output(Dynamic.Mission.LIFT)
    phase.add_timeseries_output("CL")
    phase.add_timeseries_output("CD")
    phase.add_timeseries_output("fuselage_pitch", output_name="theta", units="deg")

    return phase
