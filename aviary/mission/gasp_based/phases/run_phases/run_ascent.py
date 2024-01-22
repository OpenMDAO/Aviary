import dymos as dm
import openmdao.api as om

from aviary.mission.gasp_based.phases.ascent_phase import get_ascent
from aviary.mission.gasp_based.polynomial_fit import PolynomialFit
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.interface.default_phase_info.height_energy import default_mission_subsystems


def run_ascent(make_plots=False):

    solve_segments = False  # changed from "forward" because got a singular matrix error
    ascent_trans = dm.Radau(
        num_segments=2, order=5, compressed=True, solve_segments=solve_segments
    )
    ascent = get_ascent(
        fix_initial=True,
        transcription=ascent_trans,
        TAS_ref=1,
        alt_ref=1,
        alt_defect_ref=1,
        ode_args=get_option_defaults(),
        core_subsystems=default_mission_subsystems
    )

    p = om.Problem()
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("ascent", ascent)

    p.model.promotes(
        "traj",
        inputs=[
            ("ascent.parameters:t_init_gear", "t_init_gear"),
            ("ascent.parameters:t_init_flaps", "t_init_flaps"),
            (
                "ascent.parameters:" + Mission.Design.GROSS_MASS,
                Mission.Design.GROSS_MASS,
            ),
            ("ascent.t_duration", "t_duration"),
        ],
    )

    ascent_tx = ascent.options["transcription"]
    ascent_num_nodes = ascent_tx.grid_data.num_nodes

    p.model.add_subsystem(
        "h_fit",
        PolynomialFit(N_cp=ascent_num_nodes),
        promotes_inputs=["t_init_gear", "t_init_flaps"],
    )
    p.model.connect("traj.ascent.timeseries.time", "h_fit.time_cp")
    p.model.connect("traj.ascent.timeseries.states:altitude", "h_fit.h_cp")

    # TODO: re-parameterize time to be 0-1 (i.e. add a component that offsets by t_initial/t_duration)
    p.model.add_design_var("t_init_gear", lower=32.0, upper=45.0, units="s", ref=5.0)
    p.model.add_constraint("h_fit.h_init_gear", equals=50.0, units="ft", ref=50.0)

    p.model.add_design_var("t_init_flaps", lower=45.0, upper=100.0, units="s", ref=50.0)
    p.model.add_constraint("h_fit.h_init_flaps", equals=400.0, units="ft", ref=400.0)

    p.model.add_design_var("t_duration", lower=5, upper=100, units="s")
    p.model.set_input_defaults("t_duration", val=25, units="s")

    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "SNOPT"
    p.driver.opt_settings["iSumm"] = 6
    # p.driver.options['debug_print'] = ['desvars']
    p.driver.declare_coloring()

    # p.model.add_objective("traj.ascent.timeseries.states:mass", index=-1, ref=1.4e5, ref0=1.3e5)
    ascent.add_objective("time", loc="final")
    p.set_solver_print(level=-1)

    ascent.add_parameter(
        Aircraft.Wing.AREA,
        opt=False,
        units="ft**2",
        val=1370,
        static_target=True,
        targets=[Aircraft.Wing.AREA],
    )
    p.model.set_input_defaults(Mission.Design.GROSS_MASS, val=174000, units="lbm")
    p.setup()

    p.set_val(
        "traj.ascent.states:altitude",
        ascent.interp(Dynamic.Mission.ALTITUDE, ys=[0, 100, 500], xs=[0, 1, 10]),
        units="ft",
    )
    p.set_val("traj.ascent.states:flight_path_angle", 0.2, units="rad")

    p.set_val(
        "traj.ascent.states:TAS", ascent.interp("TAS", [153.3196491, 500]), units="kn"
    )
    p.set_val(
        "traj.ascent.states:mass",
        ascent.interp("mass", [174963.74211336, 174000]),
        units="lbm",
    )
    p.set_val(
        "traj.ascent.states:distance",
        ascent.interp(Dynamic.Mission.DISTANCE, [4330.83393029, 5000]),
        units="ft",
    )
    p.set_val("traj.ascent.t_initial", 31.2)
    # p.set_val("traj.ascent.t_duration", 10.0)

    p.set_val("t_init_gear", 40.0)  # initial guess
    p.set_val("t_init_flaps", 47.5)  # initial guess

    p.set_val(
        "traj.ascent.controls:alpha",
        ascent.interpolate(ys=[1, 1], nodes="control_input"),
        units="deg",
    )

    dm.run_problem(p, simulate=True, make_plots=make_plots)

    print("t_init_gear (s)", p.get_val("t_init_gear", units="s"))
    print("t_init_flaps (s)", p.get_val("t_init_flaps", units="s"))

    return p
