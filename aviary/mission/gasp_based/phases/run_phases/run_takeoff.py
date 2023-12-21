import dymos as dm
import openmdao.api as om

from aviary.mission.gasp_based.phases.ascent_phase import get_ascent
from aviary.mission.gasp_based.phases.groundroll_phase import get_groundroll
from aviary.mission.gasp_based.phases.rotation_phase import get_rotation
from aviary.mission.gasp_based.polynomial_fit import PolynomialFit
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic, Mission


def run_takeoff(make_plots=False):
    option_defaults = get_option_defaults()
    groundroll_trans = dm.Radau(
        num_segments=1, order=3, compressed=True, solve_segments=False
    )
    groundroll = get_groundroll(
        ode_args=option_defaults,
        fix_initial=True,
        connect_initial_mass=False,
        transcription=groundroll_trans,
    )

    p = om.Problem()

    p.model.add_subsystem(
        "event_xform",
        om.ExecComp(
            ["t_init_gear=m*tau_gear+b", "t_init_flaps=m*tau_flaps+b"], units="s"
        ),
        promotes_inputs=[
            "tau_gear",
            "tau_flaps",
            ("m", Mission.Takeoff.ASCENT_DURATION),
            ("b", "ascent:t_initial"),
        ],
        promotes_outputs=["t_init_gear", "t_init_flaps"],
    )

    p.model.add_design_var("ascent:t_initial", lower=0, upper=100)
    p.model.add_design_var(Mission.Takeoff.ASCENT_DURATION, lower=1, upper=1000)

    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("groundroll", groundroll)

    rotation_trans = dm.Radau(
        num_segments=1, order=3, compressed=True, solve_segments=False
    )
    rotation = get_rotation(
        ode_args=option_defaults,
        fix_initial=False,
        connect_initial_mass=False,
        transcription=rotation_trans,
    )
    traj.add_phase("rotation", rotation)
    traj.link_phases(["groundroll", "rotation"], ["time", "TAS", "mass", "distance"])

    ascent_trans = dm.Radau(
        num_segments=2, order=5, compressed=True, solve_segments=False
    )
    ascent = get_ascent(
        ode_args=option_defaults,
        fix_initial=False,
        connect_initial_mass=False,
        transcription=ascent_trans,
    )
    traj.add_phase("ascent", ascent)
    traj.link_phases(
        ["rotation", "ascent"], ["time", "TAS", "mass", "distance", "alpha"]
    )
    p.model.promotes(
        "traj",
        inputs=[
            ("ascent.parameters:t_init_gear", "t_init_gear"),
            ("ascent.parameters:t_init_flaps", "t_init_flaps"),
            ("ascent.t_initial", "ascent:t_initial"),
            ("ascent.t_duration", Mission.Takeoff.ASCENT_DURATION),
        ],
    )

    p.model.promotes(
        "traj",
        inputs=[
            (
                "groundroll.parameters:" + Mission.Design.GROSS_MASS,
                Mission.Design.GROSS_MASS,
            ),
            (
                "rotation.parameters:" + Mission.Design.GROSS_MASS,
                Mission.Design.GROSS_MASS,
            ),
            (
                "ascent.parameters:" + Mission.Design.GROSS_MASS,
                Mission.Design.GROSS_MASS,
            ),
        ],
    )
    p.model.set_input_defaults(Mission.Design.GROSS_MASS, 175400, units="lbm")

    p.model.set_input_defaults("ascent:t_initial", val=10.0)
    p.model.set_input_defaults(Mission.Takeoff.ASCENT_DURATION, val=30.0)

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
    p.model.add_design_var("tau_gear", lower=0.01, upper=0.75, units="s", ref=1)
    p.model.add_constraint("h_fit.h_init_gear", equals=50.0, units="ft", ref=50.0)

    p.model.add_design_var("tau_flaps", lower=0.3, upper=1.0, units="s", ref=1)
    p.model.add_constraint("h_fit.h_init_flaps", equals=400.0, units="ft", ref=400.0)

    p.model.add_objective(
        "traj.ascent.timeseries.states:mass", index=-1, ref0=-1.747e5, ref=-1.749e5
    )

    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "SNOPT"
    p.driver.opt_settings["Major feasibility tolerance"] = 1.0e-7
    p.driver.opt_settings["Major optimality tolerance"] = 1.0e-2
    p.driver.opt_settings["Function precision"] = 1e-6
    p.driver.opt_settings["Linesearch tolerance"] = 0.95
    p.driver.opt_settings["iSumm"] = 6
    p.driver.declare_coloring()

    p.model.linear_solver = om.DirectSolver()

    p.setup()

    p.set_val("traj.groundroll.states:TAS", 0, units="kn")
    p.set_val(
        "traj.groundroll.states:mass",
        groundroll.interp("mass", [175100, 174000]),
        units="lbm",
    )
    p.set_val(
        "traj.groundroll.states:distance",
        groundroll.interp("distance", [0, 1000]),
        units="ft",
    )
    p.set_val("traj.groundroll.t_duration", 50.0)

    p.set_val(
        "traj.rotation.states:alpha", rotation.interp("alpha", [0, 2.5]), units="deg"
    )
    p.set_val("traj.rotation.states:TAS", 143, units="kn")
    p.set_val(
        "traj.rotation.states:mass",
        rotation.interp("mass", [174975.12776915, 174000]),
        units="lbm",
    )
    p.set_val(
        "traj.rotation.states:distance",
        rotation.interp("distance", [3680.37217765, 4000]),
        units="ft",
    )
    p.set_val("traj.rotation.t_duration", 50.0)

    p.set_val(
        "traj.ascent.states:altitude",
        ascent.interp(Dynamic.Mission.ALTITUDE, ys=[0, 100, 500], xs=[0, 1, 10]),
        units="ft",
    )
    p.set_val("traj.ascent.states:flight_path_angle", 0.0, units="rad")
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
        ascent.interp("distance", [4330.83393029, 5000]),
        units="ft",
    )
    p.set_val("traj.ascent.t_initial", 31.2)
    p.set_val("traj.ascent.t_duration", 10.0)

    # event trigger times
    p.set_val("tau_gear", 0.2)  # initial guess
    p.set_val("tau_flaps", 0.5)  # initial guess

    p.set_solver_print(level=-1)

    dm.run_problem(p, simulate=True, make_plots=make_plots)

    return p
