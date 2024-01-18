import dymos as dm
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.phases.groundroll_phase import get_groundroll
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.options import get_option_defaults


def run_groundroll():
    num_segments = 5
    transcription = dm.Radau(
        num_segments=num_segments, order=3, compressed=True, solve_segments=False
    )

    ode_args = dict(
        aviary_options=get_option_defaults()
    )

    groundroll = get_groundroll(
        fix_initial=True,
        fix_initial_mass=True,
        connect_initial_mass=False,
        ode_args=ode_args,
        transcription=transcription,
    )

    p = om.Problem()

    # calculate speed at which to initiate rotation
    p.model.add_subsystem(
        "vrot",
        om.ExecComp(
            "Vrot = ((2 * mass * g) / (rho * wing_area * CLmax))**0.5 + dV1 + dVR",
            Vrot={"units": "ft/s"},
            mass={"units": "lbm"},
            g={"units": "lbf/lbm", "val": GRAV_ENGLISH_LBM},
            rho={"units": "slug/ft**3", "val": RHO_SEA_LEVEL_ENGLISH},
            wing_area={"units": "ft**2"},
            dV1={
                "units": "ft/s",
                "desc": "Increment of engine failure decision speed above stall",
            },
            dVR={
                "units": "ft/s",
                "desc": "Increment of takeoff rotation speed above engine failure "
                "decision speed",
            },
        ),
        promotes_inputs=[
            ("wing_area", Aircraft.Wing.AREA),
            "mass",
            "dV1",
            "dVR",
            ("CLmax", Mission.Takeoff.LIFT_COEFFICIENT_MAX),
        ],
        promotes_outputs=[("Vrot", Mission.Takeoff.ROTATION_VELOCITY)]
    )

    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("groundroll", groundroll)

    p.model.add_subsystem(
        "groundroll_boundary",
        om.EQConstraintComp(
            "TAS",
            eq_units="ft/s",
            normalize=True,
            add_constraint=True,
        ),
    )
    p.model.connect(Mission.Takeoff.ROTATION_VELOCITY, "groundroll_boundary.rhs:TAS")
    p.model.connect("traj.groundroll.states:TAS", "groundroll_boundary.lhs:TAS",
                    src_indices=[-1],
                    flat_src_indices=True,
                    )

    groundroll.add_objective('time', ref=1.0)

    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options["print_results"] = p.comm.rank == 0

    optimizer = 'IPOPT'
    print_opt_iters = True

    p.driver.options["optimizer"] = optimizer

    if optimizer == "SNOPT":
        p.driver.opt_settings["Major optimality tolerance"] = 3e-4
        p.driver.opt_settings["Major feasibility tolerance"] = 1e-6
        p.driver.opt_settings["Function precision"] = 1e-6
        p.driver.opt_settings["Linesearch tolerance"] = 0.99
        if print_opt_iters:
            p.driver.opt_settings["iSumm"] = 6
        p.driver.opt_settings["Major iterations limit"] = 75
    elif optimizer == "IPOPT":
        p.driver.opt_settings["max_iter"] = 500
        p.driver.opt_settings["tol"] = 1e-5
        if print_opt_iters:
            p.driver.opt_settings["print_level"] = 5
        p.driver.opt_settings[
            "nlp_scaling_method"
        ] = "gradient-based"  # for faster convergence
        p.driver.opt_settings["alpha_for_y"] = "safer-min-dual-infeas"
        p.driver.opt_settings["mu_strategy"] = "monotone"
        p.driver.opt_settings["bound_mult_init_method"] = "mu-based"

    p.set_solver_print(level=-1)

    p.setup()

    p.set_val("traj.groundroll.states:TAS", 0, units="kn")
    p.set_val(
        "traj.groundroll.states:mass",
        groundroll.interp("mass", [175100, 174000]),
        units="lbm",
    )
    p.set_val('mass', 175100, units='lbm')
    p.set_val("dV1", 10, units="kn")
    p.set_val("dVR", 5, units="kn")
    p.set_val(Aircraft.Wing.AREA, units="ft**2", val=1370)
    p.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2.1886)

    p.set_val(
        "traj.groundroll.states:distance",
        groundroll.interp(Dynamic.Mission.DISTANCE, [0, 1000]),
        units="ft",
    )

    p.set_val("traj.groundroll.t_duration", 27.7)

    p.run_driver()
    # simout = traj.simulate(atol=1e-6, rtol=1e-6)

    return p
