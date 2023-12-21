import dymos as dm
import openmdao.api as om

from aviary.mission.gasp_based.phases.rotation_phase import get_rotation
from aviary.variable_info.options import get_option_defaults


def run_rotation(make_plots=False):
    rotation_trans = dm.Radau(
        num_segments=1, order=3, compressed=True, solve_segments=False
    )
    rotation = get_rotation(
        ode_args=get_option_defaults(),
        fix_initial=True,
        connect_initial_mass=False,
        transcription=rotation_trans,
    )

    p = om.Problem()
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("rotation", rotation)

    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "SNOPT"
    p.driver.opt_settings["iSumm"] = 6
    p.driver.declare_coloring()

    p.model.add_objective(
        "traj.rotation.timeseries.states:mass", index=-1, ref=1.4e5, ref0=1.3e5
    )
    p.set_solver_print(level=-1)

    p.setup()

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

    dm.run_problem(p, simulate=True, make_plots=make_plots)

    return p
