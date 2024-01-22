import argparse
import os

import dymos as dm
import matplotlib.pyplot as plt
import openmdao.api as om
import pandas as pd

from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.phases.climb_phase import get_climb
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic

thisdir = os.path.dirname(__file__)
def_outdir = os.path.join(thisdir, "output")

gaspdata_dir = os.path.join(thisdir, "..", "..", "problem",
                            "large_single_aisle_validation_data")

theta_max = 15
TAS_violation_max = 0
use_surrogate = True

ode_args = {}

# name: (gaspname, path, units)
varinfo = {
    "time": ("TIME", "time", "s"),
    Dynamic.Mission.ALTITUDE: ("ALT", "states:altitude", "ft"),
    "mass": ("MASS", "states:mass", "lbm"),
    Dynamic.Mission.DISTANCE: (Dynamic.Mission.DISTANCE, "states:distance", "NM"),
    Dynamic.Mission.MACH: ("MACH", Dynamic.Mission.MACH, None),
    "EAS": ("EAS", "EAS", "kn"),
    "alpha": ("ALPHA", "alpha", "deg"),
    Dynamic.Mission.FLIGHT_PATH_ANGLE: ("GAMMA", Dynamic.Mission.FLIGHT_PATH_ANGLE, "deg"),
    "theta": ("FUSANGLE", "theta", "deg"),
    "TAS_violation": (None, "TAS_violation", "kn"),
    "CL": ("CL", "CL", None),
    "CD": ("CD", "CD", None),
    Dynamic.Mission.THRUST_TOTAL: ("THRUST", Dynamic.Mission.THRUST_TOTAL, "lbf"),
}


def setup_climb1():
    prob = om.Problem(model=om.Group(), name="main")
    setup_driver(prob)

    ode_args["aviary_options"] = get_option_defaults()

    transcription = dm.Radau(num_segments=1, order=5, compressed=False)
    climb1 = get_climb(
        ode_args=ode_args,
        transcription=transcription,
        fix_initial=True,
        alt_lower=0,
        alt_upper=11000,
        mass_lower=0,
        mass_upper=200_000,
        distance_lower=0,
        distance_upper=300,
        time_initial_bounds=(0, 0),
        duration_bounds=(1.1, 36_000),
        EAS_target=250,
        mach_cruise=0.8,
        target_mach=False,
        final_altitude=10000,
        alt_ref=10000,
        mass_ref=200000,
        distance_ref=300,
    )
    prob.model.add_subsystem(name="climb1", subsys=climb1)

    # add all params and promote them to prob.model level
    ParamPort.promote_params(prob.model, phases=["climb1"])
    ParamPort.set_default_vals(prob.model)

    climb1.add_objective("time", loc="final")

    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.LinearRunOnce()

    prob.setup()

    prob.set_val("climb1.controls:throttle",
                 climb1.interp("throttle", ys=[1.0, 1.0]))
    prob.set_val("climb1.states:mass", climb1.interp("mass", ys=[174878, 174219]))
    prob.set_val(
        "climb1.timeseries:altitude", climb1.interp(
            Dynamic.Mission.ALTITUDE, ys=[
                500, 10000]))
    prob.set_val("climb1.states:distance", climb1.interp(
        Dynamic.Mission.DISTANCE, ys=[2, 15]))
    prob.set_val("climb1.t_duration", 1000)
    prob.set_val("climb1.t_initial", 0)

    return prob


def setup_climb2():
    prob = om.Problem(model=om.Group(), name="main")

    setup_driver(prob)

    ode_args["aviary_options"] = get_option_defaults()

    transcription = dm.Radau(num_segments=3, order=3, compressed=False)
    climb2 = get_climb(
        ode_args=ode_args,
        transcription=transcription,
        fix_initial=True,
        alt_lower=9000,
        alt_upper=60000,
        mass_lower=0,
        mass_upper=200_000,
        distance_lower=0,
        distance_upper=300,
        time_initial_bounds=(0, 0),
        duration_bounds=(2, 36_000),
        EAS_target=270,
        mach_cruise=0.8,
        target_mach=True,
        final_altitude=37500,
        alt_ref=40000,
        mass_ref=200000,
        distance_ref=300,
        required_available_climb_rate=300,
    )
    prob.model.add_subsystem(name="climb2", subsys=climb2)

    # fixed initial mass, so max final mass is equivalent to min fuel burn
    climb2.add_objective("mass", loc="final", ref=-1)

    # add all params and promote them to prob.model level
    ParamPort.promote_params(prob.model, phases=["climb2"])
    ParamPort.set_default_vals(prob.model)

    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.LinearRunOnce()

    prob.setup()
    prob.set_val("climb2.states:mass", climb2.interp("mass", ys=[174219, 171481]))
    prob.set_val("climb2.controls:throttle",
                 climb2.interp("throttle", ys=[1.0, 1.0]))
    prob.set_val(
        "climb2.states:altitude", climb2.interp(
            Dynamic.Mission.ALTITUDE, ys=[
                10000, 37500]))
    prob.set_val("climb2.states:distance", climb2.interp(
        Dynamic.Mission.DISTANCE, ys=[15, 154]))
    prob.set_val("climb2.t_duration", 1500)
    prob.set_val("climb2.t_initial", 0)

    return prob


def setup_driver(prob):
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SNOPT"
    prob.driver.opt_settings["Major iterations limit"] = 100
    prob.driver.opt_settings["Major optimality tolerance"] = 5.0e-3
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-6
    prob.driver.opt_settings["iSumm"] = 6
    prob.driver.declare_coloring()
    prob.set_solver_print(level=0)


def make_recorder_filepaths(phasename, outdir):
    return (
        os.path.join(outdir, f"{phasename}_prob.sql"),
        os.path.join(outdir, f"{phasename}_sim.sql"),
    )


def run_phase(phasename, prob, outdir=def_outdir):
    os.makedirs(outdir, exist_ok=True)
    prob_rec_fp = make_recorder_filepaths(phasename, outdir)[0]

    recorder = om.SqliteRecorder(prob_rec_fp)
    prob.driver.add_recorder(recorder)
    prob.driver.recording_options["includes"] = ["*timeseries*"]
    prob.add_recorder(recorder)

    print("\nRunning driver...\n")
    prob.run_driver()

    prob.record("final")
    prob.cleanup()

    return prob


def sim_phase(phasename, prob, outdir=def_outdir):
    os.makedirs(outdir, exist_ok=True)
    sim_rec_fp = make_recorder_filepaths(phasename, outdir)[1]

    print("\nSimulating...\n")
    sys = getattr(prob.model, phasename)
    prob.set_solver_print(level=-1)
    sys.rhs_all.prop.set_solver_print(level=-1, depth=1e99)
    simprob = sys.simulate(atol=1e-6, rtol=1e-6, record_file=sim_rec_fp)

    return simprob


def load_phase(phasename, outdir=def_outdir):
    for fp in make_recorder_filepaths(phasename, outdir):
        cr = om.CaseReader(fp)
        yield cr.get_case("final")


def load_phase_nosim(phasename, outdir=def_outdir):
    fp = make_recorder_filepaths(phasename, outdir)[0]
    cr = om.CaseReader(fp)
    return cr.get_case("final")


def gen_plots(phasename, probdata, simdata, plot_dir, show=False):
    gaspdata = pd.read_csv(os.path.join(gaspdata_dir, f"{phasename}_data.csv"))
    # gasp data in hours, convert to seconds, reference to 0
    gaspdata["TIME"] = 3600 * (gaspdata["TIME"] - gaspdata["TIME"][0])

    def gen_plot(title, varnames):
        fig, axs = plt.subplots(len(varnames), 1, sharex=True)
        fig.suptitle(title)

        if len(varnames) == 1:
            axs = [axs]

        gaspname, path, units = varinfo["time"]
        path = f"{phasename}.timeseries.{path}"
        t_solved = probdata.get_val(path, units=units)
        if simdata:
            t_sim = simdata.get_val(path, units=units)
        t_gasp = gaspdata[gaspname]

        for ax, varname in zip(axs, varnames):
            gaspname, path, units = varinfo[varname]
            path = f"{phasename}.timeseries.{path}"
            ax.plot(t_solved, probdata.get_val(path, units=units), "bo", label="opt")
            if simdata:
                ax.plot(t_sim, simdata.get_val(path, units=units), "b--", label="sim")
            if gaspname is not None:
                ax.plot(t_gasp, gaspdata[gaspname], "k-", label="GASP")
            ylabel = f"{varname}"
            if units:
                ylabel += f" ({units})"
            ax.set_ylabel(ylabel)

        axs[0].legend()
        axs[-1].set_xlabel("time (s)")

        plt.tight_layout()

        return fig

    gen_plot("States", [Dynamic.Mission.ALTITUDE, "mass", Dynamic.Mission.DISTANCE])
    plt.savefig(os.path.join(plot_dir, f"{phasename}_states.pdf"))

    gen_plot("Speeds", [Dynamic.Mission.MACH, "EAS"])
    plt.savefig(os.path.join(plot_dir, f"{phasename}_speeds.pdf"))

    gen_plot("Angles", ["alpha", Dynamic.Mission.FLIGHT_PATH_ANGLE])
    plt.savefig(os.path.join(plot_dir, f"{phasename}_angles.pdf"))

    fig = gen_plot("Constraints", ["TAS_violation", "theta"])
    fig.axes[0].axhline(theta_max, color="r", linestyle=":")
    fig.axes[1].axhline(TAS_violation_max, color="r", linestyle=":")
    plt.savefig(os.path.join(plot_dir, f"{phasename}_constraints.pdf"))

    gen_plot("Aero", ["CL", "CD"])
    plt.savefig(os.path.join(plot_dir, f"{phasename}_aero.pdf"))

    gen_plot("Thrust", [Dynamic.Mission.THRUST_TOTAL])
    plt.savefig(os.path.join(plot_dir, f"{phasename}_thrust.pdf"))

    if show:
        plt.show()
