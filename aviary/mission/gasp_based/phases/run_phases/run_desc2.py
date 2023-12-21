import dymos as dm
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
import pandas as pd

from aviary.mission.gasp_based.ode.descent_ode import DescentODE
from aviary.mission.gasp_based.phases.desc_phase import get_descent
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic


def run_desc2():
    theta_max_val = 15
    plot_over_alt = False

    prob = om.Problem(model=om.Group())

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SNOPT"
    prob.driver.opt_settings["Major iterations limit"] = 100
    prob.driver.opt_settings["Major optimality tolerance"] = 5.0e-3
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-6
    prob.driver.opt_settings["iSumm"] = 6
    prob.driver.declare_coloring()

    transcription = dm.Radau(num_segments=15, order=4, compressed=False)
    desc2 = get_descent(
        DescentODE,
        fix_initial=True,
        input_initial=False,
        alt_lb=1000,
        alt_ub=60000,
        mass_lb=0,
        mass_ub=200000,
        distance_lb=2000,
        distance_ub=3000,
        time_lb=(0, 0),
        time_ub=(2, 36000),
        EAS_limit=250,
        mach_cruise=0.8,
        transcription=transcription,
        final_alt=1000,
        input_speed_type=SpeedType.EAS,
    )

    desc2.add_parameter(
        Dynamic.Mission.THROTTLE,
        opt=False,
        units="unitless",
        val=0.0,
        static_target=False)

    desc2.add_parameter(
        Aircraft.Wing.AREA, opt=False, units="ft**2", val=1370.0, static_target=True
    )

    prob.model.add_subsystem(name="desc2", subsys=desc2)
    desc2.add_timeseries_output(
        Dynamic.Mission.MACH,
        output_name=Dynamic.Mission.MACH,
        units="unitless")
    desc2.add_timeseries_output("EAS", output_name="EAS", units="kn")
    desc2.add_timeseries_output(
        Dynamic.Mission.THRUST_TOTAL,
        output_name=Dynamic.Mission.THRUST_TOTAL,
        units="lbf")
    desc2.add_timeseries_output("aero.CD", output_name="CD", units="unitless")
    desc2.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                                output_name=Dynamic.Mission.FLIGHT_PATH_ANGLE, units="deg")
    desc2.add_timeseries_output("alpha", output_name="alpha", units="deg")
    desc2.add_timeseries_output("aero.CL", output_name="CL", units="unitless")
    desc2.add_timeseries_output(
        "required_lift", output_name="required_lift", units="lbf"
    )
    desc2.add_timeseries_output("theta", output_name="theta", units="deg")
    desc2.add_timeseries_output("TAS", output_name="TAS", units="kn")

    desc2.add_objective("time", loc="final")
    prob.model.linear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["iprint"] = 0

    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

    prob.setup()
    prob.set_val("desc2.states:mass", desc2.interp("mass", ys=[147541.5, 147390]))
    prob.set_val(
        "desc2.states:altitude", desc2.interp(
            Dynamic.Mission.ALTITUDE, ys=[
                10000, 1000]))
    prob.set_val("desc2.states:distance", desc2.interp(
        Dynamic.Mission.DISTANCE, ys=[2563.5, 2598]))
    prob.set_val("desc2.t_duration", 500)
    prob.set_val("desc2.t_initial", 0)

    prob.set_solver_print(level=0)
    prob.run_driver()
    print()
    print("Done Running Model, Now Simulating")
    print()

    simout5 = desc2.simulate(atol=1e-6, rtol=1e-6)

    if make_plots == True:

        ## make plots ##

        t_solved = prob.get_val("desc2.timeseries.time")
        t_simulate = simout5.get_val("desc2.timeseries.time")

        alt_solved = prob.get_val("desc2.timeseries.states:altitude", units="ft")
        alt_simulate = simout5.get_val("desc2.timeseries.states:altitude", units="ft")

        mass_solved = prob.get_val("desc2.timeseries.states:mass", units="lbm")
        mass_simulate = simout5.get_val("desc2.timeseries.states:mass", units="lbm")

        distance_solved = prob.get_val("desc2.timeseries.states:distance", units="NM")
        distance_simulate = simout5.get_val(
            "desc2.timeseries.states:distance", units="NM"
        )

        mach_solved = prob.get_val("desc2.timeseries.mach", units="unitless")
        mach_simulate = simout5.get_val("desc2.timeseries.mach", units="unitless")

        TAS_solved = prob.get_val("desc2.timeseries.TAS", units="kn")
        TAS_simulate = simout5.get_val("desc2.timeseries.TAS", units="kn")

        gamma_solved = prob.get_val("desc2.timeseries.flight_path_angle", units="deg")
        gamma_simulate = simout5.get_val(
            "desc2.timeseries.flight_path_angle", units="deg")

        thrust_solved = prob.get_val("desc2.timeseries.thrust", units="unitless")
        thrust_simulate = simout5.get_val("desc2.timeseries.thrust", units="unitless")

        CD_solved = prob.get_val("desc2.timeseries.CD", units="unitless")
        CD_simulate = simout5.get_val("desc2.timeseries.CD", units="unitless")

        alpha_solved = prob.get_val("desc2.timeseries.alpha", units="deg")
        alpha_simulate = simout5.get_val("desc2.timeseries.alpha", units="deg")

        CL_solved = prob.get_val("desc2.timeseries.CL", units="unitless")
        CL_simulate = simout5.get_val("desc2.timeseries.CL", units="unitless")

        theta_solved = prob.get_val("desc2.timeseries.theta", units="deg")
        theta_simulate = simout5.get_val("desc2.timeseries.theta", units="deg")
        theta_max = np.ones(np.shape(theta_simulate)) * theta_max_val

        fig, ax = plt.subplots(3, 1, sharex=True)
        fig.suptitle("State Values")

        if plot_over_alt is False:
            l1 = ax[0].plot(t_solved, alt_solved, "go")
            l2 = ax[0].plot(t_simulate, alt_simulate, "g-")

            ax[1].plot(t_solved, mass_solved, "go")
            ax[1].plot(t_simulate, mass_simulate, "g-")

            ax[2].plot(t_solved, distance_solved, "go")
            ax[2].plot(t_simulate, distance_simulate, "g-")

            ax[0].set_xlabel("time (s)")
            ax[1].set_xlabel("time (s)")
            ax[2].set_xlabel("time (s)")
            ax[0].set_ylabel("alt (ft)")

            plt_name = "plots/desc2_states_time.pdf"

        else:
            l1 = ax[0].plot(alt_solved, t_solved, "go")
            l2 = ax[0].plot(alt_simulate, t_simulate, "g-")

            ax[1].plot(alt_solved, mass_solved, "go")
            ax[1].plot(alt_simulate, mass_simulate, "g-")

            ax[2].plot(alt_solved, distance_solved, "go")
            ax[2].plot(alt_simulate, distance_simulate, "g-")

            ax[0].set_xlabel("alt (ft)")
            ax[1].set_xlabel("alt (ft)")
            ax[2].set_xlabel("alt (ft)")
            ax[0].set_ylabel("time (s)")

            plt_name = "plots/desc2_states_alt.pdf"

        ax[1].set_ylabel("mass (lbm)")
        ax[2].set_ylabel("distance (NM)")
        fig.legend([l1, l2], labels=["Optimized", "Simulated"])
        plt.savefig(plt_name)

        fig2, ax2 = plt.subplots(2, 1, sharex=True)
        fig2.suptitle("Speed Values")

        if plot_over_alt is False:
            a1 = ax2[0].plot(t_solved, mach_solved, "ro")
            a2 = ax2[0].plot(t_simulate, mach_simulate, "r-")

            ax2[1].plot(t_solved, TAS_solved, "ro")
            ax2[1].plot(t_simulate, TAS_simulate, "r-")

            ax2[0].set_xlabel("time (s)")
            ax2[1].set_xlabel("time (s)")

            plt_name = "plots/desc2_speeds_time.pdf"

        else:
            a1 = ax2[0].plot(alt_solved, mach_solved, "ro")
            a2 = ax2[0].plot(alt_simulate, mach_simulate, "r-")

            ax2[1].plot(alt_solved, TAS_solved, "ro")
            ax2[1].plot(alt_simulate, TAS_simulate, "r-")

            ax2[0].set_xlabel("alt (ft)")
            ax2[1].set_xlabel("alt (ft)")

            plt_name = "plots/desc2_speeds_alt.pdf"

        ax2[0].set_ylabel(Dynamic.Mission.MACH)
        ax2[1].set_ylabel("TAS (knots)")
        fig2.legend([a1, a2], labels=["Optimized", "Simulated"])
        plt.savefig(plt_name)

        fig3, ax3 = plt.subplots(3, 1, sharex=True)
        fig3.suptitle("Force Values")

        if plot_over_alt is False:

            d1 = ax3[0].plot(t_solved, thrust_solved, "bo")
            d2 = ax3[0].plot(t_simulate, thrust_simulate, "b-")

            ax3[1].plot(t_solved, CD_solved, "bo")
            ax3[1].plot(t_simulate, CD_simulate, "b-")

            ax3[2].plot(t_solved, CL_solved, "bo")
            ax3[2].plot(t_simulate, CL_simulate, "b-")

            ax3[0].set_xlabel("time (s)")
            ax3[1].set_xlabel("time (s)")
            ax3[2].set_xlabel("time (s)")

            plt_name = "plots/desc2_forces_time.pdf"

        else:

            d1 = ax3[0].plot(alt_solved, thrust_solved, "bo")
            d2 = ax3[0].plot(alt_simulate, thrust_simulate, "b-")

            ax3[1].plot(alt_solved, CD_solved, "bo")
            ax3[1].plot(alt_simulate, CD_simulate, "b-")

            ax3[2].plot(alt_solved, CL_solved, "bo")
            ax3[2].plot(alt_simulate, CL_simulate, "b-")

            ax3[0].set_xlabel("alt (ft)")
            ax3[1].set_xlabel("alt (ft)")
            ax3[2].set_xlabel("alt (ft)")

            plt_name = "plots/desc2_forces_alt.pdf"

        ax3[0].set_ylabel("thrust (lbf)")
        ax3[1].set_ylabel("CD")
        ax3[2].set_ylabel("CL")
        fig3.legend([d1, d2], labels=["Optimized", "Simulated"])
        plt.savefig(plt_name)

        fig4, ax4 = plt.subplots(3, 1, sharex=True)
        fig4.suptitle("Angle Values")

        if plot_over_alt is False:

            ax4[0].plot(t_solved, alpha_solved, "go")
            ax4[0].plot(t_simulate, alpha_simulate, "g-")

            ax4[1].plot(t_solved, gamma_solved, "go")
            ax4[1].plot(t_simulate, gamma_simulate, "g-")

            c1 = ax4[2].plot(t_solved, theta_solved, "go")
            c2 = ax4[2].plot(t_simulate, theta_simulate, "g-")
            c3 = ax4[2].plot(t_simulate, theta_max, "r-")

            ax4[0].set_xlabel("time (s)")
            ax4[1].set_xlabel("time (s)")
            ax4[1].set_xlabel("time (s)")

            plt_name = "plots/desc2_angles_time.pdf"

        else:

            ax4[0].plot(alt_solved, alpha_solved, "go")
            ax4[0].plot(alt_simulate, alpha_simulate, "g-")

            ax4[1].plot(alt_solved, gamma_solved, "go")
            ax4[1].plot(alt_simulate, gamma_simulate, "g-")

            c1 = ax4[2].plot(alt_solved, theta_solved, "go")
            c2 = ax4[2].plot(alt_simulate, theta_simulate, "g-")
            c3 = ax4[2].plot(alt_simulate, theta_max, "r-")

            ax4[0].set_xlabel("alt (ft)")
            ax4[1].set_xlabel("alt (ft)")
            ax4[2].set_xlabel("alt (ft)")

            plt_name = "plots/desc2_angles_alt.pdf"

        ax4[0].set_ylabel("alpha (deg)")
        ax4[1].set_ylabel("gamma (deg)")
        ax4[2].set_ylabel("theta (deg)")
        fig4.legend(
            [c1, c2, c3], labels=["Optimized", "Simulated", "maximum"]
        )
        plt.savefig(plt_name)

        fig5, ax5 = plt.subplots(2, 1, sharex=True)
        fig5.suptitle("Alpha Values")

        f1 = ax5[0].plot(alpha_solved, CL_solved, "ro")

        ax5[1].plot(alpha_solved, CD_solved, "ro")

        ax5[0].set_xlabel("alpha (deg)")
        ax5[1].set_xlabel("alpha (deg)")

        ax5[0].set_ylabel("CL")
        ax5[1].set_ylabel("CD")
        fig5.legend([f1, f2], labels=["Optimized"])
        plt.savefig("plots/desc2_alphas.pdf")

    return prob
