import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def make_accel_problem(
        optimizer='IPOPT',
        print_opt_iters=False,
        constant_quantity=Dynamic.Mission.MACH):
    prob = om.Problem(model=om.Group())

    prob.driver = om.pyOptSparseDriver()

    if optimizer == "SNOPT":
        prob.driver.options["optimizer"] = "SNOPT"
        prob.driver.opt_settings['Major iterations limit'] = 100
        # prob.driver.opt_settings['Major step limit'] = 0.05
        prob.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        prob.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        if print_opt_iters:
            prob.driver.opt_settings["iSumm"] = 6
    elif optimizer == "IPOPT":
        prob.driver.options["optimizer"] = "IPOPT"
        prob.driver.opt_settings["max_iter"] = 500
        prob.driver.opt_settings["tol"] = 1e-5
        prob.driver.opt_settings["print_level"] = 5
        # prob.driver.opt_settings["nlp_scaling_method"] = "gradient-based"
        prob.driver.opt_settings["alpha_for_y"] = "safer-min-dual-infeas"
        prob.driver.opt_settings["mu_strategy"] = "monotone"
        prob.driver.opt_settings["bound_mult_init_method"] = "mu-based"

    transcription = dm.Radau(num_segments=5, order=3, compressed=True)

    ode_args = dict(
        clean=True,
        aviary_options=get_option_defaults(),
        core_subsystems=default_mission_subsystems
    )

    accel = dm.Phase(ode_class=FlightPathODE, transcription=transcription,
                     ode_init_kwargs=ode_args)

    accel.set_time_options(fix_initial=True, duration_bounds=(
        10, 5000), units="s", duration_ref=100)

    accel.set_state_options("TAS",
                            fix_initial=True, fix_final=False, lower=1, upper=1000, units="kn", ref=250, defect_ref=250)

    accel.set_state_options("mass",
                            fix_initial=True, fix_final=False, lower=1, upper=None, units="lbm", ref=200000, defect_ref=200000, rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL)

    accel.set_state_options(Dynamic.Mission.DISTANCE,
                            fix_initial=True, fix_final=False, lower=0, upper=None, units="ft", ref=10000, defect_ref=10000)

    accel.set_state_options(Dynamic.Mission.ALTITUDE,
                            fix_initial=True, fix_final=False, lower=490, upper=510, units="ft", ref=100, defect_ref=100)

    accel.set_state_options(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                            fix_initial=True, fix_final=False, units="rad", lower=0.0, ref=1.0, defect_ref=1.0)

    accel.add_control("alpha", opt=True, units="rad", val=0.0,
                      lower=np.radians(-14), upper=np.radians(14), rate_continuity=True, rate2_continuity=False)

    accel.add_boundary_constraint(
        "EAS", loc="final", equals=250, ref=250, units="kn")

    throttle_climb = 0.956
    accel.add_parameter(
        Dynamic.Mission.THROTTLE,
        opt=False,
        units="unitless",
        val=throttle_climb,
        static_target=False)

    accel.add_parameter(
        Aircraft.Wing.AREA, opt=False, units="ft**2", val=1370.0, static_target=True
    )

    accel.add_timeseries_output("EAS", output_name="EAS", units="kn")
    accel.add_timeseries_output(
        Dynamic.Mission.MACH,
        output_name=Dynamic.Mission.MACH,
        units="unitless")
    accel.add_timeseries_output("alpha", output_name="alpha", units="deg")
    accel.add_timeseries_output("CL", output_name="CL", units="unitless")
    accel.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL,
                                output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
    accel.add_timeseries_output("CD", output_name="CD", units="unitless")

    traj = prob.model.add_subsystem('traj', dm.Trajectory())
    traj.add_phase('accel', accel)

    accel.add_objective("mass", loc="final", ref=-1.e4)
    # accel.add_objective("time", loc="final", ref=1.e2)

    prob.model.linear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["iprint"] = 0

    traj.nonlinear_solver = om.NonlinearBlockGS(iprint=0)
    traj.options['assembled_jac_type'] = 'csc'
    traj.linear_solver = om.DirectSolver()

    prob.set_solver_print(level=0)

    for phase_name, phase in traj._phases.items():
        phase.add_timeseries_output('normal_force')
        phase.add_timeseries_output('fuselage_pitch')
        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE, units='ft')
        phase.add_timeseries_output('mass', units='lbm')
        phase.add_timeseries_output(Dynamic.Mission.LIFT, units='lbf')
        phase.add_timeseries_output(Dynamic.Mission.FLIGHT_PATH_ANGLE, units='deg')
        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units='lbf')
        phase.add_timeseries_output('alpha', units='deg')

    prob.setup()

    prob.set_val("traj.accel.states:TAS", accel.interp(
        "TAS", ys=[200., 250.]), units='kn')
    prob.set_val("traj.accel.states:altitude", accel.interp(
        Dynamic.Mission.ALTITUDE, ys=[500., 500.]), units='ft')
    prob.set_val("traj.accel.states:flight_path_angle", accel.interp(
        Dynamic.Mission.FLIGHT_PATH_ANGLE, ys=[0, 0]), units='deg')
    prob.set_val("traj.accel.states:mass", accel.interp(
        "mass", ys=[174219, 170000]), units='lbm')

    prob.set_val("traj.accel.states:distance", accel.interp(
        Dynamic.Mission.DISTANCE, ys=[0, 154]), units='NM')
    prob.set_val("traj.accel.t_duration", 10, units='s')
    prob.set_val("traj.accel.t_initial", 0, units='s')

    return prob


@use_tempdirs
class Testaccel(unittest.TestCase):

    def assert_constant_EAS_result(self, p):
        final_mass = p.get_val(
            "traj.accel.timeseries.mass", units="lbm")[-1, ...]
        assert_near_equal(final_mass, 174.15124423509e3, tolerance=0.0001)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_accel_ipopt(self):
        p = make_accel_problem(optimizer='IPOPT', constant_quantity='EAS')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)

        lift = p.get_val("traj.accel.timeseries.lift")
        weight = p.get_val("traj.accel.timeseries.mass") * GRAV_ENGLISH_LBM
        gamma = p.get_val("traj.accel.timeseries.flight_path_angle", units="rad")
        thrust = p.get_val("traj.accel.timeseries.thrust_net_total")
        alpha = p.get_val("traj.accel.timeseries.alpha", units="rad")

        alpha_constant_gamma = np.arcsin((lift - weight * np.cos(gamma)) / thrust)

        with np.printoptions(linewidth=1024):
            print(lift.T)
            print(weight.T)
            print(gamma.T)
            print(thrust.T)
            print(alpha.T)
            print(alpha_constant_gamma.T)
        self.assert_constant_EAS_result(p)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_accel_snopt(self):
        p = make_accel_problem(
            optimizer='SNOPT', constant_quantity='EAS', print_opt_iters=True)
        dm.run_problem(p, run_driver=True, simulate=True, make_plots=False)
        self.assert_constant_EAS_result(p)
