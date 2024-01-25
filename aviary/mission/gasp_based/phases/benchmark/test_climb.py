import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def make_climb_problem(
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
        prob.driver.opt_settings[
            "nlp_scaling_method"
        ] = "gradient-based"  # for faster convergence
        prob.driver.opt_settings["alpha_for_y"] = "safer-min-dual-infeas"
        prob.driver.opt_settings["mu_strategy"] = "monotone"
        prob.driver.opt_settings["bound_mult_init_method"] = "mu-based"

    transcription = dm.Radau(num_segments=11, order=3, compressed=True)

    ode_args = dict(
        clean=True,
        aviary_options=get_option_defaults(),
        core_subsystems=default_mission_subsystems
    )

    climb = dm.Phase(ode_class=FlightPathODE, transcription=transcription,
                     ode_init_kwargs=ode_args)

    climb.set_time_options(fix_initial=True, duration_bounds=(
        10, 5000), units="s", duration_ref=200)

    climb.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        units="kn",
        ref=250,
        defect_ref=250)

    climb.set_state_options(
        "mass",
        fix_initial=True,
        fix_final=False,
        lower=1,
        upper=None,
        units="lbm",
        ref=200000,
        defect_ref=200000,
        rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL)

    climb.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=True,
        fix_final=False,
        lower=0,
        upper=None,
        units="ft",
        ref=10000,
        defect_ref=10000)

    if constant_quantity == 'EAS':
        climb.set_state_options(
            Dynamic.Mission.ALTITUDE,
            fix_initial=True,
            fix_final=False,
            lower=400,
            upper=20000,
            units="ft",
            ref=20.e3,
            defect_ref=20.e3)
    else:
        climb.set_state_options(
            Dynamic.Mission.ALTITUDE,
            fix_initial=True,
            fix_final=False,
            lower=9000,
            upper=50000,
            units="ft",
            ref=40000,
            defect_ref=40000)

    climb.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        units="rad",
        ref=1.0,
        defect_ref=1.0)

    # climb.add_polynomial_control(
    #     "alpha", order=3, opt=True, units="rad", val=0.0, lower=np.radians(-4), upper=np.radians(14))
    climb.add_control("alpha",
                      opt=True,
                      units="rad",
                      val=0.0,
                      lower=np.radians(-14),
                      upper=np.radians(14),
                      rate_continuity=True,
                      rate2_continuity=False)

    if constant_quantity == 'EAS':
        climb.add_path_constraint("EAS", lower=249.9, upper=250.1, ref=250., units="kn")
        # climb.add_path_constraint("EAS", equals=350., ref=350., units="kn")
        climb.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE, loc="final", equals=10.e3, ref=10.e3, units="ft")
    else:
        climb.add_path_constraint(Dynamic.Mission.MACH,
                                  lower=0.799, upper=0.801)
        climb.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE, loc="final", equals=37.5e3, ref=40.e3, units="ft")

    throttle_climb = 0.956
    climb.add_parameter(
        Dynamic.Mission.THROTTLE, opt=False, units="unitless", val=throttle_climb, static_target=False
    )

    climb.add_parameter(
        Aircraft.Wing.AREA, opt=False, units="ft**2", val=1370.0, static_target=True
    )

    climb.add_timeseries_output("EAS", output_name="EAS", units="kn")
    climb.add_timeseries_output(
        Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
    climb.add_timeseries_output("alpha", output_name="alpha", units="deg")
    climb.add_timeseries_output("CL", output_name="CL", units="unitless")
    climb.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL,
                                output_name=Dynamic.Mission.THRUST_TOTAL, units="lbf")
    climb.add_timeseries_output("CD", output_name="CD", units="unitless")

    traj = prob.model.add_subsystem('traj', dm.Trajectory())
    traj.add_phase('climb', climb)

    climb.add_objective("mass", loc="final", ref=-1.e4)
    # climb.add_objective("time", loc="final", ref=1.e2)

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

        prob.setup()

        if constant_quantity == 'EAS':
            prob.set_val("traj.climb.states:TAS", climb.interp(
                "TAS", ys=[250., 250.]), units='kn')
            prob.set_val("traj.climb.states:altitude", climb.interp(
                Dynamic.Mission.ALTITUDE, ys=[500., 10.e3]), units='ft')
            prob.set_val("traj.climb.states:flight_path_angle", climb.interp(
                Dynamic.Mission.FLIGHT_PATH_ANGLE, ys=[0, 1]), units='deg')
            prob.set_val("traj.climb.states:mass", climb.interp(
                "mass", ys=[174219, 170000]), units='lbm')
        else:
            prob.set_val("traj.climb.states:TAS", climb.interp(
                "TAS", ys=[490.5, 500]), units='kn')
            prob.set_val("traj.climb.states:altitude", climb.interp(
                Dynamic.Mission.ALTITUDE, ys=[20.e3, 37.5e3]), units='ft')
            prob.set_val("traj.climb.states:flight_path_angle", climb.interp(
                Dynamic.Mission.FLIGHT_PATH_ANGLE, ys=[1, 0.5]), units='deg')
            prob.set_val("traj.climb.states:mass", climb.interp(
                "mass", ys=[172.83e3, 171.e3]), units='lbm')

        prob.set_val("traj.climb.states:distance", climb.interp(
            Dynamic.Mission.DISTANCE, ys=[15, 154]), units='NM')
        prob.set_val("traj.climb.controls:alpha",
                     climb.interp("alpha", ys=[3, 2]), units='deg')
        prob.set_val("traj.climb.t_duration", 200, units='s')
        prob.set_val("traj.climb.t_initial", 0, units='s')

        return prob


@use_tempdirs
class Testclimb(unittest.TestCase):

    def assert_constant_EAS_result(self, p):
        final_mass = p.get_val("traj.climb.timeseries.mass", units="lbm")[-1, ...]
        assert_near_equal(final_mass, 173_558, tolerance=0.0001)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_constant_EAS_climb_ipopt(self):
        p = make_climb_problem(optimizer='IPOPT', constant_quantity='EAS')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        self.assert_constant_EAS_result(p)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_constant_EAS_climb_snopt(self):
        p = make_climb_problem(
            optimizer='SNOPT', constant_quantity='EAS', print_opt_iters=True)
        dm.run_problem(p, run_driver=True, simulate=True, make_plots=False)
        self.assert_constant_EAS_result(p)

    def assert_constant_mach_result(self, p):
        final_mass = p.get_val("traj.climb.timeseries.mass", units="lbm")[-1, ...]
        assert_near_equal(final_mass, 171_073, tolerance=0.0001)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_constant_mach_climb_ipopt(self):
        p = make_climb_problem(optimizer='IPOPT', constant_quantity=Dynamic.Mission.MACH)
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        self.assert_constant_mach_result(p)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_constant_mach_climb_snopt(self):
        p = make_climb_problem(optimizer='SNOPT', constant_quantity=Dynamic.Mission.MACH)
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        self.assert_constant_mach_result(p)


if __name__ == "__main__":
    unittest.main()
