import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from packaging import version

from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def make_rotation_problem(optimizer='IPOPT', print_opt_iters=False):
    #
    # ROTATION TO TAKEOFF
    #
    # Initial States Fixed
    # Final States Free
    #
    # Controls:
    #   None
    #
    # Boundary Constraints:
    #   normal_force(final) = 0 lbf
    #
    # Path Constraints:
    #   None
    #

    rotation_trans = dm.Radau(
        num_segments=5, order=3, compressed=True, solve_segments=False
    )

    ode_args = dict(
        aviary_options=get_option_defaults(),
        core_subsystems=default_mission_subsystems
    )

    rotation = dm.Phase(
        ode_class=FlightPathODE,
        # Use the standard ode_args and update it for ground_roll dynamics
        ode_init_kwargs=dict(ode_args, ground_roll=True),
        transcription=rotation_trans,
    )

    rotation.set_time_options(
        fix_initial=True,
        fix_duration=False,
        units="s",
        targets="t_curr",
        duration_bounds=(1, 100),
        duration_ref=1.0,
    )

    rotation.add_parameter('alpha_rate', opt=False,
                           static_target=True, val=3.33, units='deg/s')
    rotation.add_parameter("t_init_gear", units="s",
                           static_target=True, opt=False, val=32.3)
    rotation.add_parameter("t_init_flaps", units="s",
                           static_target=True, opt=False, val=44.0)
    rotation.add_parameter("wing_area", units="ft**2",
                           static_target=True, opt=False, val=1370)

    # State alpha is not defined in the ODE, taken from the parameter "alpha_rate"
    rotation.add_state("alpha", units="rad", rate_source="alpha_rate",
                       fix_initial=True, fix_final=False, lower=0.0, upper=np.radians(25), ref=1.0)

    rotation.set_state_options("TAS",
                               fix_initial=True, fix_final=False, lower=0.0, upper=1000.0, ref=100.0, defect_ref=100.0)

    rotation.set_state_options("mass",
                               fix_initial=True, fix_final=False, lower=1.0, upper=190_000.0, ref=1000.0,
                               defect_ref=1000.0, rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL,
                               )

    rotation.set_state_options(Dynamic.Mission.DISTANCE,
                               fix_initial=True, fix_final=False, lower=0, upper=10.e3, ref=100, defect_ref=100)

    # boundary/path constraints + controls
    rotation.add_boundary_constraint(
        "normal_force", loc="final", equals=0, units="lbf", ref=1000.0)

    rotation.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
    rotation.add_timeseries_output("normal_force")
    rotation.add_timeseries_output(Dynamic.Mission.MACH)
    rotation.add_timeseries_output("EAS", units="kn")
    rotation.add_timeseries_output(Dynamic.Mission.LIFT)
    rotation.add_timeseries_output("CL")
    rotation.add_timeseries_output("CD")
    rotation.add_timeseries_output("fuselage_pitch", output_name="theta", units="deg")

    p = om.Problem()
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("rotation", rotation)

    rotation.timeseries_options['use_prefix'] = True

    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = optimizer

    if optimizer == 'SNOPT':
        if print_opt_iters:
            p.driver.opt_settings["iSumm"] = 6
        p.driver.opt_settings['Major step limit'] = 1
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
    elif optimizer == 'IPOPT':
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['tol'] = 1.0E-6
        p.driver.opt_settings['mu_init'] = 1e-3
        p.driver.opt_settings['max_iter'] = 100
        if print_opt_iters:
            p.driver.opt_settings['print_level'] = 5
        # for faster convergence
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'

    p.model.add_objective(
        "traj.rotation.timeseries.states:mass", index=-1, ref=1.4e5, ref0=1.3e5
    )
    p.set_solver_print(level=-1)

    p.setup()

    # SET INITIAL ALPHA INITIAL GUESS
    p.set_val("traj.rotation.states:alpha",
              rotation.interp("alpha", [0, 2.5]), units="deg")

    # SET TRUE AIRSPEED INITIAL GUESS
    p.set_val("traj.rotation.states:TAS", 143, units="kn")

    # SET MASS INITIAL GUESS
    p.set_val("traj.rotation.states:mass", rotation.interp(
        "mass", [174975.12776915, 174900]), units="lbm")

    # SET RANGE INITIAL GUESS
    p.set_val("traj.rotation.states:distance", rotation.interp(
        Dynamic.Mission.DISTANCE, [3680.37217765, 4000]), units="ft")

    # SET TIME INITIAL GUESS
    p.set_val("traj.rotation.t_initial", 30.0)
    p.set_val("traj.rotation.t_duration", 4.0)

    return p


@use_tempdirs
class TestRotation(unittest.TestCase):

    def assert_result(self, p):
        tf = p.get_val('traj.rotation.timeseries.time', units='s')[-1, 0]
        vf = p.get_val('traj.rotation.timeseries.states:TAS', units='kn')[-1, 0]

        print(f't_final: {tf:8.3f} s')
        print(f'v_final: {vf:8.3f} knots')

        assert_near_equal(tf, 32.993, tolerance=0.01)
        assert_near_equal(vf, 155.246, tolerance=0.01)

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_rotation_result_ipopt(self):
        p = make_rotation_problem(optimizer='IPOPT')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False,
                       solution_record_file='rotation_IPOPT.db')
        self.assert_result(p)

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_rotation_result_snopt(self):
        p = make_rotation_problem(optimizer='SNOPT')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False,
                       solution_record_file='rotation_SNOPT.db')
        self.assert_result(p)
