import unittest

import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def make_groundroll_problem(optimizer='IPOPT', print_opt_iters=False, solve_segments=False):
    groundroll_trans = dm.Radau(
        num_segments=5, order=3, compressed=True, solve_segments=solve_segments
    )

    ode_args = dict(
        aviary_options=get_option_defaults(),
        core_subsystems=default_mission_subsystems
    )

    groundroll = dm.Phase(
        ode_class=FlightPathODE,
        ode_init_kwargs=dict(ode_args, ground_roll=True),
        transcription=groundroll_trans,
    )

    groundroll.set_time_options(fix_initial=True, fix_duration=False,
                                units="s", targets="t_curr",
                                duration_bounds=(20, 100), duration_ref=1)

    groundroll.set_state_options("TAS",
                                 fix_initial=True, fix_final=False, lower=1.0E-6, upper=1000, ref=1, defect_ref=1)

    groundroll.set_state_options("mass",
                                 fix_initial=True, fix_final=False, lower=1, upper=195_000, ref=1000, defect_ref=1000, rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL)

    groundroll.set_state_options(Dynamic.Mission.DISTANCE,
                                 fix_initial=True, fix_final=False, lower=0, upper=100_000, ref=1, defect_ref=1)

    groundroll.add_parameter("t_init_gear", units="s",
                             static_target=True, opt=False, val=32.3)
    groundroll.add_parameter("t_init_flaps", units="s",
                             static_target=True, opt=False, val=44.0)

    groundroll.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
    groundroll.add_timeseries_output("normal_force")
    groundroll.add_timeseries_output(Dynamic.Mission.MACH)
    groundroll.add_timeseries_output("EAS", units="kn")
    groundroll.add_timeseries_output(Dynamic.Mission.LIFT)
    groundroll.add_timeseries_output("CL")
    groundroll.add_timeseries_output("CD")
    groundroll.add_timeseries_output("fuselage_pitch", output_name="theta", units="deg")

    p = om.Problem()

    traj = p.model.add_subsystem("traj", dm.Trajectory())
    groundroll = traj.add_phase("groundroll", groundroll)

    groundroll.add_objective('time', loc='final', ref=1.0)

    p.model.add_subsystem("vrot_comp", VRotateComp())
    p.model.connect('traj.groundroll.states:mass',
                    'vrot_comp.mass', src_indices=om.slicer[0, ...])

    vrot_eq_comp = p.model.add_subsystem("vrot_eq_comp", om.EQConstraintComp())
    vrot_eq_comp.add_eq_output("v_rotate_error", eq_units="kn",
                               lhs_name="v_rot_computed", rhs_name="groundroll_v_final", add_constraint=True)

    p.model.connect('vrot_comp.Vrot', 'vrot_eq_comp.v_rot_computed')
    p.model.connect('traj.groundroll.states:TAS',
                    'vrot_eq_comp.groundroll_v_final', src_indices=om.slicer[-1, ...])

    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options["print_results"] = p.comm.rank == 0

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

    # TODO: paramport
    params = ParamPort.param_data
    p.set_val('vrot_comp.' + Aircraft.Wing.AREA,
              params[Aircraft.Wing.AREA]["val"], units=params[Aircraft.Wing.AREA]["units"])

    p.set_val("vrot_comp.dV1", val=10, units="kn")
    p.set_val("vrot_comp.dVR", val=5, units="kn")
    p.set_val("vrot_comp.rho", val=RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3")
    p.set_val("vrot_comp.CL_max", val=2.1886, units="unitless")

    p.set_val("traj.groundroll.states:TAS",
              groundroll.interp("TAS", [0, 146]),
              units="kn")

    p.set_val(
        "traj.groundroll.states:mass",
        groundroll.interp("mass", [175100, 174000]),
        units="lbm",
    )

    p.set_val(
        "traj.groundroll.states:distance",
        groundroll.interp(Dynamic.Mission.DISTANCE, [0, 3000]),
        units="ft",
    )

    p.set_val("traj.groundroll.t_duration", 20.0)

    return p


@use_tempdirs
class TestGroundRoll(unittest.TestCase):

    # @require_pyoptsparse(optimizer='IPOPT')
    # def benchmark_groundroll_result_ipopt(self):
    #     p = make_groundroll_problem(optimizer='IPOPT')
    #     dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)

    #     tf = p.get_val('traj.groundroll.timeseries.time', units='s')[-1, 0]
    #     vrot = p.get_val('vrot_comp.Vrot', units='kn')[0]

    #     print(f't_final: {tf:8.3f} s')
    #     print(f'v_rotate: {vrot:8.3f} knots')

    #     assert_near_equal(tf, 29.6, tolerance=0.01)
    #     assert_near_equal(vrot, 146.4, tolerance=0.02)

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_groundroll_result_ipopt_solve(self):
        p = make_groundroll_problem(optimizer='IPOPT', solve_segments='forward')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)

        tf = p.get_val('traj.groundroll.timeseries.time', units='s')[-1, 0]
        vrot = p.get_val('vrot_comp.Vrot', units='kn')[0]

        print(f't_final: {tf:8.3f} s')
        print(f'v_rotate: {vrot:8.3f} knots')

        assert_near_equal(tf, 30.99546852, tolerance=0.01)
        assert_near_equal(vrot, 146.4, tolerance=0.02)

    # @require_pyoptsparse(optimizer='SNOPT')
    # def benchmark_groundroll_result_snopt(self):
    #     p = make_groundroll_problem(optimizer='SNOPT')
    #     dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)

    #     tf = p.get_val('traj.groundroll.timeseries.time', units='s')[-1, 0]
    #     vrot = p.get_val('vrot_comp.Vrot', units='kn')[0]

    #     print(f't_final: {tf:8.3f} s')
    #     print(f'v_rotate: {vrot:8.3f} knots')

    #     assert_near_equal(tf, 29.6, tolerance=0.01)
    #     assert_near_equal(vrot, 146.4, tolerance=0.02)
