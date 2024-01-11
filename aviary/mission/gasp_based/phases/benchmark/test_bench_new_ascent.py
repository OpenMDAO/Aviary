import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def make_ascent_problem(optimizer='IPOPT', print_opt_iters=False):

    ode_args = dict(
        aviary_options=get_option_defaults(),
        core_subsystems=default_mission_subsystems
    )

    #
    # ASCENT TO GEAR RETRACTION
    #
    # Initial States Fixed
    # Final States Free
    #
    # Controls:
    #   alpha
    #
    # Boundary Constraints:
    #   alt(final) = 50 ft
    #
    # Path Constraints:
    #   0.0 < load_factor < 1.10
    #   0.0 < fuselage_pitch < 15 deg
    #
    ascent0_tx = dm.Radau(num_segments=5, order=3, compressed=True, solve_segments=False)
    ascent_to_gear_retract = dm.Phase(
        ode_class=FlightPathODE, ode_init_kwargs=ode_args, transcription=ascent0_tx)

    ascent_to_gear_retract.set_time_options(
        units="s",    targets="t_curr",    fix_initial=True,
        duration_bounds=(0, 10))

    # Rate sources and units of states are set with tags in AscentEOM
    ascent_to_gear_retract.set_state_options(Dynamic.Mission.FLIGHT_PATH_ANGLE, units="rad",
                                             fix_initial=True, fix_final=False, lower=0, upper=0.30, ref=0.1, defect_ref=0.1)

    ascent_to_gear_retract.set_state_options(Dynamic.Mission.ALTITUDE,
                                             fix_initial=True, fix_final=False, lower=0,       upper=500,     ref=100,     defect_ref=100)

    ascent_to_gear_retract.set_state_options("TAS",
                                             fix_initial=True, fix_final=False, lower=0,       upper=500,    ref=100,     defect_ref=100)

    ascent_to_gear_retract.set_state_options("mass",
                                             fix_initial=True, fix_final=False, lower=100.e3,       upper=None,    ref=175_000, defect_ref=175_000, rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL)

    ascent_to_gear_retract.set_state_options(Dynamic.Mission.DISTANCE,
                                             fix_initial=True, fix_final=False, lower=0,       upper=15_000,  ref=5280,    defect_ref=5280)

    # Note that t_init_gear and t_init_flaps are taking values of 100 s here while not being connected to the
    # those trajectory-level parameters. This means that we'll have some small discontinuity in the aerodynamics
    # across these phases, but making the assumption of time here removes the feedback loop, removes the need for
    # a nonlinear solver on the trajectory, and makes things happen faster.
    ascent_to_gear_retract.add_parameter(
        "t_init_gear", units="s", static_target=True, opt=False, val=100.0)
    ascent_to_gear_retract.add_parameter(
        "t_init_flaps", units="s", static_target=True, opt=False, val=100.0)
    ascent_to_gear_retract.add_parameter(
        "wing_area", units="ft**2", val=1370, static_target=True, opt=False)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are in degrees.
    # ascent_to_gear_retract.add_control("alpha",
    #     units="rad", val=0.0, continuity=True, rate_continuity=False,
    #     opt=True, lower=np.radians(-30), upper=np.radians(30), ref=0.01)
    ascent_to_gear_retract.add_polynomial_control("alpha", order=1,
                                                  units="rad", val=0.0,  opt=True, lower=np.radians(-14), upper=np.radians(14), ref=0.1)

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of altitude is a design variable.
    ascent_to_gear_retract.add_boundary_constraint(Dynamic.Mission.ALTITUDE,
                                                   loc="final", equals=50, units="ft", ref=1.0, ref0=0.0, linear=True
                                                   )

    # Load factor can be treated as a linear constraint as long as i_wing is not a design variable.
    ascent_to_gear_retract.add_path_constraint("load_factor",
                                               lower=0.0, upper=1.10, linear=False)

    ascent_to_gear_retract.add_path_constraint("fuselage_pitch", constraint_name="theta",
                                               lower=0, upper=15, units="deg", ref=15, linear=False)

    #
    # ASCENT TO FLAP RETRACTION
    #
    # Initial States Free
    # Final States Free
    #
    # Controls:
    #   alpha
    #
    # Boundary Constraints:
    #   alt(final) = 400 ft
    #
    # Path Constraints:
    #   0.0 < load_factor < 1.10
    #   0.0 < fuselage_pitch < 15 deg
    #
    ascent1_tx = dm.Radau(num_segments=5, order=3, compressed=True, solve_segments=False)
    ascent_to_flap_retract = dm.Phase(
        ode_class=FlightPathODE, ode_init_kwargs=ode_args, transcription=ascent1_tx)

    ascent_to_flap_retract.set_time_options(
        units="s",    targets="t_curr",    fix_initial=False,
        initial_bounds=(5, 50), duration_bounds=(5, 50))

    # Rate sources and units of states are set with tags in AscentEOM
    ascent_to_flap_retract.set_state_options(Dynamic.Mission.FLIGHT_PATH_ANGLE, units="rad",
                                             fix_initial=False, fix_final=False, lower=0, upper=0.5, ref=0.1, defect_ref=0.1)

    ascent_to_flap_retract.set_state_options(Dynamic.Mission.ALTITUDE,
                                             fix_initial=False, fix_final=False, lower=0,       upper=500,     ref=100,     defect_ref=100)

    ascent_to_flap_retract.set_state_options("TAS",
                                             fix_initial=False, fix_final=False, lower=0,       upper=1000,    ref=100,     defect_ref=100)

    ascent_to_flap_retract.set_state_options("mass",
                                             fix_initial=False, fix_final=False, lower=1,       upper=None,    ref=175_000, defect_ref=175_000, rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL)

    ascent_to_flap_retract.set_state_options(Dynamic.Mission.DISTANCE,
                                             fix_initial=False, fix_final=False, lower=0,       upper=15_000,  ref=5280,    defect_ref=5280)

    # Targets are not needed when there is a top-level ODE input with the same name as the parameter, state, or control
    ascent_to_flap_retract.add_parameter(
        "t_init_gear", units="s", static_target=True, opt=False, val=100)
    ascent_to_flap_retract.add_parameter(
        "t_init_flaps", units="s", static_target=True, opt=False, val=100)
    ascent_to_flap_retract.add_parameter(
        "wing_area", units="ft**2", val=1370, static_target=True, opt=False)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are in degrees.
    ascent_to_flap_retract.add_polynomial_control("alpha", order=1,
                                                  units="rad", val=0.0,
                                                  opt=True, lower=np.radians(-14), upper=np.radians(14), ref=0.01)

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of altitude is a design variable.
    ascent_to_flap_retract.add_boundary_constraint(Dynamic.Mission.ALTITUDE,
                                                   loc="final", equals=400, units="ft", ref=1.0, ref0=0.0, linear=True
                                                   )

    # Load factor can be treated as a linear constraint as long as i_wing is not a design variable.
    ascent_to_flap_retract.add_path_constraint("load_factor",
                                               lower=0.0, upper=1.10, linear=False)

    ascent_to_flap_retract.add_path_constraint("fuselage_pitch", constraint_name="theta",
                                               lower=0, upper=15, units="deg", ref=15, linear=False)

    #
    # ASCENT TO CLEAN AERO CONFIG
    #
    # Initial States Free
    # Final States Free
    #
    # Controls:
    #   alpha
    #
    # Boundary Constraints:
    #   alt(final) = 500 ft
    #   flap_factor(final) = 0 (fully retracted)
    #
    # Path Constraints:
    #   0.0 < load_factor < 1.10
    #   0.0 < fuselage_pitch < 15 deg
    #
    ascent2_tx = dm.Radau(num_segments=5, order=3, compressed=True, solve_segments=False)
    ascent_to_clean_aero = dm.Phase(
        ode_class=FlightPathODE, ode_init_kwargs=ode_args, transcription=ascent2_tx)

    ascent_to_clean_aero.set_time_options(
        units="s",    targets="t_curr",    fix_initial=False,
        initial_bounds=(10, 100), duration_bounds=(1, 50))

    # Rate sources and units of states are set with tags in AscentEOM
    ascent_to_clean_aero.set_state_options(Dynamic.Mission.FLIGHT_PATH_ANGLE, units="rad",
                                           fix_initial=False, fix_final=False, lower=0, upper=0.5, ref=0.1, defect_ref=0.1)

    ascent_to_clean_aero.set_state_options(Dynamic.Mission.ALTITUDE,
                                           fix_initial=False, fix_final=False, lower=0,       upper=500,     ref=100,     defect_ref=100)

    ascent_to_clean_aero.set_state_options("TAS",
                                           fix_initial=False, fix_final=False, lower=0,       upper=1000,    ref=100,     defect_ref=100)

    ascent_to_clean_aero.set_state_options("mass",
                                           fix_initial=False, fix_final=False, lower=1,       upper=None,    ref=175_000, defect_ref=175_000, rate_source=Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL)

    ascent_to_clean_aero.set_state_options(Dynamic.Mission.DISTANCE,
                                           fix_initial=False, fix_final=False, lower=0,       upper=15_000,  ref=5_280,    defect_ref=5_280)

    # Targets are not needed when there is a top-level ODE input with the same name as the parameter, state, or control
    ascent_to_clean_aero.add_parameter(
        "t_init_gear", units="s", static_target=True, opt=False)
    ascent_to_clean_aero.add_parameter(
        "t_init_flaps", units="s", static_target=True, opt=False)
    ascent_to_clean_aero.add_parameter(
        "wing_area", units="ft**2", val=1370, static_target=True, opt=False)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are in degrees.
    ascent_to_clean_aero.add_polynomial_control("alpha", order=1,
                                                units="rad", val=0.0,  # continuity=True, rate_continuity=True,
                                                opt=True, lower=np.radians(-14), upper=np.radians(14), ref=0.01)

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of altitude is a design variable.
    ascent_to_clean_aero.add_boundary_constraint(Dynamic.Mission.ALTITUDE,
                                                 loc="final", equals=500, units="ft", ref=1.0, ref0=0.0, linear=True
                                                 )

    # Ensure gear and flaps are fully retracted
    # Note setting equals=0.0 here will result in a failure of the optimization, since flap_factor approaches
    # zero asymptotically.
    ascent_to_clean_aero.add_boundary_constraint("gear_factor",
                                                 loc="final", upper=1.0E-3, ref=1.0, ref0=0.0, linear=False
                                                 )

    ascent_to_clean_aero.add_boundary_constraint("flap_factor",
                                                 loc="final", upper=1.0E-3, ref=1.0, ref0=0.0, linear=False
                                                 )

    # Load factor can be treated as a linear constraint as long as i_wing is not a design variable.
    ascent_to_clean_aero.add_path_constraint("load_factor",
                                             lower=0.0, upper=1.10, linear=False)

    ascent_to_clean_aero.add_path_constraint("fuselage_pitch", constraint_name="theta",
                                             lower=0, upper=15, units="deg", ref=15, linear=False)

    #
    # PROBLEM DEFINITION
    #
    p = om.Problem()
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("ascent_to_gear_retract", ascent_to_gear_retract)
    traj.add_phase("ascent_to_flap_retract", ascent_to_flap_retract)
    traj.add_phase("ascent_to_clean_aero", ascent_to_clean_aero)

    traj.add_parameter('wing_area', units='ft**2', static_target=True, opt=False)

    # Inter-phase connections
    # 1. Connect final time of ascent_to_gear_retract to the trajectory as t_init_gear
    # traj.connect(src_name='ascent_to_gear_retract.timeseries.time',
    #              tgt_name=('ascent_to_flap_retract.parameters:t_init_gear',
    #                        'ascent_to_clean_aero.parameters:t_init_gear'),
    #              src_indices=om.slicer[-1, ...])

    # 2. Connect final time of ascent_to_flap_retract to the trajectory as t_init_flaps
    traj.connect(src_name='ascent_to_flap_retract.timeseries.time',
                 tgt_name='ascent_to_clean_aero.parameters:t_init_flaps',
                 src_indices=om.slicer[-1, ...])

    # 3. Enforce value continuity between all phases in ascent for time, states, and alpha control
    traj.link_phases(['ascent_to_gear_retract', 'ascent_to_flap_retract', 'ascent_to_clean_aero'],
                     vars=['time', Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.ALTITUDE, 'TAS', Dynamic.Mission.MASS, Dynamic.Mission.DISTANCE, 'alpha'])

    traj.add_linkage_constraint(phase_a='ascent_to_gear_retract',
                                phase_b='ascent_to_flap_retract',
                                var_a='time',
                                var_b='t_init_gear',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='ascent_to_gear_retract',
                                phase_b='ascent_to_clean_aero',
                                var_a='time',
                                var_b='t_init_gear',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    # traj.nonlinear_solver = om.NonlinearBlockGS(iprint=0)
    # traj.options['assembled_jac_type'] = 'csc'
    # traj.linear_solver = om.DirectSolver()

    for phase_name, phase in traj._phases.items():
        phase.add_timeseries_output('flap_factor')
        phase.add_timeseries_output('normal_force')
        # Add alpha to the timeseries as 'alpha' regardless of whether it is a control or polynomial control.
        phase.add_timeseries_output('alpha', units='deg')

    p.driver = om.pyOptSparseDriver()

    p.driver.options["optimizer"] = optimizer

    if optimizer == 'SNOPT':
        p.driver.opt_settings["iSumm"] = 6
        p.driver.opt_settings['Major step limit'] = 1
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
    elif optimizer == 'IPOPT':
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['tol'] = 1.0E-6
        p.driver.opt_settings['mu_init'] = 1e-5
        p.driver.opt_settings['max_iter'] = 100
        p.driver.opt_settings['print_level'] = 5
        # for faster convergence
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'

    # p.driver.options['debug_print'] = ['desvars']

    # Uncomment declare_coloring when the coloring needs to be reset (number of nodes have changed, constraints changed,
    # or underlying models have changed)
    # p.driver.use_fixed_coloring()
    # p.model.use_fixed_coloring(recurse=True)
    p.driver.declare_coloring(tol=1.0E-12)

    # p.model.add_objective("traj.ascent_to_gear_retract.timeseries.states:mass", index=-1, ref=1.4e5, ref0=1.3e5)
    ascent_to_clean_aero.add_objective("mass", loc="final", ref=-1.e3)

    # p.model.connect('wing_area','traj.ascent_to_gear_retract.parameters:wing_area')
    # p.model.set_input_defaults("gross_wt_initial", val=174000, units="lbf")
    p.setup()

    # SET INITIAL TIMES GUESSES
    # p.set_val('traj.parameters:t_init_gear', 40.0, units='s')
    # p.set_val('traj.parameters:t_init_flaps', 47.5, units='s')
    p.set_val('traj.parameters:wing_area', 1370, units='ft**2')

    # SET ALTITUDE GUESSES
    p.set_val("traj.ascent_to_gear_retract.states:altitude",
              ascent_to_gear_retract.interp(Dynamic.Mission.ALTITUDE, ys=[0, 50]), units="ft")
    p.set_val("traj.ascent_to_flap_retract.states:altitude",
              ascent_to_flap_retract.interp(Dynamic.Mission.ALTITUDE, ys=[50, 400]), units="ft")
    p.set_val("traj.ascent_to_clean_aero.states:altitude",
              ascent_to_clean_aero.interp(Dynamic.Mission.ALTITUDE, ys=[400, 500]), units="ft")

    # SET FLIGHT PATH ANGLE GUESSES
    p.set_val("traj.ascent_to_gear_retract.states:flight_path_angle", 0.2, units="rad")
    p.set_val("traj.ascent_to_flap_retract.states:flight_path_angle", 10, units="deg")
    p.set_val("traj.ascent_to_clean_aero.states:flight_path_angle", 10, units="deg")

    # SET TRUE AIRSPEED GUESSES
    p.set_val("traj.ascent_to_gear_retract.states:TAS",
              ascent_to_gear_retract.interp("TAS", [153.3196491, 170]), units="kn")
    p.set_val("traj.ascent_to_flap_retract.states:TAS",
              ascent_to_flap_retract.interp("TAS", [170, 190]), units="kn")
    p.set_val("traj.ascent_to_clean_aero.states:TAS",
              ascent_to_clean_aero.interp("TAS", [190, 195]), units="kn")

    # SET MASS GUESSES
    p.set_val("traj.ascent_to_gear_retract.states:mass", ascent_to_gear_retract.interp(
        "mass", [174963.74211336, 174000]), units="lbm")
    p.set_val("traj.ascent_to_flap_retract.states:mass",
              ascent_to_flap_retract.interp("mass", [174000., 173900]), units="lbm")
    p.set_val("traj.ascent_to_clean_aero.states:mass",
              ascent_to_clean_aero.interp("mass", [173900., 173800]), units="lbm")

    # SET RANGE GUESSES
    p.set_val("traj.ascent_to_gear_retract.states:distance", ascent_to_gear_retract.interp(
        Dynamic.Mission.DISTANCE, [4330.83393029, 5000]), units="ft")
    p.set_val("traj.ascent_to_flap_retract.states:distance",
              ascent_to_flap_retract.interp(Dynamic.Mission.DISTANCE, [5000., 6000]), units="ft")
    p.set_val("traj.ascent_to_clean_aero.states:distance",
              ascent_to_clean_aero.interp(Dynamic.Mission.DISTANCE, [6000., 7000]), units="ft")

    # SET TIME GUESSES
    p.set_val("traj.ascent_to_gear_retract.t_initial", 31.2)
    p.set_val("traj.ascent_to_gear_retract.t_duration", 3.0)

    p.set_val("traj.ascent_to_flap_retract.t_initial", 35.0)
    p.set_val("traj.ascent_to_flap_retract.t_duration", 5.0)

    p.set_val("traj.ascent_to_clean_aero.t_initial", 44.0)
    p.set_val("traj.ascent_to_clean_aero.t_duration", 10.0)

    # SET ALPHA GUESSES
    p.set_val("traj.ascent_to_gear_retract.polynomial_controls:alpha",
              ascent_to_gear_retract.interp('alpha', ys=[0, 0]), units="deg")
    p.set_val("traj.ascent_to_flap_retract.polynomial_controls:alpha",
              ascent_to_flap_retract.interp('alpha', ys=[0, 0]), units="deg")
    p.set_val("traj.ascent_to_clean_aero.polynomial_controls:alpha",
              ascent_to_clean_aero.interp('alpha', ys=[0, 0]), units="deg")

    p.set_solver_print(level=2, depth=1e99)

    return p


@use_tempdirs
class TestAscent(unittest.TestCase):

    def assert_result(self, p):
        t_init_gear = p.get_val(
            "traj.ascent_to_gear_retract.timeseries.time", units="s")[-1, ...]
        t_init_flaps = p.get_val(
            "traj.ascent_to_flap_retract.timeseries.time", units="s")[-1, ...]

        print("t_init_gear (s)", t_init_gear)
        print("t_init_flaps (s)", t_init_flaps)

        assert_near_equal(t_init_gear, 32.3, tolerance=0.01)
        assert_near_equal(t_init_flaps, 50.93320588, tolerance=0.01)

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_ascent_result_ipopt(self):
        p = make_ascent_problem(optimizer='IPOPT')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False,
                       solution_record_file='ascent_solution_IPOPT.db')
        self.assert_result(p)

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_ascent_result_snopt(self):
        p = make_ascent_problem(optimizer='SNOPT')
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False,
                       solution_record_file='ascent_solution_SNOPT.db')
        self.assert_result(p)


if __name__ == "__main__":
    unittest.main()
