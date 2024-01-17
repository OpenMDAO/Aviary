import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from packaging import version

from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Mission, Dynamic
from aviary.constants import RHO_SEA_LEVEL_ENGLISH

from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import UnsteadySolvedODE
from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import \
    UnsteadySolvedODE
from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.variable_info.options import get_option_defaults
from aviary.interface.default_phase_info.solved import default_mission_subsystems
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic


def run_mission(optimizer):
    mach = 0.78
    target_range = 2000.  # NM

    takeoff_mass = 175400  # lb
    climb_mach = 0.8
    cruise_alt = 37500  # ft
    cruise_mach = mach
    # target_range = 2000  # 3675  # NM
    solve_segments = False

    range_guesses = np.array([  # in meters
        0.,  # 'groundroll',
        1000.,  # 'rotation',
        1500.,  # 'ascent_to_gear_retract',
        2000.,  # 'ascent_to_flap_retract',
        2500.,  # 'ascent',
        3500.,  # Dynamic.Mission.VELOCITY_RATE,
        4500.,  # 'climb_at_constant_EAS',
        32000.,  # 'climb_at_constant_EAS_to_mach',
        300.e3,  # 'climb_at_constant_mach',
        600.e3,  # 'cruise',
        1700. * target_range,  # 'descent'
        1700. * target_range + 200000.,
    ])
    mass_guesses = np.array([
        1.0,
        0.999,
        0.999,
        0.999,
        0.998,
        0.998,
        0.998,
        0.990,
        0.969,
        0.951,
        0.873,
        0.866,
    ]) * takeoff_mass
    alt_guesses = [
        0.0,
        0.0,
        0.0,
        50.0,
        400.0,
        500.0,
        500.0,
        10000.,
        32857.,
        cruise_alt,
        cruise_alt,
        10000.,
    ]
    TAS_guesses = np.array([
        0.0,
        150.,
        200.,
        200.,
        200.,
        225.,
        251.,
        290.,
        465. * mach / 0.8,
        458. * mach / 0.8,
        483. * mach / 0.8,
        250.,
    ])

    mean_TAS = (TAS_guesses[1:] + TAS_guesses[:-1]) / 2. / 1.94
    range_durations = range_guesses[1:] - range_guesses[:-1]
    time_guesses = np.hstack((0., np.cumsum(range_durations / mean_TAS)))

    throttle_max = 1.0
    throttle_climb = 0.956
    throttle_cruise = 0.930
    throttle_idle = 0.0

    phase_info = {
        'groundroll': {
            'num_segments': 3,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': True,
            'clean': False,
            'initial_ref': 100.,
            'initial_bounds': (0., 500.),
            'duration_ref': 100.,
            'duration_bounds': (50., 1000.),
            'control_order': 1,
            'opt': True,
        },
        'rotation': {
            'num_segments': 2,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': True,
            'clean': False,
            'initial_ref': 1.e3,
            'initial_bounds': (50., 5000.),
            'duration_ref': 1.e3,
            'duration_bounds': (50., 2000.),
            'control_order': 1,
            'opt': True,
        },
        'ascent_to_gear_retract': {
            'num_segments': 2,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': False,
            'clean': False,
            'initial_ref': 1.e3,
            'initial_bounds': (10., 2.e3),
            'duration_ref': 1.e3,
            'duration_bounds': (500., 1.e4),
            'control_order': 1,
            'opt': False,
        },
        'ascent_to_flap_retract': {
            'num_segments': 2,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': False,
            'clean': False,
            'initial_ref': 1.e3,
            'initial_bounds': (10., 2.e5),
            'duration_ref': 1.e3,
            'duration_bounds': (500., 1.e4),
            'control_order': 1,
            'opt': False,
        },
        'ascent': {
            'num_segments': 2,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e3,
            'initial_bounds': (10., 2.e5),
            'duration_ref': 1.e3,
            'duration_bounds': (500., 1.e4),
            'control_order': 1,
            'opt': False,
        },
        Dynamic.Mission.VELOCITY_RATE: {
            'num_segments': 2,
            'throttle_setting': throttle_max,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e3,
            'initial_bounds': (10., 2.e5),
            'duration_ref': 1.e3,
            'duration_bounds': (500., 1.e4),
            'control_order': 1,
            'opt': False,
        },
        'climb_at_constant_EAS': {
            'num_segments': 2,
            'throttle_setting': throttle_climb,
            'input_speed_type': SpeedType.EAS,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e5,
            'initial_bounds': (100., 1.e7),
            'duration_ref': 1.e5,
            'duration_bounds': (1.e4, 5.e4),
            'control_order': 1,
            'opt': False,
            'fixed_EAS': 250.,
        },
        'climb_at_constant_EAS_to_mach': {
            'num_segments': 2,
            'throttle_setting': throttle_climb,
            'input_speed_type': SpeedType.EAS,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e5,
            'initial_bounds': (100., 1.e7),
            'duration_ref': 1.e5,
            'duration_bounds': (10.e3, 1.e6),
            'control_order': 1,
            'opt': True,
            'fixed_EAS': 270.,
        },
        'climb_at_constant_mach': {
            'num_segments': 2,
            'throttle_setting': throttle_climb,
            'input_speed_type': SpeedType.MACH,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e5,
            'initial_bounds': (100., 1.e7),
            'duration_ref': 1.e5,
            'duration_bounds': (10.e3, 1.e6),
            'control_order': 1,
            'opt': True,
        },
        'cruise': {
            'num_segments': 5,
            'throttle_setting': throttle_cruise,
            'input_speed_type': SpeedType.MACH,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e6,
            'initial_bounds': (1.e4, 2.e7),
            'duration_ref': 5.e6,
            'duration_bounds': (0.1e7, 3.e7),
            'control_order': 1,
            'opt': False,
        },
        'descent': {
            'num_segments': 3,
            'throttle_setting': throttle_idle,
            'input_speed_type': SpeedType.TAS,
            'ground_roll': False,
            'clean': True,
            'initial_ref': 1.e6,
            'initial_bounds': (100., 1.e7),
            'duration_ref': 1.e5,
            'duration_bounds': (1.5e5, 2.e5),
            'control_order': 1,
            'opt': True,
        },
    }
    phases = list(phase_info.keys())

    p = om.Problem()

    traj = p.model.add_subsystem("traj", dm.Trajectory())

    for idx, phase_name in enumerate(phases):
        throttle_setting = phase_info[phase_name]['throttle_setting']
        input_speed_type = phase_info[phase_name]['input_speed_type']
        num_segments = phase_info[phase_name]['num_segments']
        ground_roll = phase_info[phase_name]['ground_roll']
        clean = phase_info[phase_name]['clean']
        initial_bounds = phase_info[phase_name]['initial_bounds']
        duration_bounds = phase_info[phase_name]['duration_bounds']
        initial_ref = phase_info[phase_name]['initial_ref']
        duration_ref = phase_info[phase_name]['duration_ref']
        control_order = phase_info[phase_name]['control_order']
        opt = phase_info[phase_name]['opt']

        if phase_name == "groundroll":
            groundroll_trans = dm.Radau(
                num_segments=num_segments, order=3, compressed=True, solve_segments="forward",
            )

            ode_args = dict(
                aviary_options=get_option_defaults(),
                core_subsystems=default_mission_subsystems
            )

            phase = dm.Phase(
                ode_class=GroundrollODE,
                ode_init_kwargs=ode_args,
                transcription=groundroll_trans,
            )

            phase.set_time_options(fix_initial=True, fix_duration=False,
                                   units="kn", name="TAS",
                                   duration_bounds=duration_bounds, duration_ref=duration_ref, initial_ref=initial_ref)

            phase.set_state_options("time", rate_source="dt_dv", units="s",
                                    fix_initial=True, fix_final=False, ref=1., defect_ref=1.)

            phase.set_state_options("mass", rate_source="dmass_dv",
                                    fix_initial=True, fix_final=False, lower=1, upper=195_000, ref=takeoff_mass, defect_ref=takeoff_mass)

            phase.set_state_options(Dynamic.Mission.DISTANCE, rate_source="over_a",
                                    fix_initial=True, fix_final=False, lower=0, upper=2000., ref=1.e2, defect_ref=1.e2)

            phase.add_parameter("t_init_gear", units="s",
                                static_target=True, opt=False, val=32.3)
            phase.add_parameter("t_init_flaps", units="s",
                                static_target=True, opt=False, val=44.0)
            phase.add_parameter("wing_area", units="ft**2",
                                static_target=True, opt=False, val=1370)

        else:

            trans = dm.Radau(num_segments=num_segments, order=3,
                             compressed=True, solve_segments=solve_segments)

            ode_args = dict(
                input_speed_type=input_speed_type,
                clean=clean,
                ground_roll=ground_roll,
                aviary_options=get_option_defaults(),
                core_subsystems=default_mission_subsystems
            )

            phase = dm.Phase(
                ode_class=UnsteadySolvedODE,
                ode_init_kwargs=ode_args,
                transcription=trans,
            )

            phase.add_parameter(
                Dynamic.Mission.THROTTLE,
                opt=False,
                units="unitless",
                val=throttle_setting,
                static_target=False)

            phase.set_time_options(fix_initial=False, fix_duration=False,
                                   units="distance_units", name=Dynamic.Mission.DISTANCE,
                                   duration_bounds=duration_bounds, duration_ref=duration_ref,
                                   initial_bounds=initial_bounds, initial_ref=initial_ref)

            if phase_name == "cruise" or phase_name == "descent":
                time_ref = 1.e4
            else:
                time_ref = 100.

            phase.set_state_options("time", rate_source="dt_dr", targets=['t_curr'] if 'retract' in phase_name else [],
                                    fix_initial=False, fix_final=False, ref=time_ref, defect_ref=time_ref * 1.e2)

            phase.set_state_options("mass", rate_source="dmass_dr",
                                    fix_initial=False, fix_final=False, ref=170.e3, defect_ref=170.e5,
                                    val=170.e3, units='lbm', lower=10.e3)

            phase.add_parameter("wing_area", units="ft**2",
                                static_target=True, opt=False, val=1370)

            if Dynamic.Mission.VELOCITY_RATE in phase_name or 'ascent' in phase_name:
                phase.add_parameter(
                    "t_init_gear", units="s", static_target=True, opt=False, val=100)
                phase.add_parameter(
                    "t_init_flaps", units="s", static_target=True, opt=False, val=100)

            if 'rotation' in phase_name:
                phase.add_polynomial_control("TAS",
                                             order=control_order,
                                             units="kn", val=200.0,
                                             opt=opt, lower=1, upper=500, ref=250)

                phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                             order=control_order,
                                             fix_initial=False,
                                             rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                             opt=False, upper=40.e3, ref=30.e3, lower=-1.)

                phase.add_polynomial_control("alpha",
                                             order=control_order,
                                             lower=-4, upper=15,
                                             units='deg', ref=10.,
                                             val=[0., 5.],
                                             opt=opt)
            else:
                if 'constant_EAS' in phase_name:
                    fixed_EAS = phase_info[phase_name]['fixed_EAS']
                    phase.add_parameter("EAS", units="kn", val=fixed_EAS)
                elif 'constant_mach' in phase_name:
                    phase.add_parameter(
                        Dynamic.Mission.MACH,
                        units="unitless",
                        val=climb_mach)
                elif 'cruise' in phase_name:
                    phase.add_parameter(
                        Dynamic.Mission.MACH, units="unitless", val=cruise_mach)
                else:
                    phase.add_polynomial_control("TAS",
                                                 order=control_order,
                                                 fix_initial=False,
                                                 units="kn", val=200.0,
                                                 opt=True, lower=1, upper=500, ref=250)

                phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                             order=control_order,
                                             units="ft", val=0.,
                                             fix_initial=False,
                                             rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                             opt=opt, upper=40.e3, ref=30.e3, lower=-1.)

        if phase_name in phases[1:3]:
            phase.add_path_constraint(
                "fuselage_pitch", upper=15., units='deg', ref=15)
        if phase_name == "rotation":
            phase.add_boundary_constraint(
                "TAS", loc="final", upper=200., units="kn", ref=200.)
            phase.add_boundary_constraint(
                "normal_force", loc="final", equals=0., units="lbf", ref=10000.0)
        elif phase_name == "ascent_to_gear_retract":
            phase.add_path_constraint("load_factor", lower=0.0, upper=1.10)
        elif phase_name == "ascent_to_flap_retract":
            phase.add_path_constraint("load_factor", lower=0.0, upper=1.10)
        elif phase_name == "ascent":
            phase.add_path_constraint("EAS", upper=250., units="kn", ref=250.)
        elif phase_name == Dynamic.Mission.VELOCITY_RATE:
            phase.add_boundary_constraint(
                "EAS", loc="final", equals=250., units="kn", ref=250.)
        elif phase_name == "climb_at_constant_EAS":
            pass
        elif phase_name == "climb_at_constant_EAS_to_mach":
            phase.add_boundary_constraint(
                Dynamic.Mission.MACH, loc="final", equals=climb_mach, units="unitless")
        elif phase_name == "climb_at_constant_mach":
            pass
        elif phase_name == "descent":
            phase.add_boundary_constraint(
                Dynamic.Mission.DISTANCE,
                loc="final",
                equals=target_range,
                units="NM",
                ref=1.e3)
            phase.add_boundary_constraint(
                Dynamic.Mission.ALTITUDE,
                loc="final",
                equals=10.e3,
                units="ft",
                ref=10e3)
            phase.add_boundary_constraint(
                "TAS", loc="final", equals=250., units="kn", ref=250.)

        phase.add_timeseries_output(Dynamic.Mission.THRUST_TOTAL, units="lbf")
        phase.add_timeseries_output("thrust_req", units="lbf")
        phase.add_timeseries_output("normal_force")
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output("EAS", units="kn")
        phase.add_timeseries_output("TAS", units="kn")
        phase.add_timeseries_output(Dynamic.Mission.LIFT)
        phase.add_timeseries_output("CL")
        phase.add_timeseries_output("CD")
        phase.add_timeseries_output("time")
        phase.add_timeseries_output("mass")
        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)
        phase.add_timeseries_output("gear_factor")
        phase.add_timeseries_output("flap_factor")
        phase.add_timeseries_output("alpha")
        phase.add_timeseries_output(
            "fuselage_pitch", output_name="theta", units="deg")

        if 'rotation' in phase_name:
            phase.add_parameter("t_init_gear", units="s",
                                static_target=True, opt=False, val=100)
            phase.add_parameter("t_init_flaps", units="s",
                                static_target=True, opt=False, val=100)

        traj.add_phase(phase_name, phase)

        phase.timeseries_options['use_prefix'] = True

    traj.add_linkage_constraint(phase_a='ascent_to_gear_retract',
                                phase_b='ascent_to_flap_retract',
                                var_a='time',
                                var_b='t_init_gear',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='ascent_to_gear_retract',
                                phase_b='ascent',
                                var_a='time',
                                var_b='t_init_gear',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='ascent_to_gear_retract',
                                phase_b=Dynamic.Mission.VELOCITY_RATE,
                                var_a='time',
                                var_b='t_init_gear',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='ascent_to_flap_retract',
                                phase_b='ascent',
                                var_a='time',
                                var_b='t_init_flaps',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='ascent_to_flap_retract',
                                phase_b=Dynamic.Mission.VELOCITY_RATE,
                                var_a='time',
                                var_b='t_init_flaps',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.link_phases(phases[6:], vars=[Dynamic.Mission.ALTITUDE], ref=10.e3)
    traj.link_phases(phases, vars=['time'], ref=100.)
    traj.link_phases(phases, vars=['mass'], ref=10.e3)
    traj.link_phases(phases, vars=[Dynamic.Mission.DISTANCE], units='m', ref=10.e3)
    traj.link_phases(phases[:7], vars=['TAS'], units='kn', ref=200.)
    # traj.link_phases(phases[7:], vars=['TAS'], units='kn', ref=200.)

    p.model.add_subsystem("vrot_comp", VRotateComp())
    p.model.connect('traj.groundroll.states:mass',
                    'vrot_comp.mass', src_indices=om.slicer[0, ...])

    vrot_eq_comp = p.model.add_subsystem("vrot_eq_comp", om.EQConstraintComp())
    vrot_eq_comp.add_eq_output("v_rotate_error", eq_units="kn",
                               lhs_name="v_rot_computed", rhs_name="groundroll_v_final", add_constraint=True)

    p.model.connect('vrot_comp.Vrot', 'vrot_eq_comp.v_rot_computed')
    p.model.connect('traj.groundroll.timeseries.TAS',
                    'vrot_eq_comp.groundroll_v_final', src_indices=om.slicer[-1, ...])

    traj.add_parameter('wing_area', units='ft**2', static_target=True, opt=False)

    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring(show_sparsity=False, show_summary=False)
    p.driver.options["print_results"] = p.comm.rank == 0

    print_opt_iters = True

    p.driver.options["optimizer"] = optimizer

    if optimizer == "SNOPT":
        p.driver.opt_settings["Major optimality tolerance"] = 1e-6
        p.driver.opt_settings["Major feasibility tolerance"] = 1e-6
        # p.driver.opt_settings["Function precision"] = 1e-6
        # p.driver.opt_settings["Linesearch tolerance"] = 0.99
        # p.driver.opt_settings["Major step limit"] = 0.1
        if print_opt_iters:
            p.driver.opt_settings["iSumm"] = 6
        p.driver.opt_settings["Major iterations limit"] = 50
    elif optimizer == "IPOPT":
        p.driver.opt_settings["max_iter"] = 100
        p.driver.opt_settings["tol"] = 1e-6
        if print_opt_iters:
            p.driver.opt_settings["print_level"] = 5
        p.driver.opt_settings[
            "nlp_scaling_method"
        ] = "gradient-based"  # for faster convergence
        p.driver.opt_settings["alpha_for_y"] = "safer-min-dual-infeas"
        p.driver.opt_settings["mu_strategy"] = "monotone"
        p.driver.opt_settings["bound_mult_init_method"] = "mu-based"

    p.set_solver_print(level=-1)

    obj_comp = om.ExecComp(f"obj = -final_mass / {takeoff_mass} + final_time / 5.",
                           final_mass={"units": "lbm"},
                           final_time={"units": "h"})
    p.model.add_subsystem("obj_comp", obj_comp)

    final_phase_name = phases[-1]
    p.model.connect(f"traj.{final_phase_name}.timeseries.mass",
                    "obj_comp.final_mass", src_indices=[-1])
    p.model.connect(f"traj.{final_phase_name}.timeseries.time",
                    "obj_comp.final_time", src_indices=[-1])

    p.model.add_objective("obj_comp.obj", ref=1.0)

    p.setup()

    # SET INPUTS FOR POST TRAJECTORY ANALYSES
    # TODO: paramport
    params = ParamPort.param_data
    p.set_val('vrot_comp.' + Aircraft.Wing.AREA,
              params[Aircraft.Wing.AREA]["val"], units=params[Aircraft.Wing.AREA]["units"])
    p.set_val("vrot_comp.dV1", val=10, units="kn")
    p.set_val("vrot_comp.dVR", val=5, units="kn")
    p.set_val("vrot_comp.rho", val=RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3")
    p.set_val("vrot_comp.CL_max", val=2.1886, units="unitless")

    p.set_val("traj.parameters:wing_area", val=1370, units="ft**2")

    for idx, (phase_name, phase) in enumerate(traj._phases.items()):
        if phase_name != "groundroll":
            range_initial = range_guesses[idx]
            p.set_val(f"traj.{phase_name}.t_initial",
                      range_initial, units='distance_units')
            p.set_val(f"traj.{phase_name}.t_duration",
                      range_guesses[idx+1] - range_initial, units='distance_units')

            p.set_val(
                f"traj.{phase_name}.polynomial_controls:altitude",
                phase.interp(Dynamic.Mission.ALTITUDE, [
                             alt_guesses[idx], alt_guesses[idx + 1]]),
                units="ft",
            )

            if "constant_EAS" in phase_name:
                pass
            elif "constant_mach" in phase_name:
                pass
            elif "cruise" in phase_name:
                pass
            else:
                p.set_val(
                    f"traj.{phase_name}.polynomial_controls:TAS",
                    phase.interp("TAS", [TAS_guesses[idx], TAS_guesses[idx+1]]),
                    units="kn",
                )
        else:
            p.set_val(f"traj.{phase_name}.t_initial", 0., units='kn')
            p.set_val(f"traj.{phase_name}.t_duration", 100., units='kn')

        p.set_val(
            f"traj.{phase_name}.states:mass",
            phase.interp("mass", [mass_guesses[idx], mass_guesses[idx+1]]),
            units="lbm",
        )

        p.set_val(
            f"traj.{phase_name}.states:time",
            phase.interp("time", [time_guesses[idx], time_guesses[idx+1]]),
            units="s",
        )

    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False,
                   solution_record_file=f'solved_{optimizer}.sql',
                   )

    ranges = []
    masses = []
    alts = []
    TASs = []
    for idx, (phase_name, phase) in enumerate(traj._phases.items()):
        if phase_name == "groundroll":

            ranges.extend(
                p.get_val(f"traj.{phase_name}.timeseries.states:distance", units="m")[0])
            ranges.extend(
                p.get_val(f"traj.{phase_name}.timeseries.states:distance", units="m")[-1])

            masses.extend(
                p.get_val(f"traj.{phase_name}.timeseries.mass", units="lbm")[0])
            masses.extend(
                p.get_val(f"traj.{phase_name}.timeseries.mass", units="lbm")[-1])

            alts.extend([0., 0.])

            TASs.extend(
                p.get_val(f"traj.{phase_name}.timeseries.TAS", units="kn")[0])
            TASs.extend(
                p.get_val(f"traj.{phase_name}.timeseries.TAS", units="kn")[-1])
        else:
            range_initial = p.get_val(
                f"traj.{phase_name}.t_initial", units='distance_units')
            if idx > 1:
                ranges.extend(range_initial)
            if idx == (len(traj._phases) - 1):
                ranges.extend(
                    p.get_val(f"traj.{phase_name}.t_duration", units='distance_units') + range_initial)
            masses.extend(
                p.get_val(f"traj.{phase_name}.timeseries.mass", units="lbm")[-1])
            alts.extend(
                p.get_val(f"traj.{phase_name}.timeseries.altitude", units="ft")[-1])
            TASs.extend(
                p.get_val(f"traj.{phase_name}.timeseries.TAS", units="kn")[-1])

    return p


@use_tempdirs
class TestFullMission(unittest.TestCase):

    def assert_result(self, p):
        tf = p.get_val('traj.descent.timeseries.time', units='s')[-1, 0]
        vf = p.get_val('traj.descent.timeseries.TAS', units='kn')[-1, 0]
        wf = p.get_val('traj.descent.timeseries.mass', units='lbm')[-1, 0]

        print(f't_final: {tf:8.3f} s')
        print(f'v_final: {vf:8.3f} knots')
        print(f'w_final: {wf:8.3f} lbm')

        assert_near_equal(tf, 16234., tolerance=0.02)
        assert_near_equal(vf, 250., tolerance=0.02)
        assert_near_equal(wf, 150978., tolerance=0.02)

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_rotation_result_ipopt(self):
        p = run_mission(optimizer='IPOPT')
        self.assert_result(p)

    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_rotation_result_snopt(self):
        p = run_mission(optimizer='SNOPT')
        self.assert_result(p)


if __name__ == '__main__':
    use_SNOPT = True

    z = TestFullMission()

    if use_SNOPT:
        z.bench_test_rotation_result_snopt()

    else:
        z.bench_test_rotation_result_ipopt()
