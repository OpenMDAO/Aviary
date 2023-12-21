import os

import dymos as dm
import numpy as np
import openmdao.api as om
from packaging import version

if version.parse(dm.__version__) <= version.parse("1.6.1"):
    updated_dymos = False
else:
    updated_dymos = True

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.ode.breguet_cruise_ode import \
    BreguetCruiseODESolution
from aviary.mission.gasp_based.ode.flight_path_ode import FlightPathODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic

TARGET_RANGE = 3000
CRUISE_ALT = 37500.


def run_conventional_problem(make_plots=True, optimizer="IPOPT"):

    options = get_option_defaults()

    ode_args = dict(
        aviary_options=options
    )

    #
    # GROUNDROLL TO VROTATE
    #
    # Note:
    #   VRotate is computed in a post-trajectory analysis, and the difference between computed Vrot and
    #   the final speed in the groundroll phase and enforced downstream of the trajectory.
    #
    groundroll_trans = dm.Radau(
        num_segments=5, order=3, compressed=True, solve_segments=False
    )

    groundroll = dm.Phase(
        ode_class=FlightPathODE,
        ode_init_kwargs=dict(ode_args, ground_roll=True),
        transcription=groundroll_trans,
    )

    groundroll.set_time_options(
        fix_initial=True,
        fix_duration=False,
        units="s",
        targets="t_curr",
        duration_bounds=(20, 100),
        duration_ref=1)

    groundroll.set_state_options(
        "TAS",
        fix_initial=True,
        fix_final=False,
        lower=1.0E-6,
        upper=1000,
        ref=1,
        defect_ref=1)

    groundroll.set_state_options(
        "mass",
        fix_initial=True,
        fix_final=False,
        lower=1,
        upper=195_000,
        ref=1000,
        defect_ref=1000)

    groundroll.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=True,
        fix_final=False,
        lower=0,
        upper=100_000,
        ref=1,
        defect_ref=1)

    groundroll.add_parameter(Dynamic.Mission.ALTITUDE, opt=False,
                             static_target=False, val=0.0, units='ft')
    groundroll.add_parameter(
        Dynamic.Mission.FLIGHT_PATH_ANGLE, opt=False, static_target=False, val=0.0, units='rad')
    groundroll.add_parameter("t_init_gear", units="s",
                             static_target=True, opt=False, val=50)
    groundroll.add_parameter("t_init_flaps", units="s",
                             static_target=True, opt=False, val=50)

    #
    # ROTATION TO TAKEOFF
    #

    rotation_trans = dm.Radau(
        num_segments=5, order=3, compressed=True, solve_segments=False
    )

    rotation = dm.Phase(
        ode_class=FlightPathODE,
        # Use the standard ode_args and update it for ground_roll dynamics
        ode_init_kwargs=dict(ode_args, ground_roll=True),
        transcription=rotation_trans,
    )

    rotation.set_time_options(
        fix_initial=False,
        fix_duration=False,
        units="s",
        targets="t_curr",
        duration_bounds=(1, 100),
        duration_ref=1.0,
    )

    rotation.add_parameter(Dynamic.Mission.ALTITUDE, opt=False,
                           static_target=False, val=0.0, units='ft')
    rotation.add_parameter(Dynamic.Mission.FLIGHT_PATH_ANGLE,
                           opt=False, static_target=False, val=0.0, units='rad')
    rotation.add_parameter('alpha_rate', opt=False,
                           static_target=True, val=3.33, units='deg/s')
    rotation.add_parameter("t_init_gear", units="s",
                           static_target=True, opt=False, val=50)
    rotation.add_parameter("t_init_flaps", units="s",
                           static_target=True, opt=False, val=50)

    # State alpha is not defined in the ODE, taken from the parameter "alpha_rate"
    rotation.add_state(
        "alpha",
        units="rad",
        rate_source="alpha_rate",
        fix_initial=False,
        fix_final=False,
        lower=0.0,
        upper=np.radians(25),
        ref=1.0)

    rotation.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1.0,
        upper=1000.0,
        ref=100.0,
        defect_ref=100.0)

    rotation.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1.0,
        upper=190_000.0,
        ref=1000.0,
        defect_ref=1000.0,)

    rotation.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=10.e3,
        ref=100,
        defect_ref=100)

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

    #
    # ASCENT TO GEAR RETRACTION
    #
    ascent0_tx = dm.Radau(num_segments=3, order=3, compressed=True, solve_segments=False)
    ascent_to_gear_retract = dm.Phase(
        ode_class=FlightPathODE, ode_init_kwargs=ode_args, transcription=ascent0_tx)

    ascent_to_gear_retract.set_time_options(
        units="s",
        targets="t_curr",
        fix_initial=False,
        fix_duration=False,
        initial_bounds=(10, 50),
        duration_bounds=(1, 50))

    # Rate sources and units of states are set with tags in AscentEOM
    ascent_to_gear_retract.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=0.5,
        ref=1.0,
        defect_ref=100.0)

    ascent_to_gear_retract.set_state_options(
        Dynamic.Mission.ALTITUDE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=500,
        ref=100,
        defect_ref=10000)

    ascent_to_gear_retract.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        ref=100,
        defect_ref=1000)

    ascent_to_gear_retract.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=None,
        ref=175_000,
        defect_ref=175_000)

    ascent_to_gear_retract.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=15_000,
        ref=5280,
        defect_ref=5280)

    # Targets are not needed when there is a top-level ODE input with the same
    # name as the parameter, state, or control
    ascent_to_gear_retract.add_parameter(
        "t_init_gear", units="s", static_target=True, opt=False, val=50)
    ascent_to_gear_retract.add_parameter(
        "t_init_flaps", units="s", static_target=True, opt=False, val=50)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are
    # in degrees.
    if updated_dymos:
        ascent_to_gear_retract.add_control("alpha",
                                           continuity_scaler=1.0E-2,
                                           units="rad",
                                           val=0.0,  # continuity=True, rate_continuity=False,
                                           opt=True,
                                           lower=np.radians(-30),
                                           upper=np.radians(30),
                                           ref=0.01)
    else:
        ascent_to_gear_retract.add_control("alpha",
                                           units="rad",
                                           val=0.0,  # continuity=True, rate_continuity=False,
                                           opt=True,
                                           lower=np.radians(-30),
                                           upper=np.radians(30),
                                           ref=0.01)

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of
    # altitude is a design variable.
    ascent_to_gear_retract.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE,
        loc="final",
        equals=50,
        units="ft",
        ref=1.0,
        ref0=0.0,
        linear=False)

    # Load factor can be treated as a linear constraint as long as i_wing is
    # not a design variable.
    ascent_to_gear_retract.add_path_constraint(
        "load_factor",
        lower=0.0,
        upper=1.10,
        scaler=1.0E-6 / 0.001,
        linear=False)

    ascent_to_gear_retract.add_path_constraint(
        "fuselage_pitch",
        constraint_name="theta",
        lower=0,
        upper=15,
        units="deg",
        scaler=1.0E-6 / 0.001,
        linear=False)

    #
    # ASCENT TO FLAP RETRACTION
    #
    ascent1_tx = dm.Radau(num_segments=5, order=3,
                          compressed=True, solve_segments=False)
    ascent_to_flap_retract = dm.Phase(
        ode_class=FlightPathODE, ode_init_kwargs=ode_args, transcription=ascent1_tx)

    ascent_to_flap_retract.set_time_options(units="s", targets="t_curr",
                                            fix_initial=False, fix_duration=False,
                                            initial_bounds=(1, 500),
                                            duration_bounds=(0.5, 100))

    # Rate sources and units of states are set with tags in AscentEOM
    ascent_to_flap_retract.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        lower=-0.2618,
        upper=0.43633,
        ref=0.43633,
        defect_ref=0.43633)

    ascent_to_flap_retract.set_state_options(
        Dynamic.Mission.ALTITUDE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=500,
        ref=100,
        defect_ref=100)

    ascent_to_flap_retract.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        ref=100,
        defect_ref=100)

    ascent_to_flap_retract.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=None,
        ref=175_000,
        defect_ref=175_000)

    ascent_to_flap_retract.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=50_000,
        ref=5280,
        defect_ref=5280)

    # Targets are not needed when there is a top-level ODE input with the same
    # name as the parameter, state, or control
    ascent_to_flap_retract.add_parameter(
        "t_init_gear", units="s", static_target=True, opt=False, val=100)
    ascent_to_flap_retract.add_parameter(
        "t_init_flaps", units="s", static_target=True, opt=False, val=100)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are
    # in degrees.
    if updated_dymos:
        ascent_to_flap_retract.add_control("alpha",
                                           continuity_scaler=1.0E-2,
                                           units="rad",
                                           val=0.0,
                                           continuity=True,
                                           rate_continuity=False,
                                           opt=True,
                                           lower=np.radians(-14),
                                           upper=np.radians(14),
                                           ref=0.01)
    else:
        ascent_to_flap_retract.add_control("alpha",
                                           units="rad",
                                           val=0.0,
                                           continuity=True,
                                           rate_continuity=False,
                                           opt=True,
                                           lower=np.radians(-14),
                                           upper=np.radians(14),
                                           ref=0.01)

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of
    # altitude is a design variable.
    ascent_to_flap_retract.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE,
        loc="final",
        equals=400,
        units="ft",
        ref=1.0,
        ref0=0.0,
        linear=False,
    )

    # Load factor can be treated as a linear constraint as long as i_wing is
    # not a design variable.
    ascent_to_flap_retract.add_path_constraint(
        "load_factor",
        lower=0.0,
        upper=1.10,
        scaler=1.0E-6 / 0.001,
        linear=False)

    ascent_to_flap_retract.add_path_constraint(
        "fuselage_pitch",
        constraint_name="theta",
        lower=0,
        upper=15,
        units="deg",
        scaler=1.0E-6 / 0.001,
        linear=False)

    #
    # ASCENT TO CLEAN AERO CONFIG
    #
    ascent2_tx = dm.Radau(num_segments=3, order=3, compressed=True, solve_segments=False)
    ascent_to_clean_aero = dm.Phase(
        ode_class=FlightPathODE, ode_init_kwargs=ode_args, transcription=ascent2_tx)

    ascent_to_clean_aero.set_time_options(units="s", targets="t_curr",
                                          fix_initial=False, fix_duration=False,
                                          initial_bounds=(1, 500),
                                          duration_bounds=(0.5, 100))

    # Rate sources and units of states are set with tags in AscentEOM
    ascent_to_clean_aero.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        lower=-0.2618,
        upper=0.43633,
        ref=0.43633,
        defect_ref=0.43633)

    ascent_to_clean_aero.set_state_options(
        Dynamic.Mission.ALTITUDE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=500,
        ref=100,
        defect_ref=100)

    ascent_to_clean_aero.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=1000,
        ref=100,
        defect_ref=100)

    ascent_to_clean_aero.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=None,
        ref=175_000,
        defect_ref=175_000)

    ascent_to_clean_aero.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=30.e3,
        ref=5_280,
        defect_ref=5_280)

    # Targets are not needed when there is a top-level ODE input with the same
    # name as the parameter, state, or control
    ascent_to_clean_aero.add_parameter(
        "t_init_gear", units="s", static_target=True, opt=False)
    ascent_to_clean_aero.add_parameter(
        "t_init_flaps", units="s", static_target=True, opt=False)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are
    # in degrees.
    ascent_to_clean_aero.add_polynomial_control(
        "alpha", order=1,
        units="rad",
        val=0.0,  # continuity=True, rate_continuity=True,
        opt=True,
        lower=np.radians(-14),
        upper=np.radians(14),
        ref=0.01)

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of
    # altitude is a design variable.
    ascent_to_clean_aero.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE,
        loc="final",
        equals=500,
        units="ft",
        ref=1.0,
        ref0=0.0,
        linear=False)

    # Ensure flaps are fully retracted
    # Note setting equals=0.0 here will result in a failure of the optimization, since flap_factor approaches
    # zero asymptotically.
    ascent_to_clean_aero.add_boundary_constraint(
        "flap_factor",
        loc="final",
        upper=1.0E-3,
        ref=1.0,
        ref0=0.0,
        linear=False)

    # Load factor can be treated as a linear constraint as long as i_wing is
    # not a design variable.
    ascent_to_clean_aero.add_path_constraint(
        "load_factor",
        lower=0.0,
        upper=1.10,
        scaler=1.0E-6 / 0.001,
        linear=False)

    ascent_to_clean_aero.add_path_constraint(
        "fuselage_pitch",
        constraint_name="theta",
        lower=0,
        upper=15,
        units="deg",
        scaler=1.0E-6 / 0.001,
        linear=False)

    #
    # ACCEL CONFIG
    #
    accel2_tx = dm.Radau(num_segments=5, order=3, compressed=True, solve_segments=False)
    accel_ode_args = ode_args.deepcopy()
    accel = dm.Phase(ode_class=FlightPathODE,
                     ode_init_kwargs=accel_ode_args, transcription=accel2_tx)

    accel.set_time_options(units="s", targets="t_curr",
                           fix_initial=False, fix_duration=False,
                           initial_bounds=(1, 500),
                           duration_bounds=(0.5, 10))

    # Rate sources and units of states are set with tags in AscentEOM
    accel.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        lower=0.0,
        ref=1.0e1,
        defect_ref=1.0e1)

    accel.set_state_options(
        Dynamic.Mission.ALTITUDE,
        fix_initial=False,
        fix_final=False,
        units="ft",
        ref0=495,
        ref=520e1,
        defect_ref=500e1,
        lower=495,
        upper=520)

    accel.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        units="kn",
        ref=250e1,
        defect_ref=250e1)

    accel.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=500_000,
        ref=175_000e1,
        defect_ref=175_000e1)

    accel.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        lower=0,
        upper=None,
        units="ft",
        ref=1000e1,
        defect_ref=1000e1)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are
    # in degrees.
    accel.add_polynomial_control("alpha",
                                 order=2,
                                 opt=True,
                                 units="rad",
                                 val=0.0,
                                 lower=np.radians(-8),
                                 upper=np.radians(8))

    # boundary/path constraints + controls
    # Final altitude can be a linear constraint, since the final value of
    # altitude is a design variable.
    accel.add_boundary_constraint("EAS", loc="final", equals=250., ref=250, units="kn")

    accel.add_parameter("t_init_gear", units="s",
                        static_target=True, opt=False, val=32.3)
    accel.add_parameter("t_init_flaps", units="s",
                        static_target=True, opt=False, val=44.0)

    climb_ode_args = ode_args.deepcopy()
    climb_ode_args['clean'] = True
    transcription = dm.Radau(num_segments=11, order=3, compressed=True)
    constant_quantity = 'EAS'
    constant_EAS_climb = dm.Phase(
        ode_class=FlightPathODE,
        ode_init_kwargs=climb_ode_args,
        transcription=transcription)

    constant_EAS_climb.set_time_options(
        fix_initial=False, initial_bounds=(
            0, 1000), duration_bounds=(
            20, 1000), units="s", duration_ref=100)

    constant_EAS_climb.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        units="kn",
        ref=250,
        defect_ref=250)

    constant_EAS_climb.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=None,
        units="lbm",
        ref=200000,
        defect_ref=200000)

    constant_EAS_climb.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=None,
        units="ft",
        ref=10000,
        defect_ref=10000)

    if constant_quantity == 'EAS':
        constant_EAS_climb.set_state_options(
            Dynamic.Mission.ALTITUDE,
            fix_initial=False,
            fix_final=False,
            lower=400,
            upper=20000,
            units="ft",
            ref=20.e3,
            defect_ref=20.e3)
    else:
        constant_EAS_climb.set_state_options(
            Dynamic.Mission.ALTITUDE,
            fix_initial=False,
            fix_final=False,
            lower=9000,
            upper=50000,
            units="ft",
            ref=40000,
            defect_ref=40000)

    constant_EAS_climb.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        units="rad",
        lower=0.0,
        ref=1.0,
        defect_ref=1.0)

    constant_EAS_climb.add_control("alpha", opt=True, units="rad", val=0.0,
                                   lower=np.radians(-14), upper=np.radians(14), ref=0.01,
                                   rate_continuity=True, rate2_continuity=False)

    if updated_dymos:
        constant_EAS_climb.add_control("alpha",
                                       continuity_scaler=1.0E-2,
                                       rate_continuity=False,
                                       rate2_continuity=False,
                                       opt=True,
                                       units="rad",
                                       val=0.0,
                                       lower=np.radians(-14),
                                       upper=np.radians(14),
                                       ref=0.01)
    else:
        constant_EAS_climb.add_control("alpha",
                                       rate_continuity=False,
                                       rate2_continuity=False,
                                       opt=True,
                                       units="rad",
                                       val=0.0,
                                       lower=np.radians(-14),
                                       upper=np.radians(14),
                                       ref=0.01)

    # constant_EAS_climb.add_polynomial_control("alpha", order=1, opt=True, units="rad", lower=-np.radians(14), upper=np.radians(14), ref=0.01)

    if constant_quantity == 'EAS':
        constant_EAS_climb.add_path_constraint(
            "EAS", lower=249.9, upper=250.1, ref=350., units="kn")
        # constant_EAS_climb.add_path_constraint("EAS", equals=350., ref=350., units="kn")
        constant_EAS_climb.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE, loc="final", equals=10.e3, ref=10.e3, units="ft")
    else:
        constant_EAS_climb.add_path_constraint(
            Dynamic.Mission.MACH, lower=0.799, upper=0.801, ref=1.e3)
        constant_EAS_climb.add_boundary_constraint(
            Dynamic.Mission.ALTITUDE, loc="final", equals=37.5e3, ref=10.e3, units="ft")

    constant_EAS_climb.add_timeseries_output("EAS", output_name="EAS", units="kn")
    constant_EAS_climb.add_timeseries_output(
        Dynamic.Mission.MACH, output_name=Dynamic.Mission.MACH, units="unitless")
    constant_EAS_climb.add_timeseries_output("alpha", output_name="alpha", units="deg")
    constant_EAS_climb.add_timeseries_output("CL", output_name="CL", units="unitless")
    constant_EAS_climb.add_timeseries_output(
        Dynamic.Mission.THRUST_TOTAL,
        output_name=Dynamic.Mission.THRUST_TOTAL,
        units="lbf")
    constant_EAS_climb.add_timeseries_output("CD", output_name="CD", units="unitless")

    #
    # CLIMB 2 (constrained EAS)
    #
    climb_ode_args = ode_args.deepcopy()
    climb_ode_args['clean'] = True
    transcription = dm.Radau(num_segments=5, order=3, compressed=True)
    climb_to_mach = dm.Phase(ode_class=FlightPathODE,
                             ode_init_kwargs=climb_ode_args, transcription=transcription)

    climb_to_mach.set_time_options(
        units="s", fix_initial=False,
        initial_bounds=(1, 1.e3),
        duration_bounds=(10., 1.e3))

    # Rate sources and units of states are set with tags in AscentEOM
    climb_to_mach.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        units="kn",
        ref=250,
        defect_ref=250)

    climb_to_mach.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=None,
        units="lbm",
        ref=200000,
        defect_ref=200000)

    climb_to_mach.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=None,
        units="ft",
        ref=10000,
        defect_ref=10000)

    climb_to_mach.set_state_options(
        Dynamic.Mission.ALTITUDE,
        fix_initial=False,
        fix_final=False,
        lower=9.e3,
        upper=40.e3,
        units="ft",
        ref=20.e3,
        defect_ref=40000)

    climb_to_mach.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        units="rad",
        ref=1.0,
        defect_ref=1.0)

    climb_to_mach.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        lower=0,
        upper=None,
        units="ft",
        ref=1000,
        defect_ref=1000)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are in degrees.
    # climb_to_mach.add_polynomial_control("alpha", order=5, opt=True, units="rad", val=0.0, lower=np.radians(-14), upper=np.radians(14))
    if updated_dymos:
        climb_to_mach.add_control("alpha",
                                  continuity_scaler=1.0E-2,
                                  rate_continuity=False,
                                  rate2_continuity=False,
                                  opt=True,
                                  units="rad",
                                  val=0.0,
                                  lower=np.radians(-14),
                                  upper=np.radians(14),
                                  ref=0.01)
    else:
        climb_to_mach.add_control("alpha",
                                  rate_continuity=False,
                                  rate2_continuity=False,
                                  opt=True,
                                  units="rad",
                                  val=0.0,
                                  lower=np.radians(-14),
                                  upper=np.radians(14),
                                  ref=0.01)

    # boundary/path constraints + controls
    climb_to_mach.add_path_constraint(
        "EAS", lower=270., upper=270., ref=270., units='kn')
    climb_to_mach.add_boundary_constraint(Dynamic.Mission.MACH, equals=0.8, loc="final")

    #
    # CLIMB 3 (constant Mach)
    #
    climb_ode_args = ode_args.deepcopy()
    climb_ode_args['clean'] = True
    transcription = dm.Radau(num_segments=5, order=3, compressed=True)
    climb_to_cruise = dm.Phase(
        ode_class=FlightPathODE,
        ode_init_kwargs=climb_ode_args,
        transcription=transcription)

    climb_to_cruise.set_time_options(
        units="s", fix_initial=False,
        initial_bounds=(400, 1.5e3),
        duration_bounds=(10., 1.e3))

    # Rate sources and units of states are set with tags in AscentEOM
    climb_to_cruise.set_state_options(
        "TAS",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=1000,
        units="kn",
        ref=250,
        defect_ref=250)

    climb_to_cruise.set_state_options(
        "mass",
        fix_initial=False,
        fix_final=False,
        lower=1,
        upper=None,
        units="lbm",
        ref=200000,
        defect_ref=200000)

    climb_to_cruise.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        fix_final=False,
        lower=0,
        upper=None,
        units="ft",
        ref=10000,
        defect_ref=10000)

    climb_to_cruise.set_state_options(
        Dynamic.Mission.ALTITUDE,
        fix_initial=False,
        fix_final=False,
        lower=15.e3,
        upper=40.e3,
        units="ft",
        ref=20.e3,
        defect_ref=40000)

    climb_to_cruise.set_state_options(
        Dynamic.Mission.FLIGHT_PATH_ANGLE,
        fix_initial=False,
        fix_final=False,
        units="rad",
        lower=0,
        ref=1.0,
        defect_ref=1.0)

    climb_to_cruise.set_state_options(
        Dynamic.Mission.DISTANCE,
        fix_initial=False,
        lower=0,
        upper=None,
        units="ft",
        ref=1000,
        defect_ref=1000)

    # Using Radau, we need to apply rate continuity if desired for controls even when the transcription is compressed.
    # The final value of a control in a segment is interpolated, not a design variable.
    # Note we specify units as radians here, since alpha inputs in the ODE are in degrees.
    # climb_to_cruise.add_polynomial_control("alpha", order=5, opt=True, units="rad", val=0.0, lower=np.radians(-14), upper=np.radians(14))
    if updated_dymos:
        climb_to_cruise.add_control("alpha",
                                    continuity_scaler=1.0E-2,
                                    rate_continuity=False,
                                    rate2_continuity=False,
                                    opt=True,
                                    units="rad",
                                    val=0.0,
                                    lower=np.radians(-14),
                                    upper=np.radians(14),
                                    ref=0.01)
    else:
        climb_to_cruise.add_control("alpha",
                                    rate_continuity=False,
                                    rate2_continuity=False,
                                    opt=True,
                                    units="rad",
                                    val=0.0,
                                    lower=np.radians(-14),
                                    upper=np.radians(14),
                                    ref=0.01)

    # boundary/path constraints + controls
    climb_to_cruise.add_path_constraint(Dynamic.Mission.MACH, equals=0.8, ref=1.)
    climb_to_cruise.add_boundary_constraint(
        Dynamic.Mission.ALTITUDE, loc="final", equals=37.5e3, ref=30.e3, units="ft")

    #
    # CRUISE: Analytic Breguet Range Phase
    #
    cruise_ode_args = ode_args.deepcopy()
    cruise = dm.AnalyticPhase(
        ode_class=BreguetCruiseODESolution,
        ode_init_kwargs=cruise_ode_args,
        num_nodes=20
    )

    # Time here is really the independent variable through which we are integrating.
    # In the case of the Breguet Range ODE, it's mass.
    # We rely on mass being monotonically non-increasing across the phase.
    cruise.set_time_options(
        name='mass',
        fix_initial=False,
        fix_duration=False,
        units="lbm",
        targets="mass",
        initial_bounds=(10.e3, 500_000),
        initial_ref=100_000,
        duration_bounds=(-50000, -10),
        duration_ref=10000,
    )

    cruise.add_parameter(Dynamic.Mission.ALTITUDE, opt=False, val=CRUISE_ALT, units='ft')
    cruise.add_parameter(Dynamic.Mission.MACH, opt=False, val=0.8)
    cruise.add_parameter("wing_area", opt=False, val=1370, units="ft**2")
    cruise.add_parameter("initial_distance", opt=False, val=0.0,
                         units="NM", static_target=True)
    cruise.add_parameter("initial_time", opt=False, val=0.0,
                         units="s", static_target=True)

    cruise.add_timeseries_output("time", units="s")

    cruise.add_boundary_constraint(
        Dynamic.Mission.DISTANCE, loc="final", equals=TARGET_RANGE, ref=TARGET_RANGE, units="NM")

    #
    # PROBLEM DEFINITION
    #
    p = om.Problem()
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    traj.add_phase("groundroll", groundroll)
    traj.add_phase("rotation", rotation)
    traj.add_phase("ascent_to_gear_retract", ascent_to_gear_retract)
    traj.add_phase("ascent_to_flap_retract", ascent_to_flap_retract)
    traj.add_phase("ascent_to_clean_aero", ascent_to_clean_aero)
    traj.add_phase(Dynamic.Mission.VELOCITY_RATE, accel)
    traj.add_phase("constant_EAS_climb", constant_EAS_climb)
    traj.add_phase("climb_to_mach", climb_to_mach)
    traj.add_phase("climb_to_cruise", climb_to_cruise)
    traj.add_phase("cruise", cruise)

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

    traj.add_linkage_constraint(phase_a='ascent_to_gear_retract',
                                phase_b=Dynamic.Mission.VELOCITY_RATE,
                                var_a='time',
                                var_b='t_init_gear',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='ascent_to_flap_retract',
                                phase_b='ascent_to_clean_aero',
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

    traj.add_linkage_constraint(phase_a='climb_to_cruise',
                                phase_b='cruise',
                                var_a='time',
                                var_b='initial_time',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='climb_to_cruise',
                                phase_b='cruise',
                                var_a=Dynamic.Mission.DISTANCE,
                                var_b='initial_distance',
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='climb_to_cruise',
                                phase_b='cruise',
                                var_a=Dynamic.Mission.MACH,
                                var_b=Dynamic.Mission.MACH,
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='climb_to_cruise',
                                phase_b='cruise',
                                var_a=Dynamic.Mission.ALTITUDE,
                                var_b=Dynamic.Mission.ALTITUDE,
                                loc_a='final',
                                loc_b='initial',
                                connected=True)

    traj.add_linkage_constraint(phase_a='climb_to_cruise',
                                phase_b='cruise',
                                var_a='mass',
                                var_b='mass',
                                loc_a='final',
                                loc_b='initial',
                                connected=False)

    # 2. Continuity for rotation to first ascent phase. Altitude and flight
    # path angle are not states on the ground.
    traj.link_phases(['groundroll', 'rotation'],
                     vars=['time', 'TAS', 'mass', Dynamic.Mission.DISTANCE])

    # Continuity for rotation to first ascent phase. Altitude and flight path
    # angle are not states on the ground.
    traj.link_phases(['rotation', 'ascent_to_gear_retract'], vars=[
                     'time', 'TAS', 'mass', Dynamic.Mission.DISTANCE, 'alpha', Dynamic.Mission.FLIGHT_PATH_ANGLE, Dynamic.Mission.ALTITUDE])

    # 3. Enforce value continuity between all phases in ascent for time,
    # states, and alpha control
    traj.link_phases(['ascent_to_gear_retract',
                      'ascent_to_flap_retract',
                      'ascent_to_clean_aero',
                      Dynamic.Mission.VELOCITY_RATE],
                     vars=['time',
                           Dynamic.Mission.FLIGHT_PATH_ANGLE,
                           Dynamic.Mission.ALTITUDE,
                           'TAS',
                           Dynamic.Mission.MASS,
                           Dynamic.Mission.DISTANCE,
                           'alpha'])

    traj.link_phases([Dynamic.Mission.VELOCITY_RATE,
                      'constant_EAS_climb',
                      'climb_to_mach',
                      'climb_to_cruise'],
                     vars=['time',
                           Dynamic.Mission.ALTITUDE,
                           Dynamic.Mission.MASS,
                           Dynamic.Mission.DISTANCE])

    # Post trajectory analyses
    p.model.add_subsystem("vrot_comp", VRotateComp())
    p.model.connect('traj.groundroll.states:mass',
                    'vrot_comp.mass', src_indices=om.slicer[0, ...])

    vrot_eq_comp = p.model.add_subsystem("vrot_eq_comp", om.EQConstraintComp())
    vrot_eq_comp.add_eq_output(
        "v_rotate_error",
        eq_units="kn",
        lhs_name="v_rot_computed",
        rhs_name="groundroll_v_final",
        add_constraint=True)

    p.model.connect('vrot_comp.Vrot', 'vrot_eq_comp.v_rot_computed')
    p.model.connect('traj.groundroll.states:TAS',
                    'vrot_eq_comp.groundroll_v_final', src_indices=om.slicer[-1, ...])

    for phase_name, phase in traj._phases.items():
        if "clean" in phase.options["ode_init_kwargs"] and not phase.options["ode_init_kwargs"]["clean"]:
            phase.add_timeseries_output('gear_factor')
            phase.add_timeseries_output('flap_factor')
        phase.add_timeseries_output('normal_force')
        phase.add_timeseries_output('fuselage_pitch')
        # Add alpha to the timeseries as 'alpha' regardless of whether it is a
        # control or polynomial control.
        phase.add_timeseries_output('alpha', units='deg')
        phase.add_timeseries_output(Dynamic.Mission.ALTITUDE, units='ft')
        phase.add_timeseries_output(Dynamic.Mission.MACH)
        phase.add_timeseries_output('EAS', units="kn")
        phase.add_timeseries_output('TAS', units="kn")
        phase.add_timeseries_output('mass', units="lbm")
        phase.add_timeseries_output(Dynamic.Mission.DISTANCE, units="NM")

    # Setup the solvers
    # We're roughtly twice as fast per iteration with a direct solver at the phase level.
    for phase_name, phase in traj._phases.items():
        phase.linear_solver = om.DirectSolver(iprint=0, assemble_jac=True)
    p.driver = om.pyOptSparseDriver()

    if optimizer == "SNOPT":
        p.driver.options["optimizer"] = "SNOPT"
        p.driver.opt_settings["iSumm"] = 6
        p.driver.opt_settings['Major step limit'] = 0.5
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-5
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
        p.driver.opt_settings['Hessian updates'] = 5
    elif optimizer == "IPOPT":
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['tol'] = 1.0E-4
        # p.driver.opt_settings['dual_inf_tol'] = 1.0E-4
        p.driver.opt_settings['acceptable_tol'] = 5.0E-4
        p.driver.opt_settings['mu_init'] = 1e-5
        p.driver.opt_settings['max_iter'] = 100
        p.driver.opt_settings['print_level'] = 5
        # for faster convergence
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['nlp_scaling_max_gradient'] = 1000.0

        # p.driver.opt_settings['nlp_scaling_method'] = 'none'
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
        p.driver.opt_settings['limited_memory_update_type'] = 'bfgs'
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        # p.driver.opt_settings['mu_strategy'] = 'adaptive'
        # p.driver.opt_settings['grad_f_constant'] = "yes"

    # Uncomment declare_coloring when the coloring needs to be reset (number
    # of nodes have changed, constraints changed, or underlying models have
    # changed)
    p.driver.declare_coloring(tol=1.0E-12, num_full_jacs=1)

    obj_comp = om.ExecComp("obj = -final_mass / 175191 + final_time / 10",
                           final_mass={"units": "lbm"},
                           final_time={"units": "h"})
    p.model.add_subsystem("obj_comp", obj_comp)

    p.model.connect("traj.cruise.timeseries.mass",
                    "obj_comp.final_mass", src_indices=[-1])
    p.model.connect("traj.cruise.timeseries.time",
                    "obj_comp.final_time", src_indices=[-1])

    p.model.add_objective("obj_comp.obj", ref=1.0)

    p.setup()

    # SET INPUTS FOR POST TRAJECTORY ANALYSES
    # TODO: paramport
    params = ParamPort.param_data
    p.set_val('vrot_comp.' + Aircraft.Wing.AREA,
              params[Aircraft.Wing.AREA]["val"],
              units=params[Aircraft.Wing.AREA]["units"])
    p.set_val("vrot_comp.dV1", val=10, units="kn")
    p.set_val("vrot_comp.dVR", val=5, units="kn")
    p.set_val("vrot_comp.rho", val=RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3")
    p.set_val("vrot_comp.CL_max", val=2.1886, units="unitless")

    # SET TIME INITIAL GUESS
    p.set_val("traj.cruise.t_initial", 171481, units="lbm")  # Initial mass in cruise
    p.set_val("traj.cruise.t_duration", -10000, units="lbm")  # Mass of fuel consumed

    p.set_val("traj.cruise.parameters:altitude", val=37500.0, units="ft")
    p.set_val("traj.cruise.parameters:mach", val=0.8, units="unitless")
    p.set_val("traj.cruise.parameters:wing_area", val=1370, units="ft**2")
    p.set_val("traj.cruise.parameters:initial_distance", val=175.0, units="NM")
    p.set_val("traj.cruise.parameters:initial_time", val=1600.0, units="s")

    # RUN THE DRIVER AND EXPLICITLY SIMULATE THE CONTROL SOLUTION
    this_dir = os.path.dirname(os.path.realpath(__file__))
    restart_file = this_dir + os.sep + "dymos_solution_IPOPT.db"
    dm.run_problem(p, run_driver=True, simulate=False, make_plots=make_plots,
                   restart=restart_file,
                   simulation_record_file=f'dymos_simulation_{optimizer}.db',
                   solution_record_file=f'dymos_solution_{optimizer}.db')

    print("t_init_gear (s)", p.get_val(
        "traj.ascent_to_gear_retract.timeseries.time", units="s")[-1, ...])
    print("t_init_flaps (s)", p.get_val(
        "traj.ascent_to_flap_retract.timeseries.time", units="s")[-1, ...])
    print("initial cruise time (s)", p.get_val(
        "traj.cruise.parameter_vals:initial_time", units="s"))
    print("cruise time (s)", p.get_val(
        "traj.cruise.timeseries.time", units="s")[-1, ...])
    print("cruise range (NM)", p.get_val(
        "traj.cruise.timeseries.distance", units="NM")[-1, ...])
    print("cruise equivalent airspeed", p.get_val(
        "traj.cruise.timeseries.EAS", units="kn")[0, ...])

    return p
