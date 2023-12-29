'''
NOTES:
Includes:
Takeoff, Climb, Cruise, Descent, Landing
Computed Aero
Large Single Aisle 1 data
'''
import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
import scipy.constants as _units
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from packaging import version

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.mission.flops_based.phases.climb_phase import Climb
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.mission.flops_based.phases.descent_phase import Descent
from aviary.subsystems.premission import CorePreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import get_flops_inputs, get_flops_outputs
from aviary.variable_info.functions import setup_trajectory_params
from aviary.utils.preprocessors import preprocess_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.variables_in import VariablesIn
from aviary.interface.default_phase_info.height_energy import default_premission_subsystems, default_mission_subsystems

try:
    import pyoptsparse
except ImportError:
    pyoptsparse = None

# benchmark based on large single aisle 1 (fixed cruise alt) FLOPS model


def run_trajectory(sim=True):
    prob = om.Problem()
    if pyoptsparse:
        driver = prob.driver = om.pyOptSparseDriver()
        driver.options["optimizer"] = "SNOPT"
        # driver.declare_coloring()  # currently disabled pending resolve of issue 2507
        if driver.options["optimizer"] == "SNOPT":
            driver.opt_settings["Major iterations limit"] = 45
            driver.opt_settings["Major optimality tolerance"] = 1e-4
            driver.opt_settings["Major feasibility tolerance"] = 1e-6
            driver.opt_settings["iSumm"] = 6
        elif driver.options["optimizer"] == "IPOPT":
            driver.opt_settings["max_iter"] = 100
            driver.opt_settings["tol"] = 1e-3
            driver.opt_settings['print_level'] = 4

    else:
        driver = prob.driver = om.ScipyOptimizeDriver()
        opt_settings = prob.driver.opt_settings

        driver.options['optimizer'] = 'SLSQP'
        opt_settings['maxiter'] = 100
        opt_settings['ftol'] = 5.0e-3
        opt_settings['eps'] = 1e-2

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')
    aviary_outputs = get_flops_outputs('LargeSingleAisle1FLOPS')

    preprocess_options(aviary_inputs)

    alt_airport = 0  # ft

    alt_i_climb = 0*_units.foot  # m
    alt_f_climb = 35000.0*_units.foot  # m
    mass_i_climb = 180623*_units.lb  # kg
    mass_f_climb = 176765*_units.lb  # kg
    v_i_climb = 198.44*_units.knot  # m/s
    v_f_climb = 455.49*_units.knot  # m/s
    # initial mach set to lower value so it can intersect with takeoff end mach
    # mach_i_climb = 0.3
    mach_i_climb = 0.2
    mach_f_climb = 0.79
    range_i_climb = 0*_units.nautical_mile  # m
    range_f_climb = 160.3*_units.nautical_mile  # m
    t_i_climb = 2 * _units.minute  # sec
    t_f_climb = 26.20*_units.minute  # sec
    t_duration_climb = t_f_climb - t_i_climb

    alt_i_cruise = 35000*_units.foot  # m
    alt_f_cruise = 35000*_units.foot  # m
    alt_min_cruise = 35000*_units.foot  # m
    alt_max_cruise = 35000*_units.foot  # m
    mass_i_cruise = 176765*_units.lb  # kg
    mass_f_cruise = 143521*_units.lb  # kg
    v_i_cruise = 455.49*_units.knot  # m/s
    v_f_cruise = 455.49*_units.knot  # m/s
    # mach_i_cruise = 0.79
    # mach_f_cruise = 0.79
    mach_min_cruise = 0.79
    mach_max_cruise = 0.79
    range_i_cruise = 160.3*_units.nautical_mile  # m
    range_f_cruise = 3243.9*_units.nautical_mile  # m
    t_i_cruise = 26.20*_units.minute  # sec
    t_f_cruise = 432.38*_units.minute  # sec
    t_duration_cruise = t_f_cruise - t_i_cruise

    alt_i_descent = 35000*_units.foot
    # final altitude set to 35 to ensure landing is feasible point
    # alt_f_descent = 0*_units.foot
    alt_f_descent = 35*_units.foot
    v_i_descent = 455.49*_units.knot
    v_f_descent = 198.44*_units.knot
    mach_i_descent = 0.79
    mach_f_descent = 0.3
    mass_i_descent = 143521*_units.pound
    mass_f_descent = 143035*_units.pound
    range_i_descent = 3243.9*_units.nautical_mile
    range_f_descent = 3378.7*_units.nautical_mile
    t_i_descent = 432.38*_units.minute
    t_f_descent = 461.62*_units.minute
    t_duration_descent = t_f_descent - t_i_descent

    ##################
    # Define Phases  #
    ##################

    num_segments_climb = 6
    num_segments_cruise = 1
    num_segments_descent = 5

    climb_seg_ends, _ = dm.utils.lgl.lgl(num_segments_climb + 1)
    descent_seg_ends, _ = dm.utils.lgl.lgl(num_segments_descent + 1)

    transcription_climb = dm.Radau(
        num_segments=num_segments_climb, order=3, compressed=True,
        segment_ends=climb_seg_ends)
    transcription_cruise = dm.Radau(
        num_segments=num_segments_cruise, order=3, compressed=True)
    transcription_descent = dm.Radau(
        num_segments=num_segments_descent, order=3, compressed=True,
        segment_ends=descent_seg_ends)

    takeoff_options = Takeoff(
        airport_altitude=alt_airport,  # ft
        num_engines=aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)
    )

    climb_options = Climb(
        'test_climb',
        user_options=AviaryValues({
            'initial_altitude': (alt_i_climb, 'm'),
            'final_altitude': (alt_f_climb, 'm'),
            'initial_mach': (mach_i_climb, 'unitless'),
            'final_mach': (mach_f_climb, 'unitless'),
            'fix_initial': (False, 'unitless'),
            'fix_range': (False, 'unitless'),
            'input_initial': (True, 'unitless'),
        }),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_climb,
    )

    cruise_options = Cruise(
        'test_cruise',
        user_options=AviaryValues({
            'min_altitude': (alt_min_cruise, 'm'),
            'max_altitude': (alt_max_cruise, 'm'),
            'min_mach': (mach_min_cruise, 'unitless'),
            'max_mach': (mach_max_cruise, 'unitless'),
            'required_available_climb_rate': (300, 'ft/min'),
            'fix_initial': (False, 'unitless'),
            'fix_final': (False, 'unitless'),
        }),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_cruise,
    )

    descent_options = Descent(
        'test_descent',
        user_options=AviaryValues({
            'final_altitude': (alt_f_descent, 'm'),
            'initial_altitude': (alt_i_descent, 'm'),
            'initial_mach': (mach_i_descent, 'unitless'),
            'final_mach': (mach_f_descent, 'unitless'),
            'fix_initial': (False, 'unitless'),
            'fix_range': (True, 'unitless'),
        }),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_descent,
    )

    landing_options = Landing(
        ref_wing_area=aviary_inputs.get_val(Aircraft.Wing.AREA, 'ft**2'),
        Cl_max_ldg=aviary_inputs.get_val(Mission.Landing.LIFT_COEFFICIENT_MAX)
    )

    # Upstream static analysis for aero
    prob.model.add_subsystem(
        'pre_mission',
        CorePreMission(aviary_options=aviary_inputs,
                       subsystems=default_premission_subsystems),
        promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['aircraft:*', 'mission:*'])

    # directly connect phases (strong_couple = True), or use linkage constraints (weak
    # coupling / strong_couple=False)
    strong_couple = False

    takeoff = takeoff_options.build_phase(False)

    climb = climb_options.build_phase(aviary_options=aviary_inputs)

    cruise = cruise_options.build_phase(aviary_options=aviary_inputs)

    descent = descent_options.build_phase(aviary_options=aviary_inputs)

    landing = landing_options.build_phase(False)

    prob.model.add_subsystem(
        'takeoff', takeoff, promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['mission:*'])

    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    # if fix_initial is false, can we always set input_initial to be true for
    # necessary states, and then ignore if we use a linkage?
    climb.set_time_options(
        fix_initial=True, fix_duration=False, units='s',
        duration_bounds=(1, t_duration_climb*2), duration_ref=t_duration_climb)
    cruise.set_time_options(
        fix_initial=False, fix_duration=False, units='s',
        duration_bounds=(1, t_duration_cruise*2), duration_ref=t_duration_cruise)
    descent.set_time_options(
        fix_initial=False, fix_duration=False, units='s',
        duration_bounds=(1, t_duration_descent*2), duration_ref=t_duration_descent)

    traj.add_phase('climb', climb)

    traj.add_phase('cruise', cruise)

    traj.add_phase('descent', descent)

    climb.timeseries_options['use_prefix'] = True
    cruise.timeseries_options['use_prefix'] = True
    descent.timeseries_options['use_prefix'] = True

    prob.model.add_subsystem(
        'landing', landing, promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['mission:*'])

    traj.link_phases(["climb", "cruise"], ["time", Dynamic.Mission.ALTITUDE,
                     Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS, Dynamic.Mission.RANGE], connected=strong_couple)
    traj.link_phases(["cruise", "descent"], ["time", Dynamic.Mission.ALTITUDE,
                     Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS, Dynamic.Mission.RANGE], connected=strong_couple)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs)

    ##################################
    # Connect in Takeoff and Landing #
    ##################################

    prob.model.add_subsystem(
        "takeoff_constraints",
        om.ExecComp(
            [
                "takeoff_mass_con=takeoff_mass-climb_start_mass",
                "takeoff_range_con=takeoff_range-climb_start_range",
                "takeoff_vel_con=takeoff_vel-climb_start_vel",
                "takeoff_alt_con=takeoff_alt-climb_start_alt"
            ],
            takeoff_mass_con={'units': 'lbm'}, takeoff_mass={'units': 'lbm'},
            climb_start_mass={'units': 'lbm'},
            takeoff_range_con={'units': 'ft'}, takeoff_range={'units': 'ft'},
            climb_start_range={'units': 'ft'},
            takeoff_vel_con={'units': 'm/s'}, takeoff_vel={'units': 'm/s'},
            climb_start_vel={'units': 'm/s'},
            takeoff_alt_con={'units': 'ft'}, takeoff_alt={'units': 'ft'},
            climb_start_alt={'units': 'ft'}
        ),
        promotes_inputs=[
            ("takeoff_mass", Mission.Takeoff.FINAL_MASS),
            ("takeoff_range", Mission.Takeoff.GROUND_DISTANCE),
            ("takeoff_vel", Mission.Takeoff.FINAL_VELOCITY),
            ("takeoff_alt", Mission.Takeoff.FINAL_ALTITUDE),
        ],
        promotes_outputs=["takeoff_mass_con", "takeoff_range_con",
                          "takeoff_vel_con", "takeoff_alt_con"],
    )

    prob.model.connect('traj.climb.states:mass',
                       'takeoff_constraints.climb_start_mass', src_indices=[0])
    prob.model.connect('traj.climb.states:range',
                       'takeoff_constraints.climb_start_range', src_indices=[0])
    prob.model.connect('traj.climb.states:velocity',
                       'takeoff_constraints.climb_start_vel', src_indices=[0])
    prob.model.connect('traj.climb.states:altitude',
                       'takeoff_constraints.climb_start_alt', src_indices=[0])

    prob.model.connect(Mission.Takeoff.FINAL_MASS,
                       'traj.climb.initial_states:mass')
    prob.model.connect(Mission.Takeoff.GROUND_DISTANCE,
                       'traj.climb.initial_states:range')
    prob.model.connect(Mission.Takeoff.FINAL_VELOCITY,
                       'traj.climb.initial_states:velocity')
    prob.model.connect(Mission.Takeoff.FINAL_ALTITUDE,
                       'traj.climb.initial_states:altitude')

    prob.model.connect('traj.descent.states:mass',
                       Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])
    prob.model.connect('traj.descent.states:altitude', Mission.Landing.INITIAL_ALTITUDE,
                       src_indices=[-1])

    ##########################
    # Add Objective Function #
    ##########################

    # This is an example of a overall mission objective
    # create a compound objective that minimizes climb time and maximizes final mass
    # we are maxing final mass b/c we don't have an independent value for fuel_mass yet
    # we are going to normalize these (makign each of the sub-objectives approx = 1 )
    prob.model.add_subsystem(
        "regularization",
        om.ExecComp(
            # TODO: change the scaling on climb_duration
            "reg_objective = - descent_mass_final/60000",
            reg_objective=0.0,
            descent_mass_final={"units": "kg", "shape": 1},
        ),
        promotes_outputs=['reg_objective']
    )
    # connect the final mass from cruise into the objective
    prob.model.connect("traj.descent.states:mass",
                       "regularization.descent_mass_final", src_indices=[-1])

    prob.model.add_objective('reg_objective', ref=1)

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)

    # TODO: Why is this in outputs and not inputs?
    key = Aircraft.Engine.THRUST_REVERSERS_MASS
    val, units = aviary_outputs.get_item(key)
    prob.model.set_input_defaults(key, val, units)

    prob.model.add_subsystem(
        'input_sink',
        VariablesIn(aviary_options=aviary_inputs),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )

    prob.setup()

    ###########################################
    # Intial Settings for States and Controls #
    ###########################################

    prob.set_val('traj.climb.t_initial', t_i_climb, units='s')
    prob.set_val('traj.climb.t_duration', t_duration_climb, units='s')

    prob.set_val('traj.climb.states:altitude', climb.interp(
        Dynamic.Mission.ALTITUDE, ys=[alt_i_climb, alt_f_climb]), units='m')
    # prob.set_val(
    #     'traj.climb.states:velocity', climb.interp(Dynamic.Mission.VELOCITY, ys=[170, v_f_climb]),
    #     units='m/s')
    prob.set_val('traj.climb.states:velocity', climb.interp(
        Dynamic.Mission.VELOCITY, ys=[v_i_climb, v_f_climb]), units='m/s')
    prob.set_val('traj.climb.states:mass', climb.interp(
        Dynamic.Mission.MASS, ys=[mass_i_climb, mass_f_climb]), units='kg')
    prob.set_val('traj.climb.states:range', climb.interp(
        Dynamic.Mission.RANGE, ys=[range_i_climb, range_f_climb]), units='m')  # nmi

    prob.set_val('traj.climb.controls:velocity_rate',
                 climb.interp(Dynamic.Mission.VELOCITY_RATE, ys=[0.25, 0.05]),
                 units='m/s**2')
    prob.set_val('traj.climb.controls:throttle',
                 climb.interp(Dynamic.Mission.THROTTLE, ys=[0.5, 0.5]),
                 units='unitless')

    prob.set_val('traj.cruise.t_initial', t_i_cruise, units='s')
    prob.set_val('traj.cruise.t_duration', t_duration_cruise, units='s')

    prob.set_val('traj.cruise.states:altitude', cruise.interp(
        Dynamic.Mission.ALTITUDE, ys=[alt_i_cruise, alt_f_cruise]), units='m')
    prob.set_val('traj.cruise.states:velocity', cruise.interp(
        Dynamic.Mission.VELOCITY, ys=[v_i_cruise, v_f_cruise]), units='m/s')
    prob.set_val('traj.cruise.states:mass', cruise.interp(
        Dynamic.Mission.MASS, ys=[mass_i_cruise, mass_f_cruise]), units='kg')
    prob.set_val('traj.cruise.states:range', cruise.interp(
        Dynamic.Mission.RANGE, ys=[range_i_cruise, range_f_cruise]), units='m')  # nmi

    prob.set_val('traj.cruise.controls:velocity_rate',
                 cruise.interp(Dynamic.Mission.VELOCITY_RATE, ys=[0.0, 0.0]),
                 units='m/s**2')
    prob.set_val('traj.cruise.controls:throttle',
                 cruise.interp(Dynamic.Mission.THROTTLE, ys=[0.5, 0.5]),
                 units='unitless')

    prob.set_val('traj.descent.t_initial', t_i_descent, units='s')
    prob.set_val('traj.descent.t_duration', t_duration_descent, units='s')

    prob.set_val('traj.descent.states:altitude', descent.interp(
        Dynamic.Mission.ALTITUDE, ys=[alt_i_descent, alt_f_descent]), units='m')
    prob.set_val('traj.descent.states:velocity', descent.interp(
        Dynamic.Mission.VELOCITY, ys=[v_i_descent, v_f_descent]), units='m/s')
    prob.set_val('traj.descent.states:mass', descent.interp(
        Dynamic.Mission.MASS, ys=[mass_i_descent, mass_f_descent]), units='kg')
    prob.set_val('traj.descent.states:range', descent.interp(
        Dynamic.Mission.RANGE, ys=[range_i_descent, range_f_descent]), units='m')

    prob.set_val('traj.descent.controls:velocity_rate',
                 descent.interp(Dynamic.Mission.VELOCITY_RATE, ys=[-0.25, 0.05]),
                 units='m/s**2')
    prob.set_val('traj.descent.controls:throttle',
                 descent.interp(Dynamic.Mission.THROTTLE, ys=[0.0, 0.0]),
                 units='unitless')

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=sim, make_plots=False, simulate_kwargs={
                   'times_per_seg': 100, 'atol': 1e-9, 'rtol': 1e-9},
                   solution_record_file='large_single_aisle_1_solution.db')
    prob.record("final")
    prob.cleanup()

    return prob


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    def bench_test_full_mission_large_single_aisle_1(self):

        prob = run_trajectory(sim=False)

        times_climb = prob.get_val('traj.climb.timeseries.time', units='s')
        altitudes_climb = prob.get_val(
            'traj.climb.timeseries.states:altitude', units='m')
        masses_climb = prob.get_val('traj.climb.timeseries.states:mass', units='kg')
        ranges_climb = prob.get_val('traj.climb.timeseries.states:range', units='m')
        velocities_climb = prob.get_val(
            'traj.climb.timeseries.states:velocity', units='m/s')
        thrusts_climb = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')
        times_cruise = prob.get_val('traj.cruise.timeseries.time', units='s')
        altitudes_cruise = prob.get_val(
            'traj.cruise.timeseries.states:altitude', units='m')
        masses_cruise = prob.get_val('traj.cruise.timeseries.states:mass', units='kg')
        ranges_cruise = prob.get_val('traj.cruise.timeseries.states:range', units='m')
        velocities_cruise = prob.get_val(
            'traj.cruise.timeseries.states:velocity', units='m/s')
        thrusts_cruise = prob.get_val(
            'traj.cruise.timeseries.thrust_net_total', units='N')
        times_descent = prob.get_val('traj.descent.timeseries.time', units='s')
        altitudes_descent = prob.get_val(
            'traj.descent.timeseries.states:altitude', units='m')
        masses_descent = prob.get_val('traj.descent.timeseries.states:mass', units='kg')
        ranges_descent = prob.get_val('traj.descent.timeseries.states:range', units='m')
        velocities_descent = prob.get_val(
            'traj.descent.timeseries.states:velocity', units='m/s')
        thrusts_descent = prob.get_val(
            'traj.descent.timeseries.thrust_net_total', units='N')

        expected_times_s_climb = [
            [120.], [163.76325456], [224.14761365], [243.25905685],
            [243.25905685], [336.41086249], [464.94134172], [505.62079895],
            [505.62079895], [626.47614107], [793.23184628], [846.00945683],
            [846.00945683], [966.86479896], [1133.62050417], [1186.39811471],
            [1186.39811471], [1279.54992036], [1408.08039959], [1448.75985682],
            [1448.75985682], [1492.52311138], [1552.90747047], [1572.01891366]]
        expected_altitudes_m_climb = [
            [10.668], [0.], [1014.96376993], [1474.88490893],
            [1474.88490893], [3450.8992731], [5549.58991137], [6073.60289567],
            [6073.60289567], [7247.65102441], [8313.09660771], [8596.21261844],
            [8596.21261844], [9175.1805832], [9807.17741861], [9956.37909202],
            [9956.37909202], [10156.95278577], [10387.66964669], [10469.50089961],
            [10469.50089961], [10565.21436037], [10659.11220277], [10668.]]
        expected_masses_kg_climb = [
            [81929.21464651], [81845.89383716], [81698.07677997], [81651.72896811],
            [81651.72896811], [81451.5201184], [81238.04517346], [81180.70309604],
            [81180.70309604], [81030.90916993], [80852.63402635], [80799.37995098],
            [80799.37995098], [80685.66441919], [80541.42480306], [80497.91293126],
            [80497.91293126], [80422.58418106], [80320.79494098], [80289.0126768],
            [80289.0126768], [80255.43650168], [80209.81935885], [80195.47487578]]
        expected_ranges_m_climb = [
            [2023.13728418], [7309.29529844], [17625.33986498], [21042.26607802],
            [21042.26607802], [38595.24976097], [63501.20931809], [71333.49759313],
            [71333.49759313], [95066.70557732], [129238.42790298], [140520.59490531],
            [140520.59490531], [166712.77513886], [203467.60377384], [215173.95317494],
            [215173.95317494], [236133.6275649], [265602.08094631], [275049.96491681],
            [275049.96491681], [285205.23036747], [299246.15306878], [303707.66300732]]
        expected_velocities_ms_climb = [
            [86.27497122], [150.21937278], [181.89204952], [185.19401741],
            [185.19401741], [193.05567218], [193.94250629], [194.2274308],
            [194.2274308], [199.49589035], [211.4364855], [214.57351446],
            [214.57351446], [218.7009295], [221.65884364], [223.09827293],
            [223.09827293], [226.88054212], [231.57590395], [232.11980164],
            [232.11980164], [232.10881873], [233.15980248], [234.25795132]]
        expected_thrusts_N_climb = [
            [132715.85627327], [165245.22863773], [167448.73291722], [160900.8406642],
            [160900.8406642], [127136.48651829], [98045.08452667], [90929.93404254],
            [90929.93404254], [77481.47159379], [66945.17202854], [64341.32162839],
            [64341.32162839], [59055.89956031], [54151.63303545], [53020.95829213],
            [53020.95829213], [51622.0422889], [50045.94083626], [49417.41803104],
            [49417.41803104], [48644.32827828], [47944.25167088], [47933.1465328]]

        expected_times_s_cruise = [[1572.01891366], [
            10224.87387308], [22164.04764439], [25942.75532212]]
        expected_altitudes_m_cruise = [[10668.], [10668.], [10668.], [10668.]]
        expected_masses_kg_cruise = [[80195.47487578], [
            74629.35731156], [67375.28255861], [65172.93890147]]
        expected_ranges_m_cruise = [[303707.66300732], [
            2330707.73887436], [5127554.12700399], [6012746.44622707]]
        expected_velocities_ms_cruise = [[234.25795132], [
            234.25795132], [234.25795132], [234.25795132]]
        expected_thrusts_N_cruise = [[41824.96226832], [
            39641.43620269], [36936.34901257], [36154.52769977]]

        expected_times_s_descent = [
            [25942.75532212], [26006.65447531], [26094.82226471], [26122.72706861],
            [26122.72706861], [26253.22730164], [26433.29098775], [26490.28052876],
            [26490.28052876], [26645.43239405], [26859.51030121], [26927.26522688],
            [26927.26522688], [27057.76545992], [27237.82914603], [27294.81868703],
            [27294.81868703], [27358.71784022], [27446.88562963], [27474.79043352]]
        expected_altitudes_m_descent = [
            [1.06680000e+04], [1.06680000e+04], [1.01870034e+04], [9.98295734e+03],
            [9.98295734e+03], [8.97976640e+03], [7.54858109e+03], [7.09637574e+03],
            [7.09637574e+03], [5.88591590e+03], [4.22625069e+03], [3.69222380e+03],
            [3.69222380e+03], [2.61250549e+03], [1.13888643e+03], [7.11625616e+02],
            [7.11625616e+02], [3.06900655e+02], [1.06680000e+01], [1.06680000e+01]]
        expected_masses_kg_descent = [
            [65172.93890147], [65166.80185527], [65159.70921571], [65157.42471584],
            [65157.42471584], [65146.76382877], [65130.82732956], [65125.34524899],
            [65125.34524899], [65109.28561058], [65083.64486622], [65074.44308601],
            [65074.44308601], [65054.73517647], [65022.79856574], [65011.43379965],
            [65011.43379965], [64998.13023434], [64979.19121811], [64973.1898506]]
        expected_ranges_m_descent = [
            [6012746.44622707], [6026563.35875724], [6043156.61719684],
            [6048186.41354773], [6048186.41354773], [6070643.2394269],
            [6100144.8206876], [6109350.83226072], [6109350.83226072],
            [6133985.78349596], [6167266.64464002], [6177671.71555852],
            [6177671.71555852], [6197775.49290063], [6225243.66498946],
            [6233748.78818454], [6233748.78818454], [6242933.61721041],
            [6254285.51733338], [6257352.4]]
        expected_velocities_ms_descent = [
            [234.25795132], [200.84994317], [180.71529791], [177.55751262],
            [177.55751262], [167.72481926], [161.81636181], [160.6651682],
            [160.6651682], [157.4025997], [154.14780084], [153.88934075],
            [153.88934075], [154.23063023], [150.48828108], [146.76095672],
            [146.76095672], [139.54419751], [115.31288595], [102.07377559]]
        expected_thrusts_N_descent = [
            [0.00000000e+00], [8.47474799e-13], [1.05910816e-13], [-6.67104835e-13],
            [0.00000000e+00], [2.74056170e-13], [6.93382542e-13], [8.33653923e-13],
            [0.00000000e+00], [4.66122489e-14], [-1.43628962e-13], [-2.96621357e-13],
            [0.00000000e+00], [0.00000000e+00], [-1.18597257e-14], [-1.95075855e-14],
            [-9.53501847e-15], [0.00000000e+00], [0.00000000e+00], [-3.72692384e-15]]

        expected_times_s_climb = np.array(expected_times_s_climb)
        expected_altitudes_m_climb = np.array(expected_altitudes_m_climb)
        expected_masses_kg_climb = np.array(expected_masses_kg_climb)
        expected_ranges_m_climb = np.array(expected_ranges_m_climb)
        expected_velocities_ms_climb = np.array(expected_velocities_ms_climb)
        expected_thrusts_N_climb = np.array(expected_thrusts_N_climb)

        expected_times_s_cruise = np.array(expected_times_s_cruise)
        expected_altitudes_m_cruise = np.array(expected_altitudes_m_cruise)
        expected_masses_kg_cruise = np.array(expected_masses_kg_cruise)
        expected_ranges_m_cruise = np.array(expected_ranges_m_cruise)
        expected_velocities_ms_cruise = np.array(expected_velocities_ms_cruise)
        expected_thrusts_N_cruise = np.array(expected_thrusts_N_cruise)

        expected_times_s_descent = np.array(expected_times_s_descent)
        expected_altitudes_m_descent = np.array(expected_altitudes_m_descent)
        expected_masses_kg_descent = np.array(expected_masses_kg_descent)
        expected_ranges_m_descent = np.array(expected_ranges_m_descent)
        expected_velocities_ms_descent = np.array(expected_velocities_ms_descent)
        expected_thrusts_N_descent = np.array(expected_thrusts_N_descent)

        # Check Objective and other key variables to a reasonably tight tolerance.

        rtol = 1e-2

        # Mass at the end of Descent
        assert_near_equal(masses_descent[-1],
                          expected_masses_kg_descent[-1], tolerance=rtol)

        # Range at the end of Descent
        assert_near_equal(ranges_descent[-1],
                          expected_ranges_m_descent[-1], tolerance=rtol)

        # Flight time
        assert_near_equal(times_descent[-1],
                          expected_times_s_descent[-1], tolerance=rtol)

        # Check mission values.

        # NOTE rtol = 0.05 = 5% different  from truth (first timeseries)
        #      atol = 2 = no more than +/-2 meter/second/kg difference between values
        #      atol_altitude - 30 ft. There is occasional time-shifting with the N3CC
        #                      model during climb and descent so we need a looser
        #                      absolute tolerance for the points near the ground.
        rtol = .05
        atol = 2.0
        atol_altitude = 30.0

        # FLIGHT PATH
        # CLIMB
        warn_timeseries_near_equal(
            times_climb, altitudes_climb, expected_times_s_climb,
            expected_altitudes_m_climb, abs_tolerance=atol_altitude, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_climb, masses_climb, expected_times_s_climb,
            expected_masses_kg_climb, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_climb, ranges_climb, expected_times_s_climb,
            expected_ranges_m_climb, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_climb, velocities_climb, expected_times_s_climb,
            expected_velocities_ms_climb, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_climb, thrusts_climb, expected_times_s_climb,
            expected_thrusts_N_climb, abs_tolerance=atol, rel_tolerance=rtol)

        # CRUISE
        warn_timeseries_near_equal(
            times_cruise, altitudes_cruise, expected_times_s_cruise,
            expected_altitudes_m_cruise, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_cruise, masses_cruise, expected_times_s_cruise,
            expected_masses_kg_cruise, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_cruise, ranges_cruise, expected_times_s_cruise,
            expected_ranges_m_cruise, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_cruise, velocities_cruise, expected_times_s_cruise,
            expected_velocities_ms_cruise, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_cruise, thrusts_cruise, expected_times_s_cruise,
            expected_thrusts_N_cruise, abs_tolerance=atol, rel_tolerance=rtol)

        # DESCENT
        warn_timeseries_near_equal(
            times_descent, altitudes_descent, expected_times_s_descent,
            expected_altitudes_m_descent, abs_tolerance=atol_altitude,
            rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_descent, masses_descent, expected_times_s_descent,
            expected_masses_kg_descent, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_descent, ranges_descent, expected_times_s_descent,
            expected_ranges_m_descent, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_descent, velocities_descent, expected_times_s_descent,
            expected_velocities_ms_descent, abs_tolerance=atol, rel_tolerance=rtol)
        warn_timeseries_near_equal(
            times_descent, thrusts_descent, expected_times_s_descent,
            expected_thrusts_N_descent, abs_tolerance=atol, rel_tolerance=rtol)


if __name__ == '__main__':
    unittest.main()
