'''
NOTES:
Includes:
Takeoff, Climb, Cruise, Descent, Landing
Computed Aero
Large Single Aisle 2 data
'''
import unittest

import dymos as dm
import numpy as np
import openmdao.api as om
import scipy.constants as _units
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from packaging import version

from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.mission.flops_based.phases.climb_phase import Climb
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.mission.flops_based.phases.descent_phase import Descent
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.variables_in import VariablesIn
from aviary.subsystems.premission import CorePreMission
from aviary.interface.default_phase_info.height_energy import default_premission_subsystems, default_mission_subsystems
from aviary.utils.preprocessors import preprocess_options

try:
    import pyoptsparse
except ImportError:
    pyoptsparse = None

# benchmark based on Large Single Aisle 2 (fixed cruise alt) FLOPS model


def run_trajectory(sim=True):
    prob = om.Problem(model=om.Group())
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

    aviary_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')

    preprocess_options(aviary_inputs)

    alt_airport = 0  # ft

    ref_wing_area = 1370  # ft^2 TODO: where should this get connected from?

    alt_i_climb = 35.0*_units.foot  # m  (comes from takeoff)
    alt_f_climb = 35000.0*_units.foot  # m
    mass_i_climb = 178172*_units.lb  # kg  (comes from takeoff)
    mass_f_climb = 174160*_units.lb  # kg
    v_i_climb = 198.44*_units.knot  # m/s  (comes from takeoff)
    v_f_climb = 452.61*_units.knot  # m/s
    mach_i_climb = 0.2    # TODO: (need to compute this in takeoff)
    mach_f_climb = 0.785
    range_i_climb = 0*_units.nautical_mile  # m  (comes from takeoff)
    range_f_climb = 124*_units.nautical_mile  # m
    t_i_climb = 0
    t_duration_climb = 20.14*_units.minute  # sec

    alt_i_cruise = 35000*_units.foot  # m
    alt_f_cruise = 35000*_units.foot  # m
    alt_min_cruise = 35000*_units.foot  # m
    alt_max_cruise = 35000*_units.foot  # m
    mass_i_cruise = 174160*_units.lb  # kg
    mass_f_cruise = 140515*_units.lb  # kg
    v_i_cruise = 452.61*_units.knot  # m/s
    v_f_cruise = 452.61*_units.knot  # m/s
    mach_f_cruise = 0.785
    mach_min_cruise = 0.785
    mach_max_cruise = 0.785
    range_i_cruise = 124*_units.nautical_mile  # m
    range_f_cruise = 2830.8*_units.nautical_mile  # m
    t_i_cruise = t_i_climb + t_duration_climb  # sec
    t_duration_cruise = (378.95)*_units.minute - t_duration_climb  # sec

    alt_i_descent = 35000*_units.foot  # m
    alt_f_descent = 35*_units.foot  # m
    v_i_descent = 452.61*_units.knot  # m/s
    v_f_descent = 198.44*_units.knot  # m/s
    mass_i_descent = 140515*_units.pound  # kg
    mass_f_descent = 140002*_units.pound  # kg
    mach_i_descent = mach_f_cruise
    mach_f_descent = 0.3
    range_i_descent = 2830.8*_units.nautical_mile  # m
    range_f_descent = 2960.0*_units.nautical_mile  # m
    t_i_descent = t_duration_cruise+t_duration_climb  # sec
    t_duration_descent = 2000  # sec

    Cl_max_ldg = 3  # TODO: should this come from aero?

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
        ref_wing_area=ref_wing_area,  # ft**2
        Cl_max_ldg=Cl_max_ldg  # no units
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
                 descent.interp(Dynamic.Mission.VELOCITY_RATE, ys=[-0.25, 0.0]),
                 units='m/s**2')
    prob.set_val('traj.descent.controls:throttle',
                 descent.interp(Dynamic.Mission.THROTTLE, ys=[0.0, 0.0]),
                 units='unitless')

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=sim, make_plots=False, simulate_kwargs={
                   'times_per_seg': 100, 'atol': 1e-9, 'rtol': 1e-9},
                   solution_record_file='large_single_aisle_2_solution.db')
    prob.record("final")
    prob.cleanup()

    return prob


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    def bench_test_full_mission_large_single_aisle_2(self):

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

        expected_times_s_climb = [[0.], [36.42164779], [86.67608797], [102.58144646],
                                  [102.58144646], [180.10635382], [287.07490293],
                                  [320.93008299], [320.93008299], [421.51105549],
                                  [560.29226869], [604.21604816], [604.21604816],
                                  [704.79702067], [843.57823386], [887.50201334],
                                  [887.50201334], [965.0269207], [1071.9954698],
                                  [1105.85064986], [1105.85064986], [1142.27229765],
                                  [1192.52673783], [1208.43209632]]
        expected_altitudes_m_climb = [[10.668], [10.668], [756.40579584], [1094.69377576],
                                      [1094.69377576], [2638.335354], [4474.82351314],
                                      [4956.42872513], [4956.42872513], [6093.2556723],
                                      [7216.10719049], [7539.50807227], [7539.50807227],
                                      [8246.0164048], [9133.44034449], [9376.84106855],
                                      [9376.84106855], [9769.35335562], [10235.21824927],
                                      [10366.38189087], [10366.38189087], [10496.4112199],
                                      [10637.33808625], [10668.]]
        expected_masses_kg_climb = [[78716.87348217], [78630.56839717], [78501.99380919],
                                    [78461.17568729], [78461.17568729], [78282.66956759],
                                    [78079.48002171], [78024.13525417], [78024.13525417],
                                    [77874.53619474], [77690.00183752], [77632.85287533],
                                    [77632.85287533], [77506.6235826], [77341.2312645],
                                    [77291.7130525], [77291.7130525], [77206.37945353],
                                    [77094.28764289], [77060.26735586], [77060.26735586],
                                    [77024.49553459], [76976.26878085], [76961.2011732]]
        expected_ranges_m_climb = [[2020.172677], [6250.78127734], [14408.93179369],
                                   [17152.5734179], [17152.5734179], [31361.95647109],
                                   [51731.27335637], [58135.07173792], [58135.07173792],
                                   [77558.59088335], [105627.83510378], [114946.12296084],
                                   [114946.12296084], [136784.80542027], [167833.78601896],
                                   [177796.59419893], [177796.59419893], [195533.43046681],
                                   [220220.00511814], [228064.41792222], [228064.41792222],
                                   [236500.76050256], [248152.97050119], [251848.92128258]]
        expected_velocities_ms_climb = [[85.50188638], [142.5264918], [174.18896469],
                                        [178.36168293], [178.36168293], [189.05364242],
                                        [190.8409755], [191.10004849], [191.10004849],
                                        [196.39829028], [209.55408002], [213.5660258],
                                        [213.5660258], [220.53005733], [226.3795119],
                                        [227.73490457], [227.73490457], [229.83816653],
                                        [231.58844935], [231.71451888], [231.71451888],
                                        [231.66373105], [232.22063654], [232.77530606]]
        expected_thrusts_N_climb = [[189058.83725077],
                                    [170072.62422304],
                                    [158048.98358529],
                                    [154385.46295721],
                                    [154385.46295721],
                                    [136940.62457029],
                                    [116116.73671672],
                                    [110355.92368163],
                                    [110355.92368163],
                                    [97102.32305726],
                                    [84436.60128586],
                                    [81321.06998036],
                                    [81321.06998036],
                                    [75575.9138158],
                                    [69929.77214168],
                                    [68233.54819255],
                                    [68233.54819255],
                                    [66906.31800496],
                                    [64990.02370683],
                                    [64226.67297361],
                                    [64226.67297361],
                                    [63890.14810601],
                                    [62804.25695388],
                                    [61976.46190808]]

        expected_times_s_cruise = [[1208.43209632], [8852.14330109],
                                   [19398.90466014], [22736.91857014]]
        expected_altitudes_m_cruise = [[10668.], [10668.], [10668.], [10668.]]
        expected_masses_kg_cruise = [[76961.2011732], [71368.01841021],
                                     [64248.23842657], [62092.78518296]]
        expected_ranges_m_cruise = [[251848.92128258], [2031116.13639915],
                                    [4486141.73968515], [5263148.9492152]]
        expected_velocities_ms_cruise = [[232.77530606], [232.77530606],
                                         [232.77530606], [232.77530606]]
        expected_thrusts_N_cruise = [[42313.90859903],
                                     [39067.24395088],
                                     [35950.87557908],
                                     [35318.16856753]]

        expected_times_s_descent = [[22736.91857014], [22787.42540336], [22857.11452461],
                                    [22879.17089321], [22879.17089321], [22982.32020233],
                                    [23124.64519584], [23169.69056966], [23169.69056966],
                                    [23292.32489254], [23461.53522808], [23515.08972074],
                                    [23515.08972074], [23618.23902986], [23760.56402336],
                                    [23805.60939719], [23805.60939719], [23856.11623041],
                                    [23925.80535166], [24317.03614505]]
        expected_altitudes_m_descent = [[10668.], [10434.7028388], [9960.79015066],
                                        [9784.8421005], [9784.8421005], [8895.17761059],
                                        [7537.33136949], [7090.56116653], [7090.56116653],
                                        [5841.93815892], [3999.27122775], [3377.71804494],
                                        [3377.71804494], [2138.48831166], [663.47208745],
                                        [344.68854352], [344.68854352], [122.72341358],
                                        [10.668], [10.668]]
        expected_masses_kg_descent = [[62092.78518296], [62087.36440032], [62080.05896866],
                                      [62077.72767582], [62077.72767582], [62066.62726287],
                                      [62050.05526861], [62044.32456475], [62044.32456475],
                                      [62027.35019169], [62000.0762637], [61990.36426104],
                                      [61990.36426104], [61969.97789242], [61939.03538935],
                                      [61928.82648685], [61928.82648685], [61917.76865363],
                                      [61902.72792251], [60998.50259339]]
        expected_ranges_m_descent = [[5263148.9492152], [5274432.80167853], [5288836.80724673],
                                     [5293223.61558423], [
                                         5293223.61558423], [5313013.5415642],
                                     [5339348.96176189], [5347671.94897183], [
                                         5347671.94897183],
                                     [5370259.6353939], [5401419.10034695], [
                                         5411278.66121681],
                                     [5411278.66121681], [5430273.70362962], [
                                         5455475.25800144],
                                     [5462914.27715932], [5462914.27715932], [
                                         5470553.18510276],
                                     [5479541.42534675], [5481920.]]
        expected_velocities_ms_descent = [[232.77530606], [215.12062535], [200.3306912],
                                          [197.18540872], [197.18540872], [187.8799993],
                                          [184.51577102], [184.46836756], [184.46836756],
                                          [184.46836756], [184.46836756], [184.46836756],
                                          [184.46836756], [183.31878177], [168.97447568],
                                          [158.78398844], [158.78398844], [143.03675256],
                                          [113.40966999], [102.07377559]]
        expected_thrusts_N_descent = [[-10826.40055652],
                                      [-4402.33172934],
                                      [3254.14270948],
                                      [5235.45930457],
                                      [5235.45930457],
                                      [12007.31061759],
                                      [13032.5411879],
                                      [10960.76992506],
                                      [10960.76992506],
                                      [5671.53535014],
                                      [1824.89872243],
                                      [1525.32410053],
                                      [1525.32410053],
                                      [-329.33574306],
                                      [-9335.05085191],
                                      [-13485.38001055],
                                      [-13485.38001055],
                                      [-17391.90310498],
                                      [-14884.85752283],
                                      [-12642.43964087]]

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

        rtol = 1e-3

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

        # FLIGHT PATH
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

        # FLIGHT PATH
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
    z = ProblemPhaseTestCase()
    z.bench_test_full_mission_large_single_aisle_2()
