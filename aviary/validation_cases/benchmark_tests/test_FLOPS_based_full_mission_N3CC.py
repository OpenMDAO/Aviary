'''
NOTES:
Includes:
Takeoff, Climb, Cruise, Descent, Landing
Computed Aero
N3CC data
'''
import unittest
import warnings

import dymos as dm
import numpy as np
import openmdao.api as om
import scipy.constants as _units
from openmdao.core.driver import Driver
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from packaging import version

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.mission.flops_based.phases.climb_phase import Climb
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.mission.flops_based.phases.descent_phase import Descent
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.variables_in import VariablesIn
from aviary.subsystems.premission import CorePreMission
from aviary.interface.default_phase_info.height_energy import default_premission_subsystems, default_mission_subsystems
from aviary.utils.preprocessors import preprocess_crewpayload
from aviary.utils.aviary_values import AviaryValues


# benchmark based on N3CC (fixed cruise alt) FLOPS model


def run_trajectory(driver: Driver, sim=True):
    prob = om.Problem(model=om.Group())
    prob.driver = driver

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('N3CC')

    aviary_inputs.set_val(
        Mission.Landing.LIFT_COEFFICIENT_MAX, 2.4, units='unitless')
    aviary_inputs.set_val(
        Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2.0, units='unitless')
    aviary_inputs.set_val(
        Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, val=.0175,
        units='unitless')

    takeoff_fuel_burned = 577  # lbm TODO: where should this get connected from?
    takeoff_thrust_per_eng = 24555.5  # lbf TODO: where should this get connected from?
    takeoff_L_over_D = 17.35  # TODO: should this come from aero?

    aviary_inputs.set_val(
        Mission.Takeoff.FUEL_SIMPLE, takeoff_fuel_burned, units='lbm')
    aviary_inputs.set_val(
        Mission.Takeoff.LIFT_OVER_DRAG, takeoff_L_over_D, units='unitless')
    aviary_inputs.set_val(
        Mission.Design.THRUST_TAKEOFF_PER_ENG, takeoff_thrust_per_eng, units='lbf')

    alt_airport = 0  # ft
    cruise_mach = .785

    alt_i_climb = 0*_units.foot  # m
    alt_f_climb = 35000.0*_units.foot  # m
    mass_i_climb = 131000*_units.lb  # kg
    mass_f_climb = 126000*_units.lb  # kg
    v_i_climb = 198.44*_units.knot  # m/s
    v_f_climb = 455.49*_units.knot  # m/s
    # initial mach set to lower value so it can intersect with takeoff end mach
    # mach_i_climb = 0.3
    mach_i_climb = 0.2
    mach_f_climb = cruise_mach
    range_i_climb = 0*_units.nautical_mile  # m
    range_f_climb = 160.3*_units.nautical_mile  # m
    t_i_climb = 2 * _units.minute  # sec
    t_f_climb = 26.20*_units.minute  # sec
    t_duration_climb = t_f_climb - t_i_climb

    alt_i_cruise = 35000*_units.foot  # m
    alt_f_cruise = 35000*_units.foot  # m
    alt_min_cruise = 35000*_units.foot  # m
    alt_max_cruise = 35000*_units.foot  # m
    mass_i_cruise = 126000*_units.lb  # kg
    mass_f_cruise = 102000*_units.lb  # kg
    v_i_cruise = 455.49*_units.knot  # m/s
    v_f_cruise = 455.49*_units.knot  # m/s
    mach_min_cruise = cruise_mach
    mach_max_cruise = cruise_mach
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
    mach_i_descent = cruise_mach
    mach_f_descent = 0.3
    mass_i_descent = 102000*_units.pound
    mass_f_descent = 101000*_units.pound
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

    preprocess_crewpayload(aviary_inputs)

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

    prob.model.add_subsystem(
        'input_sink',
        VariablesIn(aviary_options=aviary_inputs),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )

    # suppress warnings:
    # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
    with warnings.catch_warnings():

        # Set initial default values for all LEAPS aircraft variables.
        warnings.simplefilter("ignore", om.PromotionWarning)
        set_aviary_initial_values(prob.model, aviary_inputs)

        warnings.simplefilter("ignore", om.PromotionWarning)
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
                 climb.interp(Dynamic.Mission.VELOCITY_RATE, ys=[0.25, 0.0]),
                 units='m/s**2')
    prob.set_val('traj.climb.controls:throttle',
                 climb.interp(Dynamic.Mission.THROTTLE, ys=[1.0, 1.0]),
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
                 cruise.interp(Dynamic.Mission.THROTTLE, ys=[0.8, 0.75]),
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
                   solution_record_file='N3CC_full_mission.db')
    prob.record("final")
    prob.cleanup()

    return prob


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_full_mission_N3CC_SNOPT(self):
        driver = _make_driver_SNOPT()

        self._do_run(driver)

    def _do_run(self, driver):
        prob = run_trajectory(driver=driver, sim=False)

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

        expected_times_s_climb = [[120.], [163.76276502], [224.14644864], [243.25767805],
                                  [243.25767805], [336.40844168], [464.93748315],
                                  [505.61648533], [505.61648533], [626.47047555],
                                  [793.2243154], [846.00133557], [846.00133557],
                                  [966.85532579], [1133.60916564], [1186.38618581],
                                  [1186.38618581], [1279.53694944], [1408.06599092],
                                  [1448.74499309], [1448.74499309], [1492.50775811],
                                  [1552.89144173], [1572.00267114]]
        expected_altitudes_m_climb = [[10.668], [0.], [490.31464488], [720.34099124],
                                      [720.34099124], [2288.74497628], [4816.40500694],
                                      [5464.87951761], [5464.87951761], [7025.37127556],
                                      [8437.78568802], [8802.40370192], [8802.40370192],
                                      [9511.72639581], [10229.15884616], [10382.97088649],
                                      [10382.97088649], [10560.94435598], [10660.37367157],
                                      [10668.], [10668.], [10668.], [10666.66785248], [10668.]]
        expected_masses_kg_climb = [[58584.62973209], [58553.30079295], [58486.19463517],
                                    [58461.90146352], [58461.90146352], [58339.61987692],
                                    [58184.14978151], [58144.99942835], [58144.99942835],
                                    [58041.54301581], [57927.14314383], [57894.10156256],
                                    [57894.10156256], [57825.05181264], [57740.72348501],
                                    [57716.38705856], [57716.38705856], [57675.62128122],
                                    [57622.73581734], [57606.53973509], [57606.53973509],
                                    [57589.20468199], [57563.40774787], [57554.3064769]]
        expected_ranges_m_climb = [[1469.37868509], [5605.65587129], [13572.59161311],
                                   [16576.93576947], [16576.93576947], [33254.75438651],
                                   [59051.61685726], [67185.62511739], [67185.62511739],
                                   [91756.36399005], [126369.47577928], [137576.80855552],
                                   [137576.80855552], [163427.52300246], [199432.46291748],
                                   [210871.48231952], [210871.48231952], [231238.09772766],
                                   [259750.51731468], [268890.90216347], [268890.90216347],
                                   [278771.58755554], [292561.32064207], [296983.91583583]]
        expected_velocities_ms_climb = [[77.35938331], [111.19901718], [151.93398907],
                                        [162.39904508], [162.39904508], [193.79386032],
                                        [202.92916455], [202.83097007], [202.83097007],
                                        [204.9548312], [211.13800405], [212.81677487],
                                        [212.81677487], [215.00414951], [216.65184841],
                                        [217.5075073], [217.5075073], [219.8725371],
                                        [223.95818114], [225.18659652], [225.18659652],
                                        [226.57877443], [230.5555138], [232.77530606]]
        expected_thrusts_N_climb = [[50269.53763097], [89221.79391729], [105944.78221328],
                                    [104742.61384096], [104742.61384096], [102539.17538465],
                                    [80894.5840539], [74287.02308141], [74287.02308141],
                                    [59565.00613161], [48715.65056776], [46229.53984695],
                                    [46229.53984695], [41551.41074213], [35939.62791236],
                                    [34345.57880903], [34345.57880903], [32008.34687504],
                                    [29668.0461344], [29061.17556918], [29061.17556918],
                                    [29571.25377648], [34034.12209303], [36366.13242869]]

        expected_times_s_cruise = [[1572.00267114], [10224.88157184],
                                   [22164.08837724], [25942.80651013]]
        expected_altitudes_m_cruise = [[10668.], [10668.], [10668.], [10668.]]
        expected_masses_kg_cruise = [[57554.3064769], [54168.16865196],
                                     [49649.4641858], [48253.00754766]]
        expected_ranges_m_cruise = [[296983.91583583], [2311160.45023764],
                                    [5090312.96846673], [5969905.23836297]]
        expected_velocities_ms_cruise = [[232.77530606], [232.77530606],
                                         [232.77530606], [232.77530606]]
        expected_thrusts_N_cruise = [[28599.34059358], [27655.863514],
                                     [26491.25212162], [26155.81047559]]

        expected_times_s_descent = [[25942.80651013], [26025.25799534], [26139.02421633],
                                    [26175.0308363], [26175.0308363], [26343.42020448],
                                    [26575.76316382], [26649.29891605], [26649.29891605],
                                    [26849.49721425], [27125.73000473], [27213.15673846],
                                    [27213.15673846], [27381.54610664], [27613.88906597],
                                    [27687.4248182], [27687.4248182], [27769.8763034],
                                    [27883.6425244], [27919.64914437]]
        expected_altitudes_m_descent = [[10668.], [10668.], [10142.61136584],
                                        [9909.20496834], [9909.20496834], [8801.80844468],
                                        [7273.17347505], [6802.45633892], [6802.45633892],
                                        [5619.8230507], [4125.52995372], [3659.42225117],
                                        [3659.42225117], [2735.99340162], [1425.57947265],
                                        [1009.41898888], [1009.41898888], [561.99822911],
                                        [94.31876192], [10.668]]
        expected_masses_kg_descent = [[48253.00754766], [48247.14199102], [48243.08209003],
                                      [48242.16864984], [48242.16864984], [48239.38631749],
                                      [48238.09565644], [48238.10218909], [48238.10218909],
                                      [48238.36804678], [48235.32226181], [48232.28685278],
                                      [48232.28685278], [48219.23890964], [48194.77880065],
                                      [48187.10389371], [48187.10389371], [48177.71262482],
                                      [48164.36107545], [48160.10979474]]
        expected_ranges_m_descent = [[5969905.23836297], [5987568.57791473], [6008764.81422574],
                                     [6015233.86544087], [6015233.86544087], [
                                         6044198.03874828],
                                     [6081891.04303138], [6093386.547795], [6093386.547795],
                                     [6123061.7076624], [6160342.11821306], [
                                         6171307.0346796],
                                     [6171307.0346796], [6192171.51704849], [
                                         6220846.02761427],
                                     [6230014.04117231], [6230014.04117231], [
                                         6240198.28047116],
                                     [6253503.06775949], [6257352.4]]
        expected_velocities_ms_descent = [[232.77530606], [198.63654674], [179.66599067],
                                          [176.95296882], [176.95296882], [167.50327408],
                                          [157.55154249], [154.07756214], [154.07756214],
                                          [142.57898741], [127.86775289], [125.11330664],
                                          [125.11330664], [123.46196696], [124.28423824],
                                          [124.03136002], [124.03136002], [122.19127487],
                                          [109.94487807], [102.07377559]]
        expected_thrusts_N_descent = [[4470.16874641], [2769.62977465], [1516.11359849],
                                      [1315.10754789], [1315.10754789], [652.82279778],
                                      [47.57054754], [-2.66626679e-13], [0.],
                                      [4.84263343e-14], [1.42410958e-14], [-2.42743721e-14],
                                      [0.], [0.], [1.77638636e-13], [2.84958986e-13],
                                      [-1.39722436e-13], [0.], [0.], [-5.27816527e-14]]

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


def _make_driver_IPOPT() -> Driver:
    driver = om.pyOptSparseDriver(optimizer='IPOPT')

    driver.opt_settings["max_iter"] = 100
    driver.opt_settings["tol"] = 1e-3
    driver.opt_settings['print_level'] = 4

    return driver


def _make_driver_SNOPT() -> Driver:
    driver = om.pyOptSparseDriver(optimizer='SNOPT')

    driver.opt_settings["Major iterations limit"] = 45
    driver.opt_settings["Major optimality tolerance"] = 1e-4
    driver.opt_settings["Major feasibility tolerance"] = 1e-6
    driver.opt_settings["iSumm"] = 6

    return driver


def _make_driver_SLSQP() -> Driver:
    driver = om.ScipyOptimizeDriver(optimizer='SLSQP')

    driver.opt_settings['maxiter'] = 100
    driver.opt_settings['ftol'] = 5.0e-3
    driver.opt_settings['eps'] = 1e-2

    return driver


if __name__ == '__main__':
    temp = ProblemPhaseTestCase()
    temp.bench_test_full_mission_N3CC_SNOPT()
