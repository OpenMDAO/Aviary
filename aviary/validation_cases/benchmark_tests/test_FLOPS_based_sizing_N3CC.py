"""
Sizing the N3CC using the level 3 API.

Includes:
  Takeoff, Climb, Cruise, Descent, Landing
  Computed Aero
  N3CC data
"""
import unittest

import numpy as np
import scipy.constants as _units

import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.testing_utils import require_pyoptsparse

from aviary.mission.energy_phase import EnergyPhase
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_input_defaults
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_crewpayload, preprocess_propulsion
from aviary.utils.test_utils.assert_utils import warn_timeseries_near_equal
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.validation_cases.benchmark_utils import \
    compare_against_expected_values
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_trajectory_params, setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData


try:
    import pyoptsparse
except ImportError:
    pyoptsparse = None


from dymos.transcriptions.transcription_base import TranscriptionBase
if hasattr(TranscriptionBase, 'setup_polynomial_controls'):
    use_new_dymos_syntax = False
else:
    use_new_dymos_syntax = True

FLOPS = LegacyCode.FLOPS

# benchmark for simple sizing problem on the N3CC


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

    aviary_inputs = get_flops_inputs('N3CC')

    aviary_inputs.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX,
                          2.4, units="unitless")
    aviary_inputs.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX,
                          2.0, units="unitless")
    aviary_inputs.set_val(
        Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT,
        val=.0175, units="unitless")

    takeoff_fuel_burned = 577  # lbm TODO: where should this get connected from?
    takeoff_thrust_per_eng = 24555.5  # lbf TODO: where should this get connected from?
    takeoff_L_over_D = 17.35  # TODO: should this come from aero?

    aviary_inputs.set_val(Mission.Takeoff.FUEL_SIMPLE,
                          takeoff_fuel_burned, units='lbm')
    aviary_inputs.set_val(Mission.Takeoff.LIFT_OVER_DRAG,
                          takeoff_L_over_D, units="unitless")
    aviary_inputs.set_val(Mission.Design.THRUST_TAKEOFF_PER_ENG,
                          takeoff_thrust_per_eng, units='lbf')

    alt_airport = 0  # ft

    alt_i_climb = 0*_units.foot  # m
    alt_f_climb = 35000.0*_units.foot  # m
    mass_i_climb = 131000*_units.lb  # kg
    mass_f_climb = 126000*_units.lb  # kg
    # initial mach set to lower value so it can intersect with takeoff end mach
    # mach_i_climb = 0.3
    mach_i_climb = 0.2
    mach_f_climb = 0.79
    distance_i_climb = 0*_units.nautical_mile  # m
    distance_f_climb = 160.3*_units.nautical_mile  # m
    t_i_climb = 2 * _units.minute  # sec
    t_f_climb = 26.20*_units.minute  # sec
    t_duration_climb = t_f_climb - t_i_climb

    alt_i_cruise = 35000*_units.foot  # m
    alt_f_cruise = 35000*_units.foot  # m
    alt_min_cruise = 35000*_units.foot  # m
    alt_max_cruise = 35000*_units.foot  # m
    mass_i_cruise = 126000*_units.lb  # kg
    mass_f_cruise = 102000*_units.lb  # kg
    cruise_mach = 0.79
    distance_i_cruise = 160.3*_units.nautical_mile  # m
    distance_f_cruise = 3243.9*_units.nautical_mile  # m
    t_i_cruise = 26.20*_units.minute  # sec
    t_f_cruise = 432.38*_units.minute  # sec
    t_duration_cruise = t_f_cruise - t_i_cruise

    alt_i_descent = 35000*_units.foot
    # final altitude set to 35 to ensure landing is feasible point
    # alt_f_descent = 0*_units.foot
    alt_f_descent = 35*_units.foot
    mach_i_descent = 0.79
    mach_f_descent = 0.3
    mass_i_descent = 102000*_units.pound
    mass_f_descent = 101000*_units.pound
    distance_i_descent = 3243.9*_units.nautical_mile
    distance_f_descent = 3378.7*_units.nautical_mile
    t_i_descent = 432.38*_units.minute
    t_f_descent = 461.62*_units.minute
    t_duration_descent = t_f_descent - t_i_descent

    ##########################
    # Design Variables       #
    ##########################

    # Nudge it a bit off the correct answer to verify that the optimize takes us there.
    aviary_inputs.set_val(Mission.Design.GROSS_MASS, 135000.0, units='lbm')

    prob.model.add_design_var(Mission.Design.GROSS_MASS, units='lbm',
                              lower=100000.0, upper=200000.0, ref=135000)
    prob.model.add_design_var(Mission.Summary.GROSS_MASS, units='lbm',
                              lower=100000.0, upper=200000.0, ref=135000)

    takeoff_options = Takeoff(
        airport_altitude=alt_airport,  # ft
        # no units
        num_engines=aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)
    )

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

    # default subsystems
    engine = build_engine_deck(aviary_inputs)
    preprocess_propulsion(aviary_inputs, engine)
    default_mission_subsystems = get_default_mission_subsystems('FLOPS', engine)

    climb_options = EnergyPhase(
        'test_climb',
        user_options=AviaryValues({
            'initial_altitude': (alt_i_climb, 'm'),
            'final_altitude': (alt_f_climb, 'm'),
            'initial_mach': (mach_i_climb, 'unitless'),
            'final_mach': (mach_f_climb, 'unitless'),
            'fix_initial': (False, 'unitless'),
            'input_initial': (True, 'unitless'),
            'use_polynomial_control': (False, 'unitless'),
        }),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_climb,
    )

    cruise_options = EnergyPhase(
        'test_cruise',
        user_options=AviaryValues({
            'initial_altitude': (alt_min_cruise, 'm'),
            'final_altitude': (alt_max_cruise, 'm'),
            'initial_mach': (cruise_mach, 'unitless'),
            'final_mach': (cruise_mach, 'unitless'),
            'required_available_climb_rate': (300, 'ft/min'),
            'fix_initial': (False, 'unitless'),
        }),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_cruise,
    )

    descent_options = EnergyPhase(
        'test_descent',
        user_options=AviaryValues({
            'final_altitude': (alt_f_descent, 'm'),
            'initial_altitude': (alt_i_descent, 'm'),
            'initial_mach': (mach_i_descent, 'unitless'),
            'final_mach': (mach_f_descent, 'unitless'),
            'fix_initial': (False, 'unitless'),
            'use_polynomial_control': (False, 'unitless'),
        }),
        core_subsystems=default_mission_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription_descent,
    )

    landing_options = Landing(
        ref_wing_area=aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2'),
        Cl_max_ldg=aviary_inputs.get_val(
            Mission.Landing.LIFT_COEFFICIENT_MAX)  # no units
    )

    preprocess_crewpayload(aviary_inputs)

    prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engine)
    mass = CoreMassBuilder('core_mass', BaseMetaData, FLOPS)
    aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
    geom = CoreGeometryBuilder('core_geometry',
                               BaseMetaData,
                               code_origin=FLOPS)

    core_subsystems = [prop, geom, mass, aero]

    # Upstream static analysis for aero
    prob.model.add_subsystem(
        'pre_mission',
        CorePreMission(aviary_options=aviary_inputs,
                       subsystems=core_subsystems),
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
    climb.set_time_options(fix_initial=True, fix_duration=False, units='s',
                           duration_bounds=(t_duration_climb*0.5, t_duration_climb*2),
                           duration_ref=t_duration_climb)
    cruise.set_time_options(fix_initial=False, fix_duration=False, units='s',
                            duration_bounds=(t_duration_cruise*0.5, t_duration_cruise*2),
                            duration_ref=t_duration_cruise,
                            initial_bounds=(t_duration_climb*0.5, t_duration_climb*2))
    descent.set_time_options(
        fix_initial=False, fix_duration=False, units='s',
        duration_bounds=(t_duration_descent*0.5, t_duration_descent*2),
        duration_ref=t_duration_descent,
        initial_bounds=(
            (t_duration_cruise + t_duration_climb)*0.5,
            (t_duration_cruise + t_duration_climb)*2))

    traj.add_phase('climb', climb)

    traj.add_phase('cruise', cruise)

    traj.add_phase('descent', descent)

    prob.model.add_subsystem(
        'landing', landing, promotes_inputs=['aircraft:*', 'mission:*'],
        promotes_outputs=['mission:*'])

    traj.link_phases(
        ["climb", "cruise", "descent"],
        ["time", Dynamic.Vehicle.MASS, Dynamic.Mission.DISTANCE],
        connected=strong_couple,
    )

    # Need to declare dymos parameters for every input that is promoted out of the missions.
    externs = {'climb': {}, 'cruise': {}, 'descent': {}}
    for default_subsys in default_mission_subsystems:
        params = default_subsys.get_parameters(aviary_inputs=aviary_inputs,
                                               phase_info={})
        for key, val in params.items():
            for phname in externs:
                externs[phname][key] = val

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs,
                                   external_parameters=externs)

    ##################################
    # Connect in Takeoff and Landing #
    ##################################
    prob.model.connect(Mission.Takeoff.FINAL_MASS,
                       'traj.climb.initial_states:mass')
    prob.model.connect(Mission.Takeoff.GROUND_DISTANCE,
                       'traj.climb.initial_states:distance')

    prob.model.connect('traj.descent.states:mass',
                       Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])
    # TODO: approach velocity should likely be connected
    prob.model.connect('traj.descent.control_values:altitude', Mission.Landing.INITIAL_ALTITUDE,
                       src_indices=[-1])

    ##########################
    # Constraints            #
    ##########################

    ecomp = om.ExecComp('fuel_burned = initial_mass - descent_mass_final',
                        initial_mass={'units': 'lbm', 'shape': 1},
                        descent_mass_final={'units': 'lbm', 'shape': 1},
                        fuel_burned={'units': 'lbm', 'shape': 1})

    prob.model.add_subsystem('fuel_burn', ecomp,
                             promotes_inputs=[
                                 ('initial_mass', Mission.Design.GROSS_MASS)],
                             promotes_outputs=['fuel_burned'])

    prob.model.connect("traj.descent.states:mass",
                       "fuel_burn.descent_mass_final", src_indices=[-1])

    # TODO: need to add some sort of check that this value is less than the fuel capacity
    # TODO: need to update this with actual FLOPS value, this gives unrealistic
    # appearance of accuracy
    # TODO: the overall_fuel variable is the burned fuel plus the reserve, but should
    # also include the unused fuel, and the hierarchy variable name should be more clear
    ecomp = om.ExecComp('overall_fuel = fuel_burned + fuel_reserve',
                        fuel_burned={'units': 'lbm', 'shape': 1},
                        fuel_reserve={'units': 'lbm', 'val': 2173.},
                        overall_fuel={'units': 'lbm'})
    prob.model.add_subsystem('fuel_calc', ecomp,
                             promotes_inputs=['fuel_burned'],
                             promotes_outputs=['overall_fuel'])

    ecomp = om.ExecComp(
        'mass_resid = operating_empty_mass + overall_fuel + payload_mass '
        '- initial_mass',
        operating_empty_mass={'units': 'lbm'},
        overall_fuel={'units': 'lbm'},
        payload_mass={'units': 'lbm'},
        initial_mass={'units': 'lbm'},
        mass_resid={'units': 'lbm'})

    prob.model.add_subsystem(
        'mass_constraint', ecomp,
        promotes_inputs=[
            ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
            'overall_fuel',
            ('payload_mass', Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS),
            ('initial_mass', Mission.Design.GROSS_MASS)],
        promotes_outputs=['mass_resid'])

    prob.model.add_constraint('mass_resid', equals=0.0, ref=1.0)

    prob.model.add_subsystem(
        'gtow_constraint',
        om.EQConstraintComp(
            'GTOW',
            eq_units='lbm',
            normalize=True,
            add_constraint=True,
        ),
        promotes_inputs=[
            ('lhs:GTOW', Mission.Design.GROSS_MASS),
            ('rhs:GTOW', Mission.Summary.GROSS_MASS),
        ],
    )

    ##########################
    # Add Objective Function #
    ##########################

    # This is an example of a overall mission objective
    # create a compound objective that minimizes climb time and maximizes final mass
    # we are maxing final mass b/c we don't have an independent value for fuel_mass yet
    # we are going to normalize these (making each of the sub-objectives approx = 1 )
    # TODO: change the scaling on climb_duration
    prob.model.add_subsystem(
        "regularization",
        om.ExecComp(
            "reg_objective = fuel_mass/1500",
            reg_objective=0.0,
            fuel_mass={"units": "lbm", "shape": 1},
        ),
        promotes_inputs=[('fuel_mass', Mission.Design.FUEL_MASS)],
        promotes_outputs=['reg_objective']
    )

    prob.model.add_objective('reg_objective', ref=1)

    varnames = [
        Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
        Aircraft.Wing.SWEEP,
        Aircraft.Wing.TAPER_RATIO,
        Aircraft.Wing.THICKNESS_TO_CHORD,
        Mission.Design.GROSS_MASS,
        Mission.Summary.GROSS_MASS,
    ]
    set_aviary_input_defaults(prob.model, varnames, aviary_inputs)

    setup_model_options(prob, aviary_inputs)

    prob.setup(force_alloc_complex=True)

    set_aviary_initial_values(prob, aviary_inputs)

    ############################################
    # Initial Settings for States and Controls #
    ############################################

    prob.set_val('traj.climb.t_initial', t_i_climb, units='s')
    prob.set_val('traj.climb.t_duration', t_duration_climb, units='s')

    prob.set_val(
        'traj.climb.controls:altitude',
        climb.interp(Dynamic.Mission.ALTITUDE, ys=[alt_i_climb, alt_f_climb]),
        units='m',
    )
    prob.set_val(
        'traj.climb.controls:mach',
        climb.interp(Dynamic.Atmosphere.MACH, ys=[mach_i_climb, mach_f_climb]),
        units='unitless',
    )
    prob.set_val(
        'traj.climb.states:mass',
        climb.interp(Dynamic.Vehicle.MASS, ys=[mass_i_climb, mass_f_climb]),
        units='kg',
    )
    prob.set_val('traj.climb.states:distance', climb.interp(
        Dynamic.Mission.DISTANCE, ys=[distance_i_climb, distance_f_climb]), units='m')

    prob.set_val('traj.cruise.t_initial', t_i_cruise, units='s')
    prob.set_val('traj.cruise.t_duration', t_duration_cruise, units='s')

    if use_new_dymos_syntax:
        controls_str = 'controls'
    else:
        controls_str = 'polynomial_controls'

    prob.set_val(
        f'traj.cruise.{controls_str}:altitude',
        cruise.interp(Dynamic.Mission.ALTITUDE, ys=[alt_i_cruise, alt_f_cruise]),
        units='m',
    )
    prob.set_val(
        f'traj.cruise.{controls_str}:mach',
        cruise.interp(Dynamic.Atmosphere.MACH, ys=[cruise_mach, cruise_mach]),
        units='unitless',
    )
    prob.set_val(
        'traj.cruise.states:mass',
        cruise.interp(Dynamic.Vehicle.MASS, ys=[mass_i_cruise, mass_f_cruise]),
        units='kg',
    )
    prob.set_val('traj.cruise.states:distance', cruise.interp(
        Dynamic.Mission.DISTANCE, ys=[distance_i_cruise, distance_f_cruise]), units='m')

    prob.set_val('traj.descent.t_initial', t_i_descent, units='s')
    prob.set_val('traj.descent.t_duration', t_duration_descent, units='s')

    prob.set_val(
        'traj.descent.controls:altitude',
        descent.interp(Dynamic.Mission.ALTITUDE, ys=[alt_i_descent, alt_f_descent]),
        units='m',
    )
    prob.set_val(
        'traj.descent.controls:mach',
        descent.interp(Dynamic.Atmosphere.MACH, ys=[mach_i_descent, mach_f_descent]),
        units='unitless',
    )
    prob.set_val(
        'traj.descent.states:mass',
        descent.interp(Dynamic.Vehicle.MASS, ys=[mass_i_descent, mass_f_descent]),
        units='kg',
    )
    prob.set_val('traj.descent.states:distance', descent.interp(
        Dynamic.Mission.DISTANCE, ys=[distance_i_descent, distance_f_descent]), units='m')

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=sim, make_plots=False, simulate_kwargs={
                   'times_per_seg': 100, 'atol': 1e-9, 'rtol': 1e-9},
                   solution_record_file='N3CC_sizing.db')
    # prob.run_model()
    # z=prob.check_totals(method='cs', step=2e-40, compact_print=False)
    # exit()
    prob.record("final")
    prob.cleanup()

    return prob


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test sizing using N3CC data.
    """

    @require_pyoptsparse(optimizer="SNOPT")
    def bench_test_sizing_N3CC(self):

        prob = run_trajectory(sim=False)

        times_climb = prob.get_val('traj.climb.timeseries.time', units='s')
        altitudes_climb = prob.get_val(
            'traj.climb.timeseries.altitude', units='m')
        masses_climb = prob.get_val('traj.climb.timeseries.mass', units='kg')
        distances_climb = prob.get_val('traj.climb.timeseries.distance', units='m')
        velocities_climb = prob.get_val(
            'traj.climb.timeseries.velocity', units='m/s')
        thrusts_climb = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')
        times_cruise = prob.get_val('traj.cruise.timeseries.time', units='s')
        altitudes_cruise = prob.get_val(
            'traj.cruise.timeseries.altitude', units='m')
        masses_cruise = prob.get_val('traj.cruise.timeseries.mass', units='kg')
        distances_cruise = prob.get_val('traj.cruise.timeseries.distance', units='m')
        velocities_cruise = prob.get_val(
            'traj.cruise.timeseries.velocity', units='m/s')
        thrusts_cruise = prob.get_val(
            'traj.cruise.timeseries.thrust_net_total', units='N')
        times_descent = prob.get_val('traj.descent.timeseries.time', units='s')
        altitudes_descent = prob.get_val(
            'traj.descent.timeseries.altitude', units='m')
        masses_descent = prob.get_val('traj.descent.timeseries.mass', units='kg')
        distances_descent = prob.get_val('traj.descent.timeseries.distance', units='m')
        velocities_descent = prob.get_val(
            'traj.descent.timeseries.velocity', units='m/s')
        thrusts_descent = prob.get_val(
            'traj.descent.timeseries.thrust_net_total', units='N')

        print(thrusts_climb)
        print(thrusts_cruise)
        print(thrusts_descent)

        expected_times_s_climb = [[120.], [163.76268451], [224.14625705],
                                  [243.2574513], [243.2574513], [336.40804357],
                                  [464.9368486], [505.61577594], [505.61577594],
                                  [626.46954383], [793.22307692], [846.],
                                  [846.], [966.85376789], [1133.60730098],
                                  [1186.38422406], [1186.38422406], [1279.53481633],
                                  [1408.06362136], [1448.7425487], [1448.7425487],
                                  [1492.50523321], [1552.88880575], [1572.]]

        expected_altitudes_m_climb = [[0.], [321.52914489], [765.17373981],
                                      [905.58573725], [905.58573725], [1589.97314657],
                                      [2534.28808598], [2833.16053563], [2833.16053563],
                                      [3721.08615262], [4946.24227588], [5334.],
                                      [5334.], [6221.92561699], [7447.08174026],
                                      [7834.83946437], [7834.83946437], [8519.22687369],
                                      [9463.54181311], [9762.41426275], [9762.41426275],
                                      [10083.94340764], [10527.58800256], [10668.]]
        expected_masses_kg_climb = [[58331.64520977], [58289.14326023], [58210.1552262],
                                    [58183.99841695], [58183.99841695], [58064.21371439],
                                    [57928.02594919], [57893.85802859], [57893.85802859],
                                    [57803.5753272], [57701.17154376], [57671.70469428],
                                    [57671.70469428], [57609.99954455], [57532.67046087],
                                    [57509.28816794], [57509.28816794], [57468.36184014],
                                    [57411.39726843], [57392.97463799], [57392.97463799],
                                    [57372.80054331], [57344.30023996], [57335.11578186]]
        expected_distances_m_climb = [[1453.24648698], [4541.59528274], [9218.32974593],
                                      [10798.05432237], [10798.05432237], [19175.48832752],
                                      [32552.21767508], [37217.45114105], [37217.45114105],
                                      [52277.75199853], [75940.08052044], [84108.7210583],
                                      [84108.7210583], [104013.98891024], [134150.12359132],
                                      [144315.97475612], [144315.97475612], [
                                          162975.88568142],
                                      [190190.53671538], [199150.13943988], [
                                          199150.13943988],
                                      [208971.10963721], [222827.7518861], [227286.29149481]]
        expected_velocities_ms_climb = [[77.19291754], [132.35228283], [162.28279625],
                                        [166.11250634], [166.11250634], [176.30796789],
                                        [178.55049791], [178.67862048], [178.67862048],
                                        [181.75616771], [188.37687907], [189.78671051],
                                        [189.78671051], [190.67789763], [191.79702196],
                                        [193.51306342], [193.51306342], [199.28763405],
                                        [212.62699415], [218.00379223], [218.00379223],
                                        [224.21350812], [232.18822869], [234.25795132]]
        expected_thrusts_N_climb = [[82387.90445676], [105138.56964482], [113204.81413537],
                                    [110631.69881503], [110631.69881503], [97595.73505557],
                                    [74672.35033284], [68787.54639919], [68787.54639919],
                                    [56537.02658008], [46486.74146611], [43988.99250831],
                                    [43988.99250831], [39814.93772493], [36454.43040965],
                                    [35877.85248136], [35877.85248136], [35222.20684522],
                                    [34829.78305946], [34857.84665827], [34857.84665827],
                                    [34988.40432301], [35243.28462529], [35286.14075168]]

        expected_times_s_cruise = [[1572.01903685], [10224.85688577],
                                   [22164.00704809], [25942.70725365]]
        expected_altitudes_m_cruise = [[10668.], [10668.],
                                       [10668.], [10668.]]
        expected_masses_kg_cruise = [[57335.11578186], [53895.3524649],
                                     [49306.34176818], [47887.72131688]]
        expected_distances_m_cruise = [[1572.0], [10224.87753766],
                                       [22164.08246234], [25942.8]]

        expected_velocities_ms_cruise = [[234.25795132], [234.25795132],
                                         [234.25795132], [234.25795132]]
        expected_thrusts_N_cruise = [[28998.46944214], [28027.44677784],
                                     [26853.54343662], [26522.10071819]]

        expected_times_s_descent = [[25942.8], [25979.38684893], [26029.86923298],
                                    [26045.84673492], [26045.84673492], [26120.56747962],
                                    [26223.66685657], [26256.29745687], [26256.29745687],
                                    [26345.13302939], [26467.70798786], [26506.50254313],
                                    [26506.50254313], [26581.22328782], [26684.32266477],
                                    [26716.95326508], [26716.95326508], [26753.54011401],
                                    [26804.02249805], [26820.0]]
        expected_altitudes_m_descent = [[10668.], [10223.49681269], [9610.1731386],
                                        [9416.05829274], [9416.05829274], [8508.25644201],
                                        [7255.67517298], [6859.237484], [6859.237484],
                                        [5779.95090202], [4290.75570439], [3819.430516],
                                        [3819.430516], [2911.62866527], [1659.04739624],
                                        [1262.60970726], [1262.60970726], [818.10651995],
                                        [204.78284585], [10.668]]
        expected_masses_kg_descent = [[47887.72131688], [47887.72131688], [47887.72131688],
                                      [47887.72131688], [47887.72131688], [47887.72131688],
                                      [47887.72131688], [47887.72131688], [47887.72131688],
                                      [47887.87994829], [47886.14050369], [47884.40804261],
                                      [47884.40804261], [47872.68009732], [47849.34258173],
                                      [47842.09391697], [47842.09391697], [47833.150133],
                                      [47820.60083267], [47816.69389115]]
        expected_distances_m_descent = [[5937855.75657951], [5946333.90423671],
                                        [5957754.05343732], [5961300.12527496],
                                        [5961300.12527496], [5977437.03841276],
                                        [5998456.22230278], [6004797.661059],
                                        [6004797.661059], [6021279.17813816],
                                        [6042075.9952574], [6048172.56585825],
                                        [6048172.56585825], [6059237.47650301],
                                        [6073001.57667498], [6076985.65080086],
                                        [6076985.65080086], [6081235.7432792],
                                        [6086718.20915512], [6088360.09504693]]
        expected_velocities_ms_descent = [[234.25795132], [197.64415171], [182.5029101],
                                          [181.15994177], [181.15994177], [172.42254637],
                                          [156.92424445], [152.68023428], [152.68023428],
                                          [145.15327267], [141.83129659], [141.72294853],
                                          [141.72294853], [141.82174008], [140.7384589],
                                          [139.65952688], [139.65952688], [136.46721905],
                                          [115.68627038], [102.07377559]]
        expected_thrusts_N_descent = [[0.], [8.11038373e-13], [4.89024663e-13],
                                      [-3.80747087e-14], [0.], [1.10831527e-12],
                                      [7.77983996e-13], [-6.64041439e-16], [0.],
                                      [0.], [-5.7108371e-14], [-8.28280737e-14],
                                      [0.], [1.14382967e-13], [5.36023712e-14],
                                      [-3.04972888e-14], [0.], [-1.0682523e-13],
                                      [-6.30544276e-14], [5.37050667e-15]]
        expected_times_s_climb = np.array(expected_times_s_climb)
        expected_altitudes_m_climb = np.array(expected_altitudes_m_climb)
        expected_masses_kg_climb = np.array(expected_masses_kg_climb)
        expected_distances_m_climb = np.array(expected_distances_m_climb)
        expected_velocities_ms_climb = np.array(expected_velocities_ms_climb)
        expected_thrusts_N_climb = np.array(expected_thrusts_N_climb)

        expected_times_s_cruise = np.array(expected_times_s_cruise)
        expected_altitudes_m_cruise = np.array(expected_altitudes_m_cruise)
        expected_masses_kg_cruise = np.array(expected_masses_kg_cruise)
        expected_distances_m_cruise = np.array(expected_distances_m_cruise)
        expected_velocities_ms_cruise = np.array(expected_velocities_ms_cruise)
        expected_thrusts_N_cruise = np.array(expected_thrusts_N_cruise)

        expected_times_s_descent = np.array(expected_times_s_descent)
        expected_altitudes_m_descent = np.array(expected_altitudes_m_descent)
        expected_masses_kg_descent = np.array(expected_masses_kg_descent)
        expected_distances_m_descent = np.array(expected_distances_m_descent)
        expected_velocities_ms_descent = np.array(expected_velocities_ms_descent)
        expected_thrusts_N_descent = np.array(expected_thrusts_N_descent)

        expected_dict = {}
        expected_dict['times'] = np.concatenate((expected_times_s_climb,
                                                 expected_times_s_cruise,
                                                 expected_times_s_descent))
        expected_dict['altitudes'] = np.concatenate((expected_altitudes_m_climb,
                                                     expected_altitudes_m_cruise,
                                                     expected_altitudes_m_descent))
        expected_dict['masses'] = np.concatenate((expected_masses_kg_climb,
                                                  expected_masses_kg_cruise,
                                                  expected_masses_kg_descent))
        expected_dict['ranges'] = np.concatenate((expected_distances_m_climb,
                                                  expected_distances_m_cruise,
                                                  expected_distances_m_descent))
        expected_dict['velocities'] = np.concatenate((expected_velocities_ms_climb,
                                                      expected_velocities_ms_cruise,
                                                      expected_velocities_ms_descent))
        self.expected_dict = expected_dict

        # Check Objective and other key variables to a reasonably tight tolerance.

        # TODO: update truth values once everyone is using latest Dymos
        rtol = 5e-2

        # Check mission values.

        # NOTE rtol = 0.05 = 5% different from truth (first timeseries)
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
            times_climb, thrusts_climb, expected_times_s_climb,
            expected_thrusts_N_climb, abs_tolerance=atol, rel_tolerance=rtol)

        # CRUISE
        warn_timeseries_near_equal(
            times_cruise, thrusts_cruise, expected_times_s_cruise,
            expected_thrusts_N_cruise, abs_tolerance=atol, rel_tolerance=rtol)

        # DESCENT
        warn_timeseries_near_equal(
            times_descent, thrusts_descent, expected_times_s_descent,
            expected_thrusts_N_descent, abs_tolerance=atol, rel_tolerance=rtol)

        compare_against_expected_values(prob, self.expected_dict)


if __name__ == '__main__':
    z = ProblemPhaseTestCase()
    z.bench_test_sizing_N3CC()
