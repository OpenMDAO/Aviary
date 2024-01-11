'''
NOTES:
Includes:
Cruise
Computed Aero
Large Single Aisle 2 data
CURRENTLY REMOVED FROM BENCHMARK TEST SUITE
'''

import unittest

import dymos as dm
# Suppress the annoying warnings from matplotlib when running dymos.
import matplotlib as mpl
import numpy as np
import openmdao.api as om
import scipy.constants as _units

from dymos.utils.testing_utils import assert_timeseries_near_equal
from openmdao.utils.assert_utils import assert_near_equal

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.interface.default_phase_info.height_energy import prop, aero, geom
from aviary.subsystems.premission import CorePreMission
from aviary.utils.functions import set_aviary_initial_values
from aviary.validation_cases.validation_tests import (get_flops_inputs,
                                                      get_flops_outputs)
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Aircraft, Dynamic

mpl.rc('figure', max_open_warning=0)

try:
    import pyoptsparse

except ImportError:
    pyoptsparse = None


'''
NOTE benchmark currently hits iteration limit
     problem only converges when velocity ref & ref0 in energyphase is changed to 1e3.
     Changing velocity ref breaks some full-mission benchmarks, so is currently not
     implemented
'''


def run_trajectory():
    prob = om.Problem(model=om.Group())

    if pyoptsparse:
        driver = prob.driver = om.pyOptSparseDriver()
        driver.options["optimizer"] = "SNOPT"

        if driver.options["optimizer"] == "SNOPT":
            driver.opt_settings["Major iterations limit"] = 50
            driver.opt_settings["Major optimality tolerance"] = 1e-5
            driver.opt_settings["Major feasibility tolerance"] = 1e-6
            driver.opt_settings["iSumm"] = 6
        elif driver.options["optimizer"] == "IPOPT":
            driver.opt_settings["max_iter"] = 100
            driver.opt_settings["tol"] = 1e-6
            driver.opt_settings['print_level'] = 4

    else:
        driver = prob.driver = om.ScipyOptimizeDriver()
        opt_settings = prob.driver.opt_settings

        driver.options['optimizer'] = 'SLSQP'
        opt_settings['maxiter'] = 100
        opt_settings['ftol'] = 5.0e-3
        opt_settings['eps'] = 1e-2

    # TODO enable coloring once issue has been resolved:
    # https://github.com/OpenMDAO/OpenMDAO/issues/2507
    # driver.declare_coloring()

    ##########################
    # Problem Settings       #
    ##########################
    alt_i_cruise = 35000*_units.foot  # m
    alt_f_cruise = 35000*_units.foot  # m
    alt_min_cruise = 35000*_units.foot  # m
    alt_max_cruise = 35000*_units.foot  # m
    mass_i_cruise = 169730*_units.lb  # kg
    mass_f_cruise = 139810*_units.lb  # kg
    v_i_cruise = 452.61*_units.knot  # m/s
    v_f_cruise = 452.61*_units.knot  # m/s
    mach_i_cruise = 0.785
    mach_f_cruise = 0.785
    mach_min_cruise = 0.785
    mach_max_cruise = 0.785
    range_i_cruise = 116.6*_units.nautical_mile  # m
    range_f_cruise = 2558.3*_units.nautical_mile  # m
    t_i_cruise = 19.08*_units.minute  # sec
    t_f_cruise = 342.77*_units.minute  # sec
    t_duration_cruise = t_f_cruise - t_i_cruise  # sec

    prob.set_solver_print(level=2)

    transcription = dm.Radau(num_segments=1, order=3, compressed=True)

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')
    aviary_outputs = get_flops_outputs('LargeSingleAisle2FLOPS')

    cruise_options = Cruise(
        'test_cruise',
        min_altitude=alt_min_cruise,  # m
        max_altitude=alt_max_cruise,  # m
        # note, Mach = 0.0 causes an error in aero, perhaps in other code
        min_mach=mach_min_cruise,
        max_mach=mach_max_cruise,
        required_available_climb_rate=300*_units.foot/_units.minute,  # ft/min to m/s
        fix_initial=True,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        aviary_options=aviary_inputs,
    )

    # replace debug_no_mass flag with list of subsystem builders without mass
    core_subsystems = [prop, geom, aero]

    # Upstream static analysis for aero
    prob.model.add_subsystem(
        'core_premission',
        CorePreMission(aviary_options=aviary_inputs, subsystems=core_subsystems),
        promotes_inputs=['aircraft:*'],
        promotes_outputs=['aircraft:*', 'mission:*'])

    cruise = cruise_options.build_phase(MissionODE, transcription)

    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    traj.add_phase('cruise', cruise)

    cruise.set_time_options(
        fix_initial=True, fix_duration=False, units='s',
        duration_bounds=(1, t_duration_cruise*2), duration_ref=t_duration_cruise)

    cruise.add_objective(Dynamic.Mission.MASS, loc='final', ref=-1e4)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs)

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)
    key = Aircraft.Engine.THRUST_REVERSERS_MASS
    val, units = aviary_outputs.get_item(key)
    prob.model.set_input_defaults(key, val, units)

    prob.setup()

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

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=False, make_plots=False,
                   solution_record_file='cruise_large_single_aisle_2.db')
    # TODO Simulate=True is crashing here so turned it off

    prob.cleanup()
    prob.record("final")

    return prob


@unittest.skip('benchmark_cruise_large_single_aisle_2 currently broken')
class CruisePhaseTestCase(unittest.TestCase):
    def bench_test_cruise_large_single_asile_2(self):

        prob = run_trajectory()

        times = prob.get_val('traj.cruise.timeseries.time', units='s')
        altitudes = prob.get_val('traj.cruise.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.cruise.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.cruise.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.cruise.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.cruise.timeseries.thrust_net', units='N')

        expected_times_s = [[1144.8], [8042.22760495],
                            [17559.26991489], [20571.38126654]]
        expected_altitudes_m = [[10668.], [10668.], [10668.], [10668.]]
        expected_masses_kg = [[76988.2329601], [
            72122.85057432], [65811.40723995], [63888.48266274]]
        expected_ranges_m = [[215943.2], [1821494.02176252],
                             [4036826.45823737], [4737971.6]]
        expected_velocities_ms = [[232.77530606], [
            232.77530606], [232.77530606], [232.77530606]]
        expected_thrusts_N = [[40492.90002165], [
            38252.99661427], [35778.79636585], [35144.49009256]]

        expected_times_s = np.array(expected_times_s)
        expected_altitudes_m = np.array(expected_altitudes_m)
        expected_masses_kg = np.array(expected_masses_kg)
        expected_ranges_m = np.array(expected_ranges_m)
        expected_velocities_ms = np.array(expected_velocities_ms)
        expected_thrusts_N = np.array(expected_thrusts_N)

        # NOTE rtol = 0.01 = 1% different  from truth (first timeseries)
        #      atol = 1 = no more than +/-1 meter difference between values
        atol = 1e-2
        rtol = 1e-3

        # Objective
        assert_near_equal(times[-1], expected_times_s[-1], tolerance=rtol)

        # Flight path

        assert_timeseries_near_equal(
            expected_times_s, expected_masses_kg, times, masses,
            abs_tolerance=atol, rel_tolerance=rtol)
        assert_timeseries_near_equal(
            expected_times_s, expected_ranges_m, times, ranges,
            abs_tolerance=atol, rel_tolerance=rtol)
        assert_timeseries_near_equal(
            expected_times_s, expected_velocities_ms, times, velocities,
            abs_tolerance=atol, rel_tolerance=rtol)
        assert_timeseries_near_equal(
            expected_times_s, expected_thrusts_N, times, thrusts,
            abs_tolerance=atol, rel_tolerance=rtol)


if __name__ == "__main__":
    # Changes to hardcoded tabular aero data changed this benchmark. Update benchmark
    # test's expected values when aircraft-specific tabluar aero is avaliable
    unittest.main()
