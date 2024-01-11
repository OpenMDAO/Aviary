'''
NOTES:
Includes:
Cruise
Computed Aero
Large Single Aisle 1 data
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
from openmdao.utils.testing_utils import use_tempdirs
from packaging import version
from aviary.variable_info.variables_in import VariablesIn

from aviary.utils.aviary_values import AviaryValues
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.interface.default_phase_info.height_energy import prop, aero, geom
from aviary.subsystems.premission import CorePreMission
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import (get_flops_inputs,
                                                      get_flops_outputs)
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Aircraft, Dynamic

mpl.rc('figure', max_open_warning=0)

try:
    import pyoptsparse

except ImportError:
    pyoptsparse = None


# benchmark based on Large Single Aisle 1 (fixed cruise alt) FLOPS model

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
    mass_i_cruise = 176765*_units.lb  # kg
    mass_f_cruise = 143521*_units.lb  # kg
    v_i_cruise = 455.49*_units.knot  # m/s
    v_f_cruise = 455.49*_units.knot  # m/s
    mach_i_cruise = 0.79
    mach_f_cruise = 0.79
    mach_min_cruise = 0.79
    mach_max_cruise = 0.79
    range_i_cruise = 160.3*_units.nautical_mile  # m
    range_f_cruise = 3243.9*_units.nautical_mile  # m
    t_i_cruise = 26.20*_units.minute  # sec
    t_f_cruise = 432.38*_units.minute  # sec
    t_duration_cruise = t_f_cruise - t_i_cruise

    prob.set_solver_print(level=2)

    transcription = dm.Radau(num_segments=1, order=3, compressed=True)

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')
    aviary_outputs = get_flops_outputs('LargeSingleAisle1FLOPS')

    # replace debug_no_mass flag with list of subsystem builders without mass
    core_subsystems = [prop, geom, aero]

    cruise_options = Cruise(
        'test_cruise',
        user_options=AviaryValues({
            'min_altitude': (alt_min_cruise, 'm'),
            'max_altitude': (alt_max_cruise, 'm'),
            'min_mach': (mach_min_cruise, 'unitless'),
            'max_mach': (mach_max_cruise, 'unitless'),
            'required_available_climb_rate': (300, 'ft/min'),
            'fix_initial': (True, 'unitless'),
            'fix_final': (True, 'unitless'),
        }),
        core_subsystems=core_subsystems,
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        transcription=transcription,
    )

    # Upstream static analysis for aero
    prob.model.add_subsystem(
        'core_premission',
        CorePreMission(aviary_options=aviary_inputs, subsystems=core_subsystems),
        promotes_inputs=['aircraft:*'],
        promotes_outputs=['aircraft:*', 'mission:*'])

    cruise = cruise_options.build_phase(aviary_options=aviary_inputs)

    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    traj.add_phase('cruise', cruise)

    cruise.timeseries_options['use_prefix'] = True

    cruise.set_time_options(
        fix_initial=True, fix_duration=False, units='s',
        duration_bounds=(1, t_duration_cruise*2), duration_ref=t_duration_cruise)

    cruise.add_objective(Dynamic.Mission.MASS, loc='final', ref=-1e4)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs, phases=['cruise'])

    prob.model.add_subsystem(
        'input_sink',
        VariablesIn(aviary_options=aviary_inputs),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )

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
                   solution_record_file='cruise_max.db')
    # TODO Simulate=True is crashing here so turned it off

    prob.cleanup()
    prob.record("final")

    return prob


@use_tempdirs
class CruisePhaseTestCase(unittest.TestCase):
    def bench_test_cruise_large_single_aisle_1(self):

        prob = run_trajectory()

        times = prob.get_val('traj.cruise.timeseries.time', units='s')
        altitudes = prob.get_val('traj.cruise.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.cruise.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.cruise.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.cruise.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.cruise.timeseries.thrust_net_total', units='N')

        expected_times_s = [[1572.], [10227.56555775],
                            [22170.47940152], [25950.37079939]]
        expected_altitudes_m = [[10668.], [10668.], [10668.], [10668.]]
        expected_masses_kg = [[80179.25528305], [
            74612.30190345], [67357.05992796], [65154.3328912]]
        expected_ranges_m = [[296875.6], [2324510.6550793],
                             [5122233.1849208], [6007702.8]]
        expected_velocities_ms = [[234.25795132], [
            234.25795132], [234.25795132], [234.25795132]]
        expected_thrusts_N = [[41817.82877662], [
            39634.49609004], [36930.60549609], [36149.38784885]]

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
    unittest.main()
