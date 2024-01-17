'''
NOTES:
Includes:
Climb
Computed Aero
Large Single Aisle 1 data
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
from aviary.mission.flops_based.phases.climb_phase import Climb
from aviary.interface.default_phase_info.height_energy import prop, aero, geom
from aviary.subsystems.premission import CorePreMission
from aviary.utils.functions import set_aviary_initial_values
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Dynamic

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
            driver.opt_settings["tol"] = 1e-3
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
    alt_i_climb = 0*_units.foot  # m
    alt_f_climb = 35000.0*_units.foot  # m
    mass_i_climb = 180623*_units.lb  # kg
    mass_f_climb = 176765*_units.lb  # kg
    v_i_climb = 198.44*_units.knot  # m/s
    v_f_climb = 455.49*_units.knot  # m/s
    mach_i_climb = 0.3
    mach_f_climb = 0.79
    range_i_climb = 0*_units.nautical_mile  # m
    range_f_climb = 160.3*_units.nautical_mile  # m
    t_i_climb = 2 * _units.minute  # sec
    t_f_climb = 26.20*_units.minute  # sec
    t_duration_climb = t_f_climb - t_i_climb

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')

    climb_options = Climb(
        'test_climb',
        initial_altitude=alt_i_climb,
        final_altitude=alt_f_climb,
        no_descent=False,
        # note, Mach = 0.0 causes an error in aero, perhaps in other code
        initial_mach=mach_i_climb,
        final_mach=mach_f_climb,
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

    transcription = dm.Radau(num_segments=6, order=3, compressed=False)
    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    climb = climb_options.build_phase(MissionODE, transcription)

    traj.add_phase('climb', climb)

    climb.add_objective('time', loc='final', ref=1e3)
    # climb.add_objective(Dynamic.Mission.MASS, loc='final', ref=-1e4)

    climb.set_time_options(
        fix_initial=True, fix_duration=False, units='s',
        duration_bounds=(1, t_duration_climb*2), duration_ref=t_duration_climb)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs)

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)

    prob.setup()

    prob.set_val('traj.climb.t_initial', t_i_climb, units='s')
    prob.set_val('traj.climb.t_duration', t_duration_climb, units='s')

    prob.set_val('traj.climb.states:altitude', climb.interp(
        Dynamic.Mission.ALTITUDE, ys=[alt_i_climb, alt_f_climb]), units='m')
    prob.set_val('traj.climb.states:velocity', climb.interp(
        Dynamic.Mission.VELOCITY, ys=[v_i_climb, v_f_climb]), units='m/s')
    prob.set_val('traj.climb.states:mass', climb.interp(
        Dynamic.Mission.MASS, ys=[mass_i_climb, mass_f_climb]), units='kg')
    prob.set_val('traj.climb.states:range', climb.interp(
        Dynamic.Mission.RANGE, ys=[range_i_climb, range_f_climb]), units='m')

    prob.set_val('traj.climb.controls:velocity_rate',
                 climb.interp(Dynamic.Mission.VELOCITY_RATE, ys=[0.25, 0.05]),
                 units='m/s**2')
    prob.set_val('traj.climb.controls:throttle',
                 climb.interp(Dynamic.Mission.THROTTLE, ys=[0.5, 0.5]),
                 units='unitless')

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=False, make_plots=False,
                   solution_record_file='climb_max.db')
    prob.record("final")
    prob.cleanup()

    return prob


@unittest.skip('benchmark_climb_large_single_aisle_1 currently broken')
class ClimbPhaseTestCase(unittest.TestCase):
    def bench_test_climb_large_single_aisle_1(self):

        prob = run_trajectory()

        times = prob.get_val('traj.climb.timeseries.time', units='s')
        altitudes = prob.get_val('traj.climb.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.climb.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.climb.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.climb.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')

        expected_times_s = [
            [120.], [194.07139282], [296.27479672], [328.62182461],
            [328.62182461], [402.69321743], [504.89662133], [537.24364922],
            [537.24364922], [611.31504204], [713.51844594], [745.86547383],
            [745.86547383], [819.93686665], [922.14027055], [954.48729844],
            [954.48729844], [1028.55869126], [1130.76209516], [1163.10912305],
            [1163.10912305], [1237.18051587], [1339.38391977], [1371.73094766]]
        expected_altitudes_m = [
            [0.], [682.4957554], [2476.9915886], [3148.23528025],
            [3148.23528025], [4477.82410739], [5953.00480303], [6351.40548265],
            [6351.40548265], [7144.57711953], [8022.01978103], [8258.96785373],
            [8258.96785373], [8740.86702454], [9286.23944667], [9434.70712276],
            [9434.70712276], [9740.61479402], [10091.17192047], [10187.1054111],
            [10187.1054111], [10383.06677812], [10606.85289388], [10668.]]
        expected_masses_kg = [
            [81929.21464651], [81734.3159625], [81482.28032417], [81409.67409534],
            [81409.67409534], [81261.52268984], [81089.79910736], [81041.35712905],
            [81041.35712905], [80939.57475401], [80815.51316484], [80779.24867668],
            [80779.24867668], [80700.64682801], [80600.66210256], [80570.67210015],
            [80570.67210015], [80504.18318007], [80416.95604372], [80390.28585848],
            [80390.28585848], [80330.69704984], [80251.40682218], [80226.91650584]]
        expected_ranges_m = [
            [0.], [11134.82482615], [33305.36225275], [40552.08037408],
            [40552.08037408], [57694.75967874], [81567.92031544], [89129.68416299],
            [89129.68416299], [106463.26453256], [130389.14274257], [137962.93786],
            [137962.93786], [155308.03326502], [179243.7729914], [186819.85564691],
            [186819.85564691], [204168.96379317], [228108.3526042], [235685.29047396],
            [235685.29047396], [253035.99320713], [276976.90557005], [284554.20584747]]
        expected_velocities_ms = [
            [102.08605905], [190.58935413], [228.38194188], [230.69562115],
            [230.69562115], [233.30227629], [234.24467254], [234.25795132],
            [234.25795132], [234.25795132], [234.25795132], [234.25795132],
            [234.25795132], [234.25795132], [234.25795132], [234.25795132],
            [234.25795132], [234.25795132], [234.25795132], [234.25795132],
            [234.25795132], [234.25795132], [234.25795132], [234.25795132]]
        expected_thrusts_N = [
            [218457.50162414], [171332.33326933], [141120.77188326], [131766.72736631],
            [131766.72736631], [112828.24508393], [94178.64187388], [89529.25529378],
            [89529.25529378], [80383.22196309], [71047.86501907], [68765.74648505],
            [68765.74648505], [64112.90745184], [59068.27795964], [57877.0412593],
            [57877.0412593], [55418.69760271], [52595.0053262], [51821.03735429],
            [51821.03735429], [50238.39534494], [48428.26539664], [47933.1465328]]
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
    unittest.main()
