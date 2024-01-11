'''
NOTES:
Includes:
Descent
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
from aviary.mission.flops_based.phases.descent_phase import Descent
from aviary.interface.default_phase_info.height_energy import prop, aero, geom
from aviary.subsystems.premission import CorePreMission
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Dynamic

mpl.rc('figure', max_open_warning=0)


try:
    import pyoptsparse

except ImportError:
    pyoptsparse = None


'''
NOTE benchmark only reaches 0-3 after 50 iterations with bad plots
     problem easily converges with 0-1 when velocity ref & ref0 in energyphase is changed
     to 1e3.
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
            driver.opt_settings["Major optimality tolerance"] = 5e-3
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
    alt_i = 35000*_units.foot
    alt_f = 35*_units.foot
    v_i = 452.61*_units.knot
    v_f = 198.44*_units.knot
    mach_i = 0.785
    mach_f = 0.3
    mass_i = 140515*_units.pound
    mass_f = 140002*_units.pound
    range_i = 2830.8*_units.nautical_mile
    range_f = 2960.0*_units.nautical_mile
    t_i_descent = 0.0
    t_f_descent = 2000.0

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')

    descent_options = Descent(
        'test_descent',
        final_altitude=alt_f,
        initial_altitude=alt_i,
        # note, Mach = 0.0 causes an error in aero, perhaps in other code
        final_mach=mach_f,
        initial_mach=mach_i,
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

    transcription = dm.Radau(num_segments=5, order=3, compressed=True)

    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    descent = descent_options.build_phase(MissionODE, transcription)

    descent.add_objective(Dynamic.Mission.RANGE, ref=-1e5, loc='final')
    traj.add_phase('descent', descent)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs)

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)

    prob.setup()

    # set initial conditions
    prob.set_val('traj.descent.t_initial', t_i_descent, units='s')
    prob.set_val('traj.descent.t_duration', t_f_descent, units='s')

    prob.set_val('traj.descent.states:altitude', descent.interp(
        Dynamic.Mission.ALTITUDE, ys=[alt_i, alt_f]), units='m')
    prob.set_val('traj.descent.states:velocity', descent.interp(
        Dynamic.Mission.VELOCITY, ys=[v_i, v_f]), units='m/s')
    prob.set_val('traj.descent.states:mass', descent.interp(
        Dynamic.Mission.MASS, ys=[mass_i, mass_f]), units='kg')
    prob.set_val('traj.descent.states:range', descent.interp(
        Dynamic.Mission.RANGE, ys=[range_i, range_f]), units='m')

    prob.set_val('traj.descent.controls:velocity_rate',
                 descent.interp(Dynamic.Mission.VELOCITY_RATE, ys=[0.0, 0.0]),
                 units='m/s**2')
    prob.set_val('traj.descent.controls:throttle',
                 descent.interp(Dynamic.Mission.THROTTLE, ys=[0.0, 0.0]),
                 units='unitless')

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=False, make_plots=False,
                   solution_record_file='descent_large_single_aisle_2.db')
    prob.record("final")
    prob.cleanup()

    return prob


@unittest.skip('benchmark_descent_large_single_aisle_2 currently broken')
class DescentPhaseTestCase(unittest.TestCase):
    def bench_test_descent_large_single_aisle_2(self):

        prob = run_trajectory()

        times = prob.get_val('traj.descent.timeseries.time', units='s')
        altitudes = prob.get_val('traj.descent.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.descent.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.descent.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.descent.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.descent.timeseries.thrust_net', units='N')

        expected_times_s = [
            [0.], [142.02539503], [337.99145239], [400.01403952],
            [400.01403952], [542.03943455], [738.0054919], [800.02807903],
            [800.02807903], [942.05347406], [1138.01953142], [1200.04211855],
            [1200.04211855], [1342.06751358], [1538.03357093], [1600.05615806],
            [1600.05615806], [1742.08155309], [1938.04761045], [2000.07019758]]
        expected_altitudes_m = [
            [10668.], [9548.97609287], [8538.42241675], [8275.01276129],
            [8275.01276129], [7601.31902636], [6519.47805911], [6141.13701298],
            [6141.13701298], [5205.26802003], [3963.90368243], [3634.87695679],
            [3634.87695679], [3055.62551952], [2221.91470683], [1840.14771729],
            [1840.14771729], [870.95857222], [10.668], [10.668]]
        expected_masses_kg = [
            [63736.53187055], [63688.55056043], [63592.93766122], [63558.62366657],
            [63558.62366657], [63480.12720033], [63368.68894184], [63331.87300229],
            [63331.87300229], [63243.53017169], [63103.08546544], [63051.9162013],
            [63051.9162013], [62922.36463692], [62748.94842397], [62704.12656931],
            [62704.12656931], [62627.76510694], [62566.74093186], [62554.55364275]]
        expected_ranges_m = [
            [5242641.6], [5275682.08024547], [5321287.43849558],
            [5335722.05453227], [5335722.05453227], [5368775.18998604],
            [5414378.40648958], [5428810.78044196], [5428810.78044196],
            [5461857.5756481], [5507456.7101919], [5521890.39254914],
            [5521890.39254914], [5554951.82532665], [5599686.46586361],
            [5613370.90489204], [5613370.90489204], [5642618.8601419],
            [5674799.0551216], [5682083.57281856]]
        expected_velocities_ms = [
            [232.77530606], [232.77530606], [232.77530606], [232.77530606],
            [232.77530606], [232.77530606], [232.77530606], [232.77530606],
            [232.77530606], [232.77530606], [232.77530606], [232.77530606],
            [232.77530606], [232.05265856], [223.03560479], [216.62970611],
            [216.62970611], [192.53473927], [130.45545087], [102.07377559]]
        expected_thrusts_N = [
            [10348.85799036], [20322.69938272], [27901.58834456],
            [28262.62341568], [28262.62341568], [28062.43245824],
            [28590.47833586], [28815.02361611], [28815.02361611],
            [30869.45928917], [38383.90751338], [41973.2222718],
            [41973.2222718], [46287.80542225], [36032.4169276],
            [28755.10323868], [28755.10323868], [13691.34754408], [0.], [0.]]

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
