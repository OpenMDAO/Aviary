'''
NOTES:
Includes:
Climb
Computed Aero
Large Single Aisle 2 data
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


def run_trajectory():
    prob = om.Problem(model=om.Group())

    if pyoptsparse:
        driver = prob.driver = om.pyOptSparseDriver()
        driver.options["optimizer"] = "SNOPT"
        if driver.options["optimizer"] == "SNOPT":
            driver.opt_settings["Major iterations limit"] = 40
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
    alt_i_climb = 35 * _units.foot  # m
    alt_f_climb = 35000.0 * _units.foot  # m
    mass_i_climb = 169172 * _units.lb  # kg
    mass_f_climb = 164160 * _units.lb  # kg
    v_i_climb = 198.44 * _units.knot  # m/s
    v_f_climb = 452.61 * _units.knot  # m/s
    mach_i_climb = 0.3
    mach_f_climb = 0.775
    range_i_climb = 0 * _units.nautical_mile  # m
    range_f_climb = 124 * _units.nautical_mile  # m
    t_i_climb = 0
    t_f_climb = 20.14 * _units.minute  # sec

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')
    transcription = dm.Radau(num_segments=6, order=3, compressed=False)
    climb_options = Climb(
        'test_climb',
        user_options=AviaryValues({
            'initial_altitude': (alt_i_climb, 'm'),
            'final_altitude': (alt_f_climb, 'm'),
            'initial_mach': (mach_i_climb, 'unitless'),
            'final_mach': (mach_f_climb, 'unitless'),
            'fix_initial': (True, 'unitless'),
            'fix_range': (False, 'unitless'),
            'input_initial': (False, 'unitless'),
        }),
        subsystem_options={'core_aerodynamics': {'method': 'computed'}},
        core_subsystems=[aero, prop],
        transcription=transcription,
    )

    # replace debug_no_mass flag with list of subsystem builders without mass
    core_subsystems = [prop, geom, aero]

    # Upstream static analysis for aero
    prob.model.add_subsystem(
        'core_premission',
        CorePreMission(aviary_options=aviary_inputs, subsystems=core_subsystems),
        promotes_inputs=['aircraft:*'],
        promotes_outputs=['aircraft:*', 'mission:*'])

    traj = prob.model.add_subsystem('traj', dm.Trajectory())

    climb = climb_options.build_phase(aviary_options=aviary_inputs)

    traj.add_phase('climb', climb)

    climb.timeseries_options['use_prefix'] = True

    climb.add_objective('time', loc='final', ref=t_f_climb)

    climb.set_time_options(
        fix_initial=True, fix_duration=False, units='s',
        duration_bounds=(t_f_climb * 0.5, t_f_climb * 2), duration_ref=t_f_climb)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs, phases=['climb'])

    prob.model.add_subsystem(
        'input_sink',
        VariablesIn(aviary_options=aviary_inputs),
        promotes_inputs=['*'],
        promotes_outputs=['*']
    )

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)

    prob.setup()

    prob.set_val('traj.climb.t_initial', t_i_climb, units='s')
    prob.set_val('traj.climb.t_duration', t_f_climb, units='s')

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

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=False, make_plots=False,
                   solution_record_file='climb_large_single_aisle_2.db')
    prob.record("final")
    prob.cleanup()

    return prob


@use_tempdirs
class ClimbPhaseTestCase(unittest.TestCase):
    def bench_test_climb_large_single_aisle_2(self):

        prob = run_trajectory()

        times = prob.get_val('traj.climb.timeseries.time', units='s')
        altitudes = prob.get_val('traj.climb.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.climb.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.climb.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.climb.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')

        expected_times_s = [[0.],
                            [47.24020828],
                            [112.42205385],
                            [133.05188512],
                            [133.05188512],
                            [180.2920934],
                            [245.47393897],
                            [266.10377023],
                            [266.10377023],
                            [313.34397852],
                            [378.52582409],
                            [399.15565535],
                            [399.15565535],
                            [446.39586363],
                            [511.5777092],
                            [532.20754047],
                            [532.20754047],
                            [579.44774875],
                            [644.62959432],
                            [665.25942558],
                            [665.25942558],
                            [712.49963387],
                            [777.68147944],
                            [798.3113107]]
        expected_altitudes_m = [[1.06680000e+01],
                                [1.64717443e+02],
                                [1.21077165e+03],
                                [1.65467452e+03],
                                [1.65467452e+03],
                                [2.63158548e+03],
                                [3.89195114e+03],
                                [4.26784031e+03],
                                [4.26784031e+03],
                                [5.08501440e+03],
                                [6.12319963e+03],
                                [6.43302565e+03],
                                [6.43302565e+03],
                                [7.11281142e+03],
                                [7.98114971e+03],
                                [8.23811059e+03],
                                [8.23811059e+03],
                                [8.78587957e+03],
                                [9.45185516e+03],
                                [9.64164118e+03],
                                [9.64164118e+03],
                                [1.00442465e+04],
                                [1.05301137e+04],
                                [1.06680000e+04]]
        expected_masses_kg = [[76735.12841764],
                              [76606.6291851],
                              [76423.55875817],
                              [76369.21098493],
                              [76369.21098493],
                              [76249.27821197],
                              [76096.52180626],
                              [76051.03001076],
                              [76051.03001076],
                              [75951.61028918],
                              [75824.47689696],
                              [75786.47518169],
                              [75786.47518169],
                              [75703.26334226],
                              [75596.63038983],
                              [75564.65308165],
                              [75564.65308165],
                              [75493.66861479],
                              [75400.85356264],
                              [75372.67886333],
                              [75372.67886333],
                              [75310.24572772],
                              [75228.35359487],
                              [75203.34794937]]
        expected_ranges_m = [[0.],
                             [6656.35106052],
                             [19411.34037117],
                             [23606.44276229],
                             [23606.44276229],
                             [33597.9222712],
                             [47757.51502804],
                             [52308.8364503],
                             [52308.8364503],
                             [62843.02681099],
                             [77579.88662088],
                             [82280.86896803],
                             [82280.86896803],
                             [93082.94529627],
                             [108032.30577631],
                             [112765.20449737],
                             [112765.20449737],
                             [123607.6358803],
                             [138572.23956024],
                             [143309.37828794],
                             [143309.37828794],
                             [154158.17439593],
                             [169129.72020176],
                             [173868.65393385]]
        expected_velocities_ms = [[102.08635556],
                                  [173.4078533],
                                  [206.56979636],
                                  [209.71234319],
                                  [209.71234319],
                                  [215.12096513],
                                  [220.69657248],
                                  [222.14669648],
                                  [222.14669648],
                                  [225.06833215],
                                  [228.02045426],
                                  [228.64470582],
                                  [228.64470582],
                                  [229.49739505],
                                  [229.80567178],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554],
                                  [229.81001554]]
        expected_thrusts_N = [[183100.97258682],
                              [163996.9938714],
                              [144753.36497874],
                              [140427.87965941],
                              [140427.87965941],
                              [131136.19372391],
                              [119398.47487166],
                              [115939.50935006],
                              [115939.50935006],
                              [108449.59352749],
                              [99269.67371293],
                              [96533.89180294],
                              [96533.89180294],
                              [90902.4892038],
                              [84031.52375462],
                              [81889.41102505],
                              [81889.41102505],
                              [77138.3352612],
                              [71330.53467419],
                              [69720.96587699],
                              [69720.96587699],
                              [66405.48655242],
                              [62587.8225939],
                              [61541.80174696]]

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
    test = ClimbPhaseTestCase()
    test.bench_test_climb_large_single_aisle_2()
