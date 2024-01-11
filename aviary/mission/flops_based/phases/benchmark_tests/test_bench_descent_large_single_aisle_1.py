'''
NOTES:
Includes:
Descent
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
NOTE descent profile is improved when velocity ref in EnergyPhase change to 1e3
     Implementing that velocity ref change breaks several mission benchmarks
'''
# benchmark based on Large Single Aisle 1 (fixed cruise alt) FLOPS model


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
    alt_i_descent = 35000*_units.foot
    alt_f_descent = 0*_units.foot
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

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')

    descent_options = Descent(
        'test_descent',
        final_altitude=alt_f_descent,
        initial_altitude=alt_i_descent,
        # note, Mach = 0.0 causes an error in aero, perhaps in other code
        final_mach=mach_f_descent,
        initial_mach=mach_i_descent,
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

    # descent.add_objective(Dynamic.Mission.MASS, ref=-1e4, loc='final')
    descent.add_objective(Dynamic.Mission.RANGE, ref=-1e5, loc='final')

    traj.add_phase('descent', descent)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs)

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)

    prob.setup()

    # set initial conditions
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
                 descent.interp(Dynamic.Mission.VELOCITY_RATE, ys=[0.0, 0.0]),
                 units='m/s**2')
    prob.set_val('traj.descent.controls:throttle',
                 descent.interp(Dynamic.Mission.THROTTLE, ys=[0.0, 0.0]),
                 units='unitless')

    # Turn off solver printing so that the SNOPT output is readable.
    prob.set_solver_print(level=0)

    dm.run_problem(prob, simulate=False, make_plots=False,
                   solution_record_file='descent_max.db')
    prob.record("final")
    prob.cleanup()

    return prob


@unittest.skip('benchmark_descent_large_single_aisle_1 currently broken')
class DescentPhaseTestCase(unittest.TestCase):
    def bench_test_descent_large_single_aisle_1(self):

        prob = run_trajectory()

        times = prob.get_val('traj.descent.timeseries.time', units='s')
        altitudes = prob.get_val('traj.descent.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.descent.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.descent.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.descent.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.descent.timeseries.thrust_net', units='N')

        expected_times_s = [
            [25942.8], [26067.38110599], [26239.2776049], [26293.68225907],
            [26293.68225907], [26418.26336507], [26590.15986397], [26644.56451815],
            [26644.56451815], [26769.14562414], [26941.04212305], [26995.44677722],
            [26995.44677722], [27120.02788321], [27291.92438212], [27346.3290363],
            [27346.3290363], [27470.91014229], [27642.80664119], [27697.21129537]]
        expected_altitudes_m = [
            [10668.], [9630.58727675], [8607.44716937], [8325.35146852],
            [8325.35146852], [7618.26503578], [6519.12597312], [6144.16296733],
            [6144.16296733], [5222.63590813], [3906.74184382], [3501.58245681],
            [3501.58245681], [2693.56449806], [1613.14062128], [1227.34540813],
            [1227.34540813], [472.05630168], [0.], [0.]]
        expected_masses_kg = [
            [65100.03053477], [65068.4860871], [65009.01063053], [64988.47429787],
            [64988.47429787], [64942.20142996], [64875.19680192], [64851.91888669],
            [64851.91888669], [64793.85853886], [64706.58329093], [64678.30905669],
            [64678.30905669], [64614.11135757], [64531.47572247], [64507.94243863],
            [64507.94243863], [64464.03072407], [64420.28475648], [64408.7201039]]
        expected_ranges_m = [
            [6007702.8], [6036868.3987895], [6077125.1955833],
            [6089867.31136598], [6089867.31136598], [6119045.21397484],
            [6159302.44709504], [6172042.94318126], [6172042.94318126],
            [6201214.75820313], [6241466.48424821], [6254207.73380962],
            [6254207.73380962], [6283402.65691348], [6322851.72600445],
            [6334884.03220066], [6334884.03220066], [6360443.88994895],
            [6388386.24507732], [6394712.53233535]]
        expected_velocities_ms = [
            [234.25795132], [234.25795132], [234.25795132], [234.25795132],
            [234.25795132], [234.25795132], [234.25795132], [234.25795132],
            [234.25795132], [234.25795132], [234.25795132], [234.25795132],
            [234.25795132], [233.47298378], [223.67831268], [216.71997882],
            [216.71997882], [190.9857791], [129.0266439], [102.08605905]]
        expected_thrusts_N = [
            [9516.07730964], [18867.54655988], [26134.27560127], [26569.54809341],
            [26569.54809341], [26513.67959312], [27279.65638053], [27558.344662],
            [27558.344662], [29558.29975054], [37141.0454636], [40960.56830135],
            [40960.56830135], [46197.31464087], [36817.32495137], [29341.51748247],
            [29341.51748247], [14369.16584473], [0.], [2.27381718e-11]]

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
