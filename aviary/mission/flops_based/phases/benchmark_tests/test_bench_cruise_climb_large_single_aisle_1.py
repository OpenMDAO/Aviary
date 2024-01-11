###############################################################
# NOTES:
# Includes:
# Cruise
# Computed Aero
# Large Single Aisle 1 data
###############################################################
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
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_trajectory_params
from aviary.variable_info.variables import Dynamic

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
            driver.opt_settings["Major iterations limit"] = 30
            driver.opt_settings["Major optimality tolerance"] = 1e-5
            driver.opt_settings["Major feasibility tolerance"] = 1e-5
            driver.opt_settings["iSumm"] = 6
            driver.opt_settings["Verify level"] = 3
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
    alt_min_cruise = 30000*_units.foot  # m
    alt_max_cruise = 36000*_units.foot  # m
    alt_i_cruise = alt_min_cruise
    alt_f_cruise = 36000*_units.foot  # m
    mass_i_cruise = 176765*_units.lb  # kg
    mass_f_cruise = 143521*_units.lb  # kg
    velocity_i_cruise = 465.55*_units.knot  # m/s
    velocity_f_cruise = 465.55*_units.knot  # m/s
    mach_i_cruise = 0.79
    mach_f_cruise = 0.79
    mach_min_cruise = 0.78999
    mach_max_cruise = 0.79001
    range_i_cruise = 160.3*_units.nautical_mile  # m
    range_f_cruise = 3243.9*_units.nautical_mile  # m
    t_i_cruise = 26.20*_units.minute  # sec
    t_f_cruise = 432.38*_units.minute  # sec
    t_duration_cruise = t_f_cruise - t_i_cruise

    prob.set_solver_print(level=0)

    transcription = dm.Radau(num_segments=8, order=3, compressed=True)

    ##########################################
    # Aircraft Input Variables and Options   #
    ##########################################

    aviary_inputs = get_flops_inputs('LargeSingleAisle1FLOPS')
    # engine_models = aviary_inputs['engine_models']

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
        # engine_models=engine_models,
        no_descent=True,
        velocity_f_cruise=velocity_f_cruise,
        mass_f_cruise=mass_f_cruise,
        range_f_cruise=range_f_cruise,

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

    cruise.add_objective(Dynamic.Mission.MASS, loc='final', ref=-mass_f_cruise)

    traj = setup_trajectory_params(prob.model, traj, aviary_inputs)

    # Set initial default values for all aircraft variables.
    set_aviary_initial_values(prob.model, aviary_inputs)

    # IVC = prob.model.add_subsystem('IVC', om.IndepVarComp(), promotes_outputs=['*'])
    # IVC.add_output('alt_i', val=32000, units='ft')
    prob.model.add_constraint(
        'traj.cruise.states:altitude',
        indices=[0], equals=alt_min_cruise, units='m', ref=alt_max_cruise)

    # Alternative implmentation of above
    # holds the first node in altitude constant
    # cruise.set_state_options(Dynamic.Mission.ALTITUDE, fix_initial=True)

    prob.setup()

    prob.set_val('traj.cruise.t_initial', t_i_cruise, units='s')
    prob.set_val('traj.cruise.t_duration', t_duration_cruise, units='s')

    prob.set_val(
        'traj.cruise.states:altitude', cruise.interp(
            Dynamic.Mission.ALTITUDE, ys=[
                alt_i_cruise, (alt_f_cruise + alt_i_cruise) / 2]), units='m')
    prob.set_val(
        'traj.cruise.states:velocity', cruise.interp(
            Dynamic.Mission.VELOCITY, ys=[
                velocity_i_cruise, velocity_f_cruise]), units='m/s')
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

    dm.run_problem(prob, simulate=False, make_plots=False,
                   solution_record_file='cruise_climb_max.db')
    # TODO Simulate=True is crashing here so turned it off

    prob.cleanup()
    prob.record("final")

    return prob


@unittest.skip('benchmark cruise climb currently broken')
class CruisePhaseTestCase(unittest.TestCase):
    def bench_test_cruise_climb_large_single_aisle_1(self):

        prob = run_trajectory()

        times = prob.get_val('traj.cruise.timeseries.time', units='s')
        altitudes = prob.get_val('traj.cruise.timeseries.states:altitude', units='m')
        masses = prob.get_val('traj.cruise.timeseries.states:mass', units='kg')
        ranges = prob.get_val('traj.cruise.timeseries.states:range', units='m')
        velocities = prob.get_val('traj.cruise.timeseries.states:velocity', units='m/s')
        thrusts = prob.get_val('traj.cruise.timeseries.thrust_net_total', units='N')

        expected_times_s = [
            [1572.0], [2654.1067424751645], [4147.193185981006],
            [4619.749940380141], [4619.749940380141], [5701.856682855307],
            [7194.943126361148], [7667.499880760283], [7667.499880760283],
            [8749.606623235448], [10242.693066741289], [10715.249821140425],
            [10715.249821140425], [11797.356563615589], [13290.44300712143],
            [13762.999761520567], [13762.999761520567], [14845.106503995732],
            [16338.192947501573], [16810.749701900706], [16810.749701900706],
            [17892.856444375873], [19385.942887881713], [19858.49964228085],
            [19858.49964228085], [20940.606384756014], [22433.692828261857],
            [22906.249582660992], [22906.249582660992], [23988.356325136156],
            [25481.442768642], [25953.999523041133]]
        expected_altitudes_m = [
            [9143.999999999998], [9708.208298692567], [10127.822388940609],
            [10214.217379879337], [10214.217379879337], [10369.633608560018],
            [10522.493213427248], [10561.102831034279], [10561.102831034279],
            [10628.89921313077], [10700.84689461149], [10720.94842484514],
            [10720.94842484514], [10759.7268947497], [10811.373374694173],
            [10828.933975559758], [10828.933975559758], [10866.85055516194],
            [10912.976573928156], [10924.787065056735], [10924.787065056735],
            [10948.28098891448], [10967.163582971652], [10969.578328436102],
            [10969.578328436102], [10971.188528219269], [10971.305320817934],
            [10972.40306787028], [10972.40306787028], [10972.800000000001],
            [10972.682828875833], [10972.8]]
        expected_masses_kg = [
            [80179.25528304999], [79388.27899969438], [78357.5492916318],
            [78038.12469387538], [78038.12469387538], [77318.48636094612],
            [76344.12665022336], [76039.40895912607], [76039.40895912607],
            [75347.90752400359], [74405.46275374181], [74109.60273338258],
            [74109.60273338258], [73436.39272100825], [72516.3728326044],
            [72227.22710472894], [72227.22710472894], [71569.02331279762],
            [70669.56072490575], [70386.99122373879], [70386.99122373879],
            [69743.47490624938], [68863.48019743375], [68586.78382987661],
            [68586.78382987661], [67956.17424058718], [67092.43136900471],
            [66820.49389265837], [66820.49389265837], [66200.47169407656],
            [65350.50275054499], [65082.74154168183]]
        expected_ranges_m = [
            [296875.60000000003], [554932.6549301515], [908399.8849826958],
            [1019938.0213070473], [1019938.0213070473], [1274830.9442516388],
            [1625714.5770829467], [1736624.0715069345], [1736624.0715069345],
            [1990386.259724133], [2340154.5629672543], [2450786.080862452],
            [2450786.080862452], [2704005.1890667905], [3053155.1477462808],
            [3163604.8984363866], [3163604.8984363866], [3416419.8294669725],
            [3765031.23567717], [3875315.9608939146], [3875315.9608939146],
            [4127790.908533788], [4476045.826943081], [4586252.017090351],
            [4586252.017090351], [4838601.188787049], [5186784.687764149],
            [5296985.604312349], [5296985.604312349], [5549329.03066934],
            [5897505.925045533], [6007702.8]]
        expected_velocities_ms = [
            [239.49901683172416], [237.57721365239863], [236.1272332817375],
            [235.83480578606836], [235.83480578606836], [235.29228140558052],
            [234.76087668521515], [234.63268475430843], [234.63268475430843],
            [234.3910317141664], [234.14254607637204], [234.07650600638797],
            [234.07650600638797], [233.93612811601653], [233.75522209810927],
            [233.69840302886462], [233.69840302886462], [233.56754384846963],
            [233.4030426703855], [233.35949436934354], [233.35949436934354],
            [233.28100688012185], [233.21763783283092], [233.20729273195352],
            [233.20729273195352], [233.19770503012535], [233.1994910278998],
            [233.19937577943125], [233.19937577943125], [233.19406389286172],
            [233.19249705434646], [233.197992365194]]
        expected_thrusts_N = [
            [46413.113617016286], [43959.216926887144], [42322.72191477366],
            [42001.69876565644], [42001.698765656445], [41407.67462380937],
            [40737.526959795134], [40531.856232353945], [40531.85623235394],
            [40132.12747513132], [39640.66854140172], [39484.44541552107],
            [39484.44541552107], [39159.04488467944], [38737.64967064259],
            [38601.13281610918], [38601.13281610918], [38291.27389642972],
            [37859.254019237305], [37729.15520426054], [37729.155204260525],
            [37427.990254061384], [37028.69942780689], [36907.64255965252],
            [36907.64255965253], [36649.2311134786], [36319.78199219473],
            [36213.12929993469], [36213.12929994349], [35988.316744926444],
            [35701.59647680185], [35614.98883340026]]

        expected_times_s = np.array(expected_times_s)
        expected_altitudes_m = np.array(expected_altitudes_m)
        expected_masses_kg = np.array(expected_masses_kg)
        expected_ranges_m = np.array(expected_ranges_m)
        expected_velocities_ms = np.array(expected_velocities_ms)
        expected_thrusts_N = np.array(expected_thrusts_N)

        # Objective

        rtol = 1e-2

        assert_near_equal(times[-1], expected_times_s[-1], tolerance=rtol)

        # Flight path

        # NOTE rtol = 0.01 = 1% different  from truth (first timeseries)
        #      atol = 1 = no more than +/-1 meter difference between values
        atol = 1e-2
        rtol = 1e-3

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
