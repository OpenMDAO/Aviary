from copy import deepcopy
import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import aviary.api as av
from aviary.models.external_subsystems.UAV.mass.mass_builder import MassBuilder
from aviary.models.missions.UAV_energy_phase import phase_info
from aviary.models.external_subsystems.UAV.propulsion.prop_builder import PropBuilder
from aviary.models.external_subsystems.UAV.aerodynamics.aero_builder import AeroBuilder
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import (
    ExtendedMetaData as UAVExtendedMetaData,
)
from aviary.variable_info.variables import Mission, Settings


def CruiseExample():
    prob = av.AviaryProblem(verbosity=2, meta_data=UAVExtendedMetaData)
    prob.options['group_by_pre_opt_post'] = True
    # just selecting cruise
    cruise_phase_info = {
        'pre_mission': deepcopy(phase_info['pre_mission']),
        'cruise': deepcopy(phase_info['cruise']),
        'post_mission': deepcopy(phase_info['post_mission']),
    }

    prob.load_inputs('aviary/models/aircraft/UAV/small_scale_uav.csv', cruise_phase_info)

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.load_external_subsystems(external_subsystems=[PropBuilder(), AeroBuilder(), MassBuilder()])

    prob.check_and_preprocess_inputs()

    prob.build_model()

    prob.add_driver('IPOPT', use_coloring=False, max_iter=150)

    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['mu_strategy'] = 'monotone'
    prob.driver.opt_settings['tol'] = 1e-5
    prob.driver.opt_settings['mu_init'] = 1.0
    prob.driver.opt_settings['limited_memory_max_history'] = 50
    prob.driver.opt_settings['acceptable_tol'] = 5e-5
    prob.driver.opt_settings['constr_viol_tol'] = 1e-5
    prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-5
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

    prob.add_design_variables()

    prob.add_objective(objective_type='time')

    # Add special solver scaling for small aircraft
    prob.model.set_output_solver_options('link_cruise_mass.mass', ref=1)
    prob.setup()

    # Add special rescaling for small aircraft
    prob.model.set_constraint_options(Mission.Constraints.MASS_RESIDUAL, ref=1)
    prob.model.set_design_var_options(Aircraft.Design.GROSS_MASS, lower=2, upper=100, ref=1)
    prob.model.set_design_var_options(Mission.GROSS_MASS, lower=2, upper=50, ref=1)
    prob.model.set_constraint_options('cruise_distance_constraint.distance_resid', ref=1)
    prob.model.traj.phases.cruise.rhs_all.set_constraint_options(
        'thrust_residual', ref=0.01, upper=0.01, lower=-0.01
    )
    reports_dir = prob.get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    prob.set_solver_print(level=0)
    prob.set_initial_guesses()

    # prob.set_val('traj.cruise.states:mass', 4.1, units='kg')
    n_state = prob.get_val('traj.cruise.states:energy_used').shape[0]
    energy_guess = np.linspace(0.0, 16.0, n_state).reshape(-1, 1)
    prob.set_val('traj.cruise.states:energy_used', energy_guess, units='W*h')
    prob.set_val('traj.cruise.controls:rpm_slack', 24, units='rev/s')
    prob.set_val('traj.cruise.controls:throttle', 0.35)
    prob.set_val('traj.cruise.controls:rpm_slack', 4000.0, units='rpm')
    prob.set_val('traj.cruise.controls:throttle', 0.3)

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.run_aviary_problem(run_driver=True)

    print(prob.get_val('traj.cruise.rhs_all.thrust_required', units='lbf'))
    print(prob.get_val('traj.cruise.rhs_all.thrust_residual', units='lbf'))
    print(prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
    print(prob.get_val('traj.cruise.rhs_all.thrust_net_total', units='lbf'))
    gross_mass = prob.get_val('mission:gross_mass', units='lbm')
    zero_fuel_mass = prob.get_val('mission:zero_fuel_mass', units='lbm')
    taxi_out_fuel = prob.get_val('mission:taxi:fuel_mass_taxi_out', units='lbm')
    takeoff_fuel = prob.get_val('mission:takeoff:fuel_mass', units='lbm')

    print('gross_mass:', gross_mass)
    print('zero_fuel_mass:', zero_fuel_mass)
    print('gross_mass - zero_fuel_mass:', gross_mass - zero_fuel_mass)
    print('taxi_out_fuel:', taxi_out_fuel)
    print('takeoff_fuel:', takeoff_fuel)
    print('gross_mass - taxi_out_fuel - takeoff_fuel:', gross_mass - taxi_out_fuel - takeoff_fuel)
    print('settings:problem_type:', prob.aviary_inputs.get_val(Settings.PROBLEM_TYPE))
    print('settings:equations_of_motion:', prob.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION))
    print('settings:mass_method:', prob.aviary_inputs.get_val(Settings.MASS_METHOD))
    return prob


@use_tempdirs
class TestUAVCruiseExample(unittest.TestCase):
    def test_subsystems_in_cruise_attempt(self):
        prob = CruiseExample()


if __name__ == '__main__':
    unittest.main()
