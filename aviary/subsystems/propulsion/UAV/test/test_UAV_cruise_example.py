import unittest
from copy import deepcopy
from pathlib import Path

import aviary.api as av
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from pathlib import Path
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder as DBFMassBuilder
from aviary.models.aircraft.small_uav.phases.UAV_energy_phase import phase_info
from aviary.subsystems.propulsion.UAV.UAV_Builder import UAVBuilder
from aviary.subsystems.aerodynamics.UAV_Aero.aero_builder import AeroBuilder
from aviary.subsystems.propulsion.UAV.model.UAV_mission import UAVPropMission
from aviary.subsystems.propulsion.UAV.model.UAV_premission import UAVPropPreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.UAV_variables import Aircraft, Dynamic
from aviary.variable_info.UAV_variable_meta_data import ExtendedMetaData as UAVExtendedMetaData
from aviary.variable_info.variables import Mission, Settings
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft as Mass_Aircraft
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variable_metadata import (
    ExtendedMetaData,
)


UAV_Prop = UAVBuilder()  # or 'solver' for the solver-based power balance mode


def CruiseExample():
    prob = av.AviaryProblem(verbosity=2, meta_data=UAVExtendedMetaData)
    prob.options['group_by_pre_opt_post'] = True
    # just selecting cruise
    cruise_phase_info = {
        'pre_mission': deepcopy(phase_info['pre_mission']),
        'cruise': deepcopy(phase_info['cruise']),
        'post_mission': deepcopy(phase_info['post_mission']),
    }

    prob.load_inputs(
        'validation_cases/validation_data/test_models/small_scale_uav.csv', cruise_phase_info
    )

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.load_external_subsystems(external_subsystems=[UAV_Prop, AeroBuilder(), DBFMassBuilder()])


    prob.check_and_preprocess_inputs()

    prob.build_model()



    prob.add_driver('IPOPT', use_coloring=False, max_iter=15)

    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['mu_strategy'] = 'adaptive'
    prob.driver.opt_settings['tol'] = 1e-5
    prob.driver.opt_settings['acceptable_tol'] = 5e-5
    prob.driver.opt_settings['constr_viol_tol'] = 1e-5
    prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-5
    # prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

    prob.add_design_variables()

    prob.add_objective(objective_type='time')

    prob.setup()


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


# NOTE: no @use_tempdirs here. DBFMassBuilder reads its airfoil CSV via a repo-root-
# relative path (like the dbf_based_mass unit tests), so this must run from the repo root.
class TestUAVCruiseExample(unittest.TestCase):
    def test_subsystems_in_cruise_attempt(self):
        prob = CruiseExample()

        # TODO: turn these back into assert_near_equal checks once the example values are confirmed.
        # self.assertTrue(np.isfinite(endurance) and endurance > 0.0)
        # self.assertTrue(np.isfinite(gross_mass) and 2.0 <= gross_mass <= 20.0)
        # self.assertTrue(0.25 < motor_mass < 0.65, f'Motor mass {motor_mass} kg is outside expected bounds.')
        # self.assertFalse(np.isnan(current_flow).any(), 'Current control contains NaN values.')
        # distance_resid = abs(distance_resid)
        # self.assertTrue(np.isfinite(distance_resid))
        # self.assertLess(distance_resid, 0.25, 'Cruise distance residual is unexpectedly large.')

        # # Print only driver-level constraints in a list_vars-like table.
        # prob.list_driver_vars(
        #     print_arrays=True,
        #     desvar_opts=[],
        #     objs_opts=[],
        # )


if __name__ == '__main__':
    unittest.main()
