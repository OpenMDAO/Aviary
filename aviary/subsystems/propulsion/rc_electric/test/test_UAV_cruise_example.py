import unittest

import aviary.api as av
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.UAV_Aero.custom_aero_builder import CustomAeroBuilder
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder as DBFMassBuilder
from aviary.models.aircraft.small_uav.phases.UAV_energy_phase import get_cruise_phase_info
from aviary.subsystems.propulsion.rc_electric.UAV_Builder import RCBuilder
from aviary.subsystems.propulsion.rc_electric.model.UAV_mission import RCPropMission
from aviary.subsystems.propulsion.rc_electric.model.UAV_premission import RCPropPreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.dbf_variables import Aircraft, Dynamic
from aviary.variable_info.variables import Mission
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft as Mass_Aircraft


#This is where you set the power balance mode for the RCPropMission. Options are 'feedforward' or 'solver'.
#Example for solver, RCBBuilder(power_balance_mode='solver')
rc_prop = RCBuilder()  # or 'solver' for the solver-based power balance mode


def _build_cruise_phase_info():
    phase_kwargs = {
        'external_subsystems': [CustomAeroBuilder()],
    }

    # Only solver mode needs a bounded throttle phase override here.
    if rc_prop.power_balance_mode == 'solver':
        phase_kwargs['throttle_enforcement'] = 'bounded'
        phase_kwargs['throttle_bounds'] = ((0.2, 0.9), 'unitless')

    return get_cruise_phase_info(**phase_kwargs)


def CruiseExample():
    prob = av.AviaryProblem(verbosity=0)
    prob.options['group_by_pre_opt_post'] = True

    prob.load_inputs(
        'validation_cases/validation_data/test_models/small_scale_uav.csv',
        _build_cruise_phase_info(),
    )
    
    prob.load_external_subsystems(external_subsystems=[rc_prop, CustomAeroBuilder(), DBFMassBuilder()])
    

    prob.check_and_preprocess_inputs()

    

    prob.build_model()

    

    prob.add_driver('IPOPT', use_coloring=False, max_iter=0)
    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['mu_strategy'] = 'adaptive'
    prob.driver.opt_settings['tol'] = 1e-6
    prob.driver.opt_settings['acceptable_tol'] = 5e-6
    prob.driver.opt_settings['constr_viol_tol'] = 1e-6
    prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-6
    prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

    prob.add_design_variables()

    # Set UAV-scale gross mass DV for this test setup.
    # prob.model.add_design_var(Mission.GROSS_MASS, units='kg', lower=0.5, upper=20, ref=4)

    # Geometry design variables at UAV scale.
    

    
    # Keep direct geometric bounds.



    # prob.model.add_constraint(Mission.TOTAL_FUEL_MASS, equals=0.0, units='kg', ref=1e2)
    prob.add_objective(objective_type='time')
    

    prob.setup()



    prob.set_solver_print(level=0)
    prob.set_initial_guesses()
    
    prob.set_val('traj.cruise.states:mass', 4.1, units='kg')


    
    if rc_prop.power_balance_mode == 'feedforward':
        prob.set_val('traj.cruise.controls:rpm_slack', 4000.0, units='rpm')
        prob.set_val('traj.cruise.controls:throttle', 0.7, units='unitless')
        prob.set_val('traj.cruise.controls:current_flow', 40.0, units='A')
        prob.set_val('traj.cruise.states:mass', 4.4, units='kg')
        

    prob.run_aviary_problem(run_driver=True)


    return prob

    
# class MeanPowerComp(om.ExplicitComponent):
#     # Simple smoother objective input: average cruise electric power over the full phase.
#     def setup(self):
#         self.add_input('p_cruise_kw', shape_by_conn=True, units='kW')
#         self.add_output('p_avg_kw', val=1.0, units='kW')
#         self.declare_partials('p_avg_kw', 'p_cruise_kw', method='fd')

#     def compute(self, inputs, outputs):
#         outputs['p_avg_kw'] = np.mean(inputs['p_cruise_kw'])


# NOTE: no @use_tempdirs here. DBFMassBuilder reads its airfoil CSV via a repo-root-
# relative path (like the dbf_based_mass unit tests), so this must run from the repo root.
class TestUAVCruiseExample(unittest.TestCase):
    def test_subsystems_in_cruise_attempt(self):
        prob = CruiseExample()

        # print('\n=== UAV Cruise Debug (metric/SI) ===')

        # t_duration = prob.get_val('traj.cruise.t_duration', units='s')
        # ts_time = prob.get_val('traj.cruise.timeseries.time', units='s')

        # mission_gross_mass = prob.get_val('mission:gross_mass', units='kg')
        # design_gross_mass = prob.get_val('aircraft:design:gross_mass', units='kg')
        # motor_mass = prob.get_val('aircraft:engine:motor:mass', units='kg')
        # battery_mass = prob.get_val('aircraft:battery:mass', units='kg')
        # mass_resid = prob.get_val('mission:constraints:mass_residual', units='kg')
        # link_cruise_mass = prob.get_val('link_cruise_mass.mass')

        # distance_resid = prob.get_val('cruise_distance_constraint.distance_resid', units='m')
        # states_distance = prob.get_val('traj.cruise.states:distance', units='m')
        # ts_distance = prob.get_val('traj.cruise.timeseries.distance', units='m')
        # ts_velocity = prob.get_val('traj.cruise.timeseries.velocity', units='m/s')
        # controls_mach = prob.get_val('traj.cruise.controls:mach')

        # esc_current_out = prob.get_val('traj.cruise.rhs_all.rc_electric.esc.current_out', units='A')
        # ts_electric_power = prob.get_val('traj.cruise.timeseries.electric_power_in_total', units='W')

        # current_constraint_nominal = prob.get_val('traj.cruise.rhs_all.rc_electric.current_constraint_nominal')
        # prop_rpm_constraint = prob.get_val('traj.cruise.rhs_all.rc_electric.prop.rpm_constraint', units='rev/s')
        # defect_distance = prob.get_val('traj.cruise.collocation_constraint.defects:distance')
        # defect_mass = prob.get_val('traj.cruise.collocation_constraint.defects:mass')

        # print('traj.cruise.t_duration =', np.array2string(np.asarray(t_duration), precision=6, separator=', '), 's')
        # print('traj.cruise.timeseries.time =', np.array2string(np.asarray(ts_time).flatten(), precision=6, separator=', '), 's')

        # print('mission:gross_mass =', np.array2string(np.asarray(mission_gross_mass), precision=6, separator=', '), 'kg')
        # print('aircraft:design:gross_mass =', np.array2string(np.asarray(design_gross_mass), precision=6, separator=', '), 'kg')
        # print('aircraft:engine:motor:mass =', np.array2string(np.asarray(motor_mass), precision=6, separator=', '), 'kg')
        # print('aircraft:battery:mass =', np.array2string(np.asarray(battery_mass), precision=6, separator=', '), 'kg')
        # print('mission:constraints:mass_residual =', np.array2string(np.asarray(mass_resid), precision=6, separator=', '), 'kg')
        # print('link_cruise_mass.mass =', np.array2string(np.asarray(link_cruise_mass), precision=6, separator=', '))

        # print('cruise_distance_constraint.distance_resid =', np.array2string(np.asarray(distance_resid), precision=6, separator=', '), 'm')
        # print('traj.cruise.states:distance =', np.array2string(np.asarray(states_distance).flatten(), precision=6, separator=', '), 'm')
        # print('traj.cruise.timeseries.distance =', np.array2string(np.asarray(ts_distance).flatten(), precision=6, separator=', '), 'm')
        # print('traj.cruise.timeseries.velocity =', np.array2string(np.asarray(ts_velocity).flatten(), precision=6, separator=', '), 'm/s')
        # print('traj.cruise.controls:mach =', np.array2string(np.asarray(controls_mach).flatten(), precision=6, separator=', '))

        # print('traj.cruise.rhs_all.rc_electric.esc.current_out =', np.array2string(np.asarray(esc_current_out).flatten(), precision=6, separator=', '), 'A')
        # print('traj.cruise.timeseries.electric_power_in_total =', np.array2string(np.asarray(ts_electric_power).flatten(), precision=6, separator=', '), 'W')

        # print('traj.cruise.rhs_all.rc_electric.current_constraint_nominal =', np.array2string(np.asarray(current_constraint_nominal).flatten(), precision=6, separator=', '))
        # print('traj.cruise.rhs_all.rc_electric.prop.rpm_constraint =', np.array2string(np.asarray(prop_rpm_constraint).flatten(), precision=6, separator=', '), 'rev/s')
        # print('traj.cruise.collocation_constraint.defects:distance =', np.array2string(np.asarray(defect_distance).flatten(), precision=6, separator=', '))
        # print('traj.cruise.collocation_constraint.defects:mass =', np.array2string(np.asarray(defect_mass).flatten(), precision=6, separator=', '))

        # print('=== End Debug Dump ===\n')
       
       

        # TODO: turn these back into assert_near_equal checks once the example values are confirmed.
        # self.assertTrue(np.isfinite(endurance) and endurance > 0.0)
        # self.assertTrue(np.isfinite(gross_mass) and 2.0 <= gross_mass <= 20.0)
        # self.assertTrue(0.25 < motor_mass < 0.65, f'Motor mass {motor_mass} kg is outside expected bounds.')
        # self.assertFalse(np.isnan(current_flow).any(), 'Current control contains NaN values.')
        # distance_resid = abs(distance_resid)
        # self.assertTrue(np.isfinite(distance_resid))
        # self.assertLess(distance_resid, 0.25, 'Cruise distance residual is unexpectedly large.')
        prob.model.traj.phases.cruise.rhs_all.list_vars(
            units=True,
            print_arrays=True,
            prom_name=True,
            includes=['*power_net*'],
        )

        # Print only driver-level constraints in a list_vars-like table.
        prob.list_driver_vars(
            print_arrays=True,
            desvar_opts=[],
            objs_opts=[],
        )
        


if __name__ == '__main__':
    unittest.main()
