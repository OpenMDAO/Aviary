
from copy import deepcopy


import aviary.api as av
import numpy as np
import openmdao.api as om

from aviary.subsystems.aerodynamics.UAV_Aero.aero_builder import AeroBuilder
from aviary.subsystems.mass.UAV_mass.mass_builder import MassBuilder as DBFMassBuilder
from aviary.models.aircraft.small_uav.phases.UAV_energy_phase import phase_info
from aviary.subsystems.propulsion.UAV.UAV_Builder import UAVBuilder
from aviary.variable_info.UAV_variables import Aircraft, Dynamic
from aviary.variable_info.variables import  Settings


from aviary.variable_info.UAV_variable_meta_data import ExtendedMetaData


UAV_Prop = UAVBuilder()


def CruiseExample():
    prob = av.AviaryProblem(name='min_energy_cruise',verbosity=2, meta_data=ExtendedMetaData)
    prob.options['group_by_pre_opt_post'] = True
    # just selecting cruise
    cruise_phase_info = {
        'pre_mission': deepcopy(phase_info['pre_mission']),
        'cruise': {
        'subsystem_options': {'aerodynamics': {'method': 'external'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': True,

            'mach_initial': (0.0538, 'unitless'),

            'mach_bounds': ((0.05, 0.3), 'unitless'),
            # 'mach_ref': (0.05, 'unitless'),
            'mass_ref': (4.0, 'kg'),

            # 'alt_ref': (100, 'ft'),
            # 'mach_final': (0.05, 'unitless'),


            'altitude_optimize': True,
            'altitude_initial': (200.0, 'ft'),
            'altitude_bounds': ((50,400), 'ft'),
            # 'altitude_final': (200.0, 'ft'),
            'distance_initial': (0.0, 'm'),

            'distance_ref': (1000.0, 'm'),
            'target_distance': (1000.0, 'm'),
            'throttle_enforcement': 'control',

            # 'throttle_polynomial_order': 1,

            #Time
            'time_initial': (0.0, 's'),
            'time_duration_bounds': ((0,180), 's'),
        },
        'initial_guesses': {
            'distance': ([0, 2000], 'm'),
            'time': ([0, 60], 's'),
        },
    },
        'post_mission': deepcopy(phase_info['post_mission']),
    }

    prob.load_inputs(
        'validation_cases/validation_data/test_models/small_scale_uav.csv', cruise_phase_info
    )

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.load_external_subsystems(
        external_subsystems=[UAV_Prop, AeroBuilder(), DBFMassBuilder()]
    )

    prob.check_and_preprocess_inputs()

    prob.build_model()

    """Objective: Minimize energy consumption during cruise flight. This is done by adding an objective to the cruise phase that minimizes the energy constraint at the final time step. The energy constraint is defined as the integral of the power required to maintain level flight over the duration of the cruise phase. By minimizing this objective, we can find the optimal flight profile that minimizes energy consumption while still meeting all other constraints and requirements."""
    cruise_phase = prob.model.traj.phases.cruise
    cruise_phase.add_objective('rc_electric.energy_constraint', loc='final', ref = -100, units='W*hr')

    prob.add_driver('IPOPT', use_coloring=False, max_iter=0)

    prob.driver.opt_settings['print_level'] = 5
    prob.driver.opt_settings['mu_strategy'] = 'monotone'
    prob.driver.opt_settings['tol'] = 1e-5
    prob.driver.opt_settings['mu_init'] = 1.0
    prob.driver.opt_settings['limited_memory_max_history'] = 50
    prob.driver.opt_settings['acceptable_tol'] = 5e-5
    prob.driver.opt_settings['constr_viol_tol'] = 1e-5
    prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-5
    # prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

    prob.add_design_variables()



    prob.setup()

    prob.set_solver_print(level=0)
    prob.set_initial_guesses()

    # prob.set_val('traj.cruise.states:mass', 4.1, units='kg')

    prob.set_val('traj.cruise.controls:rpm_slack', 4000.0, units='rpm')
    prob.set_val('traj.cruise.controls:throttle', 1.0)
    prob.set_val('traj.cruise.controls:alpha', 3.0, units='deg')
    prob.set_val(Dynamic.Vehicle.Propulsion.THRUST_MAX, 9.69, units='lbf')

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.run_aviary_problem(run_driver=True)

    print('throttle:', prob.get_val('traj.cruise.controls:throttle', units='unitless'))
    print('battery voltage:', prob.get_val('traj.cruise.rhs_all.rc_electric.battery.voltage_out', units='V'))
    print('esc voltage out:', prob.get_val('traj.cruise.rhs_all.rc_electric.esc.voltage_out', units='V'))
    print('motor power:', prob.get_val('traj.cruise.rhs_all.rc_electric.motor.power', units='W'))
    print('prop power:', prob.get_val('traj.cruise.rhs_all.rc_electric.prop_power', units='W'))
    print('electric power in:', prob.get_val('traj.cruise.rhs_all.rc_electric.electric_power_in', units='W'))
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





if __name__ == '__main__':
    CruiseExample()
