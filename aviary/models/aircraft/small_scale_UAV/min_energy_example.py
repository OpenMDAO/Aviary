from copy import deepcopy

import aviary.api as av
from aviary.variable_info.enums import AtmosphereModel
import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.models.external_subsystems.UAV.aerodynamics.aero_builder import AeroBuilder
from aviary.models.external_subsystems.UAV.mass.mass_builder import MassBuilder as DBFMassBuilder
from aviary.models.missions.UAV_energy_phase import phase_info
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import (
    Aircraft,
    Dynamic,
    Settings,
    Mission,
)
from aviary.models.external_subsystems.UAV.propulsion.prop_builder import PropBuilder

from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import (
    ExtendedMetaData,
)
from aviary.variable_info.enums import Transcription

UAV_Prop = PropBuilder()


@use_tempdirs
def CruiseExample():
    prob = av.AviaryProblem(name='min_energy_cruise', verbosity=2, meta_data=ExtendedMetaData)
    prob.options['group_by_pre_opt_post'] = True
    # just selecting cruise
    cruise_phase_info = {
        'pre_mission': deepcopy(phase_info['pre_mission']),
        'cruise': deepcopy(phase_info['cruise']),
        'post_mission': deepcopy(phase_info['post_mission']),
    }
    # adjust phase info for the cruise example
    cruise_phase_info['cruise']['user_options']['distance_initial'] = (0.0, 'm')
    cruise_phase_info['cruise']['user_options']['time_initial'] = (0.0, 's')
    cruise_phase_info['cruise']['user_options']['distance_initial'] = (0, 'm')
    cruise_phase_info['cruise']['user_options']['time_duration_bounds'] = ((None, None), 's')
    cruise_phase_info['cruise']['user_options']['mach_initial'] = (None, 'unitless')
    cruise_phase_info['cruise']['initial_guesses']['time'] = ([0, 2000], 's')
    cruise_phase_info['cruise']['initial_guesses']['distance'] = ([0, 30], 'km')
    # cruise_phase_info['cruise']['user_options']['transcription'] = Transcription.COLLOCATION

    prob.load_inputs('aviary/models/aircraft/UAV/small_scale_uav.csv', cruise_phase_info)

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.load_external_subsystems(external_subsystems=[UAV_Prop, AeroBuilder(), DBFMassBuilder()])

    prob.aviary_inputs.set_val(
        Settings.ATMOSPHERE_MODEL, AtmosphereModel.STANDARD, units='unitless'
    )
    prob.check_and_preprocess_inputs()

    prob.build_model()

    """Objective: Minimize energy consumption during cruise flight. This is done by adding an objective to the cruise phase that minimizes the energy constraint at the final time step. The energy constraint is defined as the integral of the power required to maintain level flight over the duration of the cruise phase. By minimizing this objective, we can find the optimal flight profile that minimizes energy consumption while still meeting all other constraints and requirements."""
    cruise_phase = prob.model.traj.phases.cruise

    cruise_phase.add_objective('distance', loc='final', ref=-10000, units='m')

    driver = 'SNOPT'  # set 'SNOPT' or 'IPOPT'
    prob.add_driver(driver, use_coloring=True, max_iter=100)
    if driver == 'SNOPT':
        prob.driver.opt_settings['Major optimality tolerance'] = 5e-5
        prob.driver.opt_settings['Major feasibility tolerance'] = 1e-6
        prob.driver.opt_settings['Major step limit'] = 1.0
    elif driver == 'IPOPT':
        prob.driver.opt_settings['mu_strategy'] = 'monotone'
        prob.driver.opt_settings['tol'] = 1e-5
        prob.driver.opt_settings['mu_init'] = 1.0
        prob.driver.opt_settings['limited_memory_max_history'] = 50
        prob.driver.opt_settings['acceptable_tol'] = 5e-5
        prob.driver.opt_settings['constr_viol_tol'] = 1e-5
        prob.driver.opt_settings['acceptable_constr_viol_tol'] = 5e-5
        # Report exactly which Jacobian entries go NaN/Inf instead of a bare EXIT message.
        prob.driver.opt_settings['check_derivatives_for_naninf'] = 'yes'
        prob.driver.opt_settings['recalc_y'] = 'yes'
        prob.driver.opt_settings['recalc_y_feas_tol'] = 1e-2

    # prob.driver.opt_settings['acceptable_iter'] = 0
    # prob.driver.opt_settings['print_level'] = 5
    # prob.driver.options['debug_print'] = ['desvars', 'objs', 'nl_cons', 'ln_cons']

    prob.add_design_variables()

    # Add special solver scaling for small aircraft
    prob.model.set_output_solver_options(
        'link_cruise_mass.mass', ref=1
    )  # energy_state_problem_configurator.py
    # prob.model.set_output_solver_options('throttle_balance', res_ref=1e3) # energy_state_ODE.py

    prob.setup()

    # use to see all the constraints in the problem
    # print("")
    # print("====== A list of Constraints on the problem ======")
    # for system in prob.model.system_iter(recurse=True, include_self=True):
    #     for name, meta in system._responses.items():
    #         if meta['type'] == 'con':
    #             print(f'{system.pathname}: {name}')
    # exit()
    # an extremely verbose way to viewing constraints
    # prob.final_setup()
    # constraints = prob.model.get_constraints(recurse=True)
    # for name, meta in constraints.items():
    #     print(name, meta)
    # exit()

    # Add special rescaling for small aircraft
    prob.model.set_constraint_options(Mission.Constraints.MASS_RESIDUAL, ref=1)  # aviary_group.py
    prob.model.set_design_var_options(
        Aircraft.Design.GROSS_MASS, lower=2, upper=50, ref=1
    )  # aviary_group.py
    prob.model.set_design_var_options(
        Mission.GROSS_MASS, lower=2, upper=50, ref=1
    )  # aviary_group.py
    # prob.model.set_constraint_options('cruise_distance_constraint.distance_resid', ref=1) # aviary_group.py
    # prob.model.set_constraint_options('cruise_duration_constraint.duration_resid', ref=10) # aviary_group.py
    # prob.model.set_constraint_options(Mission.Constraints.RANGE_RESIDUAL, ref=1) # aviary_group.py
    prob.model.traj.phases.cruise.rhs_all.set_constraint_options(
        'thrust_residual',
        ref=1.0,
        equals=0.0,
    )

    prob.set_solver_print(level=0)
    prob.set_initial_guesses()

    # prob.set_val('traj.cruise.states:mass', 4.1, units='kg')

    prob.set_val('traj.cruise.controls:rpm_slack', 2877.0, units='rpm')
    prob.set_val('traj.cruise.controls:throttle', 0.561)
    prob.set_val('traj.cruise.controls:mach', 0.0538)
    prob.set_val('traj.cruise.rhs_all.thrust_net_max_total', 9.69, units='lbf')

    number = prob.aviary_inputs.get_val(Aircraft.Wing.WETTED_AREA, units='m**2')
    print('Wetted Area:', number)

    prob.run_aviary_problem(run_driver=True, simulate=True)

    """Debug Print"""
    # print('throttle:', prob.get_val('traj.cruise.controls:throttle', units='unitless'))
    # print('battery voltage:', prob.get_val('traj.cruise.rhs_all.rc_electric.battery.voltage_out', units='V'))
    # print('esc voltage out:', prob.get_val('traj.cruise.rhs_all.rc_electric.esc.voltage_out', units='V'))
    # print('motor power:', prob.get_val('traj.cruise.rhs_all.rc_electric.motor.power', units='W'))
    # print('prop power:', prob.get_val('traj.cruise.rhs_all.rc_electric.prop_power', units='W'))
    # print('electric power in:', prob.get_val('traj.cruise.rhs_all.electric_power_in_total', units='W'))
    # print(prob.get_val('traj.cruise.rhs_all.thrust_required', units='lbf'))
    # print(prob.get_val('traj.cruise.rhs_all.thrust_residual', units='lbf'))
    # print(prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
    # print(prob.get_val('traj.cruise.rhs_all.thrust_net_total', units='lbf'))
    # gross_mass = prob.get_val('mission:gross_mass', units='lbm')
    # zero_fuel_mass = prob.get_val('mission:zero_fuel_mass', units='lbm')
    # taxi_out_fuel = prob.get_val('mission:taxi:fuel_mass_taxi_out', units='lbm')
    # takeoff_fuel = prob.get_val('mission:takeoff:fuel_mass', units='lbm')

    # print('gross_mass:', gross_mass)
    # print('zero_fuel_mass:', zero_fuel_mass)
    # print('gross_mass - zero_fuel_mass:', gross_mass - zero_fuel_mass)
    # print('taxi_out_fuel:', taxi_out_fuel)
    # print('takeoff_fuel:', takeoff_fuel)
    # print('gross_mass - taxi_out_fuel - takeoff_fuel:', gross_mass - taxi_out_fuel - takeoff_fuel)

    # print('settings:problem_type:', prob.aviary_inputs.get_val(Settings.PROBLEM_TYPE))
    # print('settings:equations_of_motion:', prob.aviary_inputs.get_val(Settings.EQUATIONS_OF_MOTION))
    # print('settings:mass_method:', prob.aviary_inputs.get_val(Settings.MASS_METHOD))

    print('mission:range (km): ', prob.get_val('mission:range', units='km'))
    print('time_duration (s)', prob.get_val('traj.cruise.t_duration', units='s'))
    print('mach', prob.get_val('traj.cruise.timeseries.mach'))
    print('Aircraft.Battery.MASS (kg)', prob.get_val(Aircraft.Battery.MASS, units='kg'))
    print(
        'Aircraft.Engine.Motor.IDLE_CURRENT (A)',
        prob.get_val(Aircraft.Engine.Motor.IDLE_CURRENT, units='A'),
    )
    print('Aircraft.Engine.Motor.MASS (kg)', prob.get_val(Aircraft.Engine.Motor.MASS, units='kg'))

    return prob


if __name__ == '__main__':
    CruiseExample()
