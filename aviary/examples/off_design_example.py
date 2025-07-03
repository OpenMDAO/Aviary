"""
This is a slightly more complex Aviary example of running a coupled aircraft design-mission optimization.
It runs the same mission as the `level1_example.py` script, but it uses the AviaryProblem class to set up the problem.
This exposes more options and flexibility to the user and uses the "Level 2" API within Aviary.

We define a `phase_info` object, which tells Aviary how to model the mission.
Here we have climb, cruise, and descent phases.
We then call the correct methods in order to set up and run an Aviary optimization problem.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""

import aviary.api as av
from aviary.interface.default_phase_info.height_energy import phase_info_parameterization

phase_info = {
    'pre_mission': {'include_takeoff': True, 'optimize_mass': True},
    'climb': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 6,
            'order': 3,
            'mach_optimize': True,
            'mach_bounds': ((0.1, 0.8), 'unitless'),
            'altitude_optimize': True,
            'altitude_bounds': ((0.0, 35000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'time_initial_bounds': ((0.0, 2.0), 'min'),
            'time_duration_bounds': ((5.0, 50.0), 'min'),
            'no_descent': False,
        },
        'initial_guesses': {
            'time': ([0, 40.0], 'min'),
            'altitude': ([35, 35000.0], 'ft'),
            'mach': ([0.3, 0.79], 'unitless'),
        },
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 1,
            'order': 3,
            'mach_optimize': True,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.79, 0.79), 'unitless'),
            'altitude_optimize': True,
            'altitude_polynomial_order': 1,
            'altitude_bounds': ((35000.0, 35000.0), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'time_initial_bounds': ((64.0, 192.0), 'min'),
            'time_duration_bounds': ((60.0, 720.0), 'min'),
        },
        'initial_guesses': {
            'time': ([64, 113], 'min'),
            'altitude': ([35000, 35000.0], 'ft'),
            'mach': ([0.79, 0.79], 'unitless'),
        },
    },
    'descent': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'mach_optimize': True,
            'mach_final': (0.3, 'unitless'),
            'mach_bounds': ((0.2, 0.8), 'unitless'),
            'altitude_optimize': True,
            'altitude_final': (35.0, 'ft'),
            'altitude_bounds': ((0.0, 35000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'time_initial_bounds': ((120.0, 800.0), 'min'),
            'time_duration_bounds': ((5.0, 35.0), 'min'),
            'no_climb': True,
        },
        'initial_guesses': {
            'time': ([241, 30], 'min'),
            'altitude': ([35000, 35.0], 'ft'),
            'mach': ([0.79, 0.3], 'unitless'),
        },
    },
    'post_mission': {
        'include_landing': True,
        'constrain_range': True,
        'target_range': (3375.0, 'nmi'),
    },
}

##################
# Sizing Mission #
##################
prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

# Preprocess inputs
prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases(phase_info_parameterization=phase_info_parameterization)
prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()
prob.add_driver('SLSQP', max_iter=50)
prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()
prob.setup()
prob.set_initial_guesses()
prob.run_aviary_problem()
prob.save_sizing_to_json()

# Fallout Mission
prob_fallout = prob.fallout_mission()

# Alternate Mission
prob_alternate = prob.alternate_mission()

print('--------------')
print('Sizing Results')
print('--------------')
print(f'Design Range = {prob.get_val(av.Mission.Design.RANGE)}')
print(f'Summary Range = {prob.get_val(av.Mission.Summary.RANGE)}')
print(f'Fuel mass = {prob.get_val(av.Mission.Design.FUEL_MASS)}')
print(f'Total fuel mass = {prob.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)}')
print(f'Empty mass = {prob.get_val(av.Aircraft.Design.OPERATING_MASS)}')
print(f'Payload mass = {prob.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)}')
print(f'Design Gross mass = {prob.get_val(av.Mission.Design.GROSS_MASS)}')
print(f'Summary Gross mass = {prob.get_val(av.Mission.Summary.GROSS_MASS)}')

print('---------------')
print('Fallout Results')
print('---------------')
print(f'Design Range = {prob_fallout.get_val(av.Mission.Design.RANGE)}')
print(f'Summary Range = {prob_fallout.get_val(av.Mission.Summary.RANGE)}')
print(f'Fuel mass = {prob_fallout.get_val(av.Mission.Design.FUEL_MASS)}')
print(f'Total fuel mass = {prob_fallout.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)}')
print(f'Empty mass = {prob_fallout.get_val(av.Aircraft.Design.OPERATING_MASS)}')
print(f'Payload mass = {prob_fallout.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)}')
print(f'Design Gross mass = {prob_fallout.get_val(av.Mission.Design.GROSS_MASS)}')
print(f'Summary Gross mass = {prob_fallout.get_val(av.Mission.Summary.GROSS_MASS)}')

print('---------------')
print('Alternate Results')
print('---------------')
print(f'Design Range = {prob_alternate.get_val(av.Mission.Design.RANGE)}')
print(f'Summary Range = {prob_alternate.get_val(av.Mission.Summary.RANGE)}')
print(f'Fuel mass = {prob_alternate.get_val(av.Mission.Design.FUEL_MASS)}')
print(f'Total fuel mass = {prob_alternate.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)}')
print(f'Empty mass = {prob_alternate.get_val(av.Aircraft.Design.OPERATING_MASS)}')
print(f'Payload mass = {prob_alternate.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)}')
print(f'Design Gross mass = {prob_alternate.get_val(av.Mission.Design.GROSS_MASS)}')
print(f'Summary Gross mass = {prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)}')
