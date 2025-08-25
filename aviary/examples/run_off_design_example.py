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

phase_info = av.default_height_energy_phase_info

##################
# Sizing Mission #
##################
prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)
prob.check_and_preprocess_inputs()

prob.build_model()

prob.add_driver('SLSQP', max_iter=50)
prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()
prob.setup()
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
