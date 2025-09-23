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

# We will size the aircraft in this example for a longer design range than specified in the default
# phase_info
phase_info = av.default_height_energy_phase_info
phase_info['post_mission']['target_range'] = (2500.0, 'nmi')

##################
# Sizing Mission #
##################
prob = av.AviaryProblem(verbosity=0)

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs(
    'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv', phase_info
)

# Preprocess inputs
prob.check_and_preprocess_inputs()

prob.build_model()

prob.add_driver('SLSQP', max_iter=50)
prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()
prob.setup()
print('Running Design Mission')
prob.run_aviary_problem()

# Fallout Mission
print('Running fixed-mass, varying range off-design problem')
prob_fallout = prob.run_off_design_mission(problem_type='fallout', mission_gross_mass=115000)

# Alternate Mission
print('Running fixed-range, varying fuel off-design problem')
prob_alternate = prob.run_off_design_mission(problem_type='alternate', mission_range=1250)

print('\n--------------')
print('Sizing Results')
print('--------------')
print(f'Design Range = {prob.get_val(av.Mission.Design.RANGE)[0]} nmi')
print(f'Mission Range = {prob.get_val(av.Mission.Summary.RANGE)[0]} nmi')
print(f'Fuel Mass = {prob.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)[0]} lbm')
print(f'Operating Empty Mass = {prob.get_val(av.Mission.Summary.OPERATING_MASS)[0]} lbm')
print(f'Payload Mass = {prob.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0]} lbm')
print(f'Design Gross Mass = {prob.get_val(av.Mission.Design.GROSS_MASS)[0]} lbm')
print(f'Mission Gross Mass = {prob.get_val(av.Mission.Summary.GROSS_MASS)[0]} lbm')

print('\n---------------')
print('Fallout Results')
print('---------------')
print(f'Design Range = {prob_fallout.get_val(av.Mission.Design.RANGE)[0]} nmi')
print(f'Mission Range = {prob_fallout.get_val(av.Mission.Summary.RANGE)[0]} nmi')
print(f'Fuel Mass = {prob_fallout.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)[0]} lbm')
print(f'Operating Empty Mass = {prob_fallout.get_val(av.Mission.Summary.OPERATING_MASS)[0]} lbm')
print(f'Payload Mass = {prob_fallout.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0]} lbm')
print(f'Design Gross Mass = {prob_fallout.get_val(av.Mission.Design.GROSS_MASS)[0]} lbm')
print(f'Mission Gross Mass = {prob_fallout.get_val(av.Mission.Summary.GROSS_MASS)[0]} lbm')

print('\n-----------------')
print('Alternate Results')
print('-----------------')
print(f'Design Range = {prob_alternate.get_val(av.Mission.Design.RANGE)[0]} nmi')
print(f'Mission Range = {prob_alternate.get_val(av.Mission.Summary.RANGE)[0]} nmi')
print(f'Fuel Mass = {prob_alternate.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)[0]} lbm')
print(f'Operating Empty Mass = {prob_alternate.get_val(av.Mission.Summary.OPERATING_MASS)[0]} lbm')
print(f'Payload Mass = {prob_alternate.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)[0]} lbm')
print(f'Design Gross Mass = {prob_alternate.get_val(av.Mission.Design.GROSS_MASS)[0]} lbm')
print(f'Mission Gross Mass = {prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)[0]} lbm')
