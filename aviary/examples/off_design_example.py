"""
This is a slightly more complex Aviary example of running a coupled aircraft design-mission optimization.
It runs the same mission as the `run_basic_aviary_example.py` script, but it uses the AviaryProblem class to set up the problem.
This exposes more options and flexibility to the user and uses the "Level 2" API within Aviary.

We define a `phase_info` object, which tells Aviary how to model the mission.
Here we have climb, cruise, and descent phases.
We then call the correct methods in order to set up and run an Aviary optimization problem.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""
import aviary.api as av
from example_phase_info import phase_info

from aviary.interface.default_phase_info.height_energy import phase_info_parameterization
from aviary.variable_info.enums import SpeedType, ProblemType, Verbosity
from aviary.variable_info.variables import Mission, Aircraft, Settings
from aviary.utils.process_input_decks import initial_guessing, parse_inputs

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

prob.add_driver('SNOPT', max_iter=100, verbosity=Verbosity.DEBUG)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
prob_fallout = av.AviaryProblem()

# Load inputs from .csv file
prob_fallout.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

# print the problem type
print(prob_fallout.problem_type)

# now change the problem type:
print("Changing problem type to fallout - check:")
prob_fallout.problem_type = ProblemType.FALLOUT
prob_fallout.aviary_inputs.set_val('problem_type', ProblemType.FALLOUT, units='unitless')

print(prob_fallout.problem_type)

mission_mass = prob.get_val(Mission.Summary.GROSS_MASS, units='lbm')
# mission_mass = 162007.6236365
prob_fallout.aviary_inputs.set_val('mission:design:gross_mass', mission_mass, units='lbm')
prob_fallout.aviary_inputs.set_val('mission:summary:gross_mass', mission_mass, units='lbm')

prob_fallout.check_and_preprocess_inputs()
prob_fallout.add_pre_mission_systems()
prob_fallout.add_phases(phase_info_parameterization=phase_info_parameterization)
prob_fallout.add_post_mission_systems()
prob_fallout.link_phases()
prob_fallout.add_driver('SNOPT', max_iter=100, verbosity=Verbosity.DEBUG)
prob_fallout.add_design_variables()
prob_fallout.add_objective()
prob_fallout.setup()
prob_fallout.set_initial_guesses()
prob_fallout.run_aviary_problem()

prob_alternate = av.AviaryProblem()

# Load inputs from .csv file
prob_alternate.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

# print the problem type
print(prob_alternate.problem_type)

# now change the problem type:
print("Changing problem type to sizing - check:")
prob_alternate.problem_type = ProblemType.ALTERNATE
prob_alternate.aviary_inputs.set_val('problem_type', ProblemType.ALTERNATE, units='unitless')

print(prob_alternate.problem_type)

mission_mass = prob.get_val(Mission.Summary.GROSS_MASS, units='lbm')
# mission_mass = 162007.6236365
prob_alternate.aviary_inputs.set_val('mission:design:gross_mass', mission_mass, units='lbm')
prob_alternate.aviary_inputs.set_val('mission:summary:gross_mass', mission_mass, units='lbm')

prob_alternate.check_and_preprocess_inputs()
prob_alternate.add_pre_mission_systems()
prob_alternate.add_phases(phase_info_parameterization=phase_info_parameterization)
prob_alternate.add_post_mission_systems()
prob_alternate.link_phases()
prob_alternate.add_driver('SNOPT', max_iter=100, verbosity=Verbosity.DEBUG)
prob_alternate.add_design_variables()
prob_alternate.add_objective()
prob_alternate.setup()
prob_alternate.set_initial_guesses()
prob_alternate.run_aviary_problem()

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


# print('Fallout Mission Vars')
# prob_fallout.model.list_vars(includes='*mass')
print('Alternate Mission Vars')
prob_alternate.model.list_vars(includes='*mass')
