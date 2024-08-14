# Import the necessary things
import aviary.api as av
from aviary.variable_info.enums import AlphaModes, AnalysisScheme, EquationsOfMotion, GASPEngineType, FlapType, LegacyCode, ProblemType, SpeedType, ThrottleAllocation, Verbosity
from aviary.variable_info.variables import Mission, Aircraft, Settings
from aviary.interface.default_phase_info.two_dof import phase_info
from aviary.interface.save_sizing import save_sizing_json, load_off_design

# Initialize an aviary problem for sizing mission
prob = av.AviaryProblem()

# Load inputs
prob.load_inputs('models/test_aircraft/aircraft_for_bench_GwGm.csv', phase_info)

# Run problem setup
prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()
prob.add_driver("IPOPT", max_iter=100)
prob.add_design_variables()
prob.add_objective()
prob.setup()
prob.set_initial_guesses()

# Execute the problem
prob.run_aviary_problem()

# Print the results to cmd line
print("Results from sizing mission")
print("Sizing Mission.Summary.RANGE                          ",
      prob.get_val(Mission.Summary.RANGE, units='nmi'))
print("Sizing Aircraft.Design.OPERATING_MASS                 ",
      prob.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'))
print("Sizing Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS    ",
      prob.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm'))
print("Sizing Mission.Summary.TOTAL_FUEL_MASS                ",
      prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'))
print("Sizing Mission.Design.GROSS_MASS                      ",
      prob.get_val(Mission.Design.GROSS_MASS, units='lbm'))
print("Sizing Mission.Summary.GROSS_MASS                     ",
      prob.get_val(Mission.Summary.GROSS_MASS, units='lbm'))
print("Operating + Payload + Fuel =                          ", prob.get_val(Aircraft.Design.OPERATING_MASS, units='lbm') +
      prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm')+prob.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm'))
print()
print("Sizing Landing Ground Distance                        ",
      prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'))
print("Sizing Mission.Landing.TOUCHDOWN_MASS                 ",
      prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'))

################################################################################
# Save the aviary inputs to a json file
save_sizing_json(prob, 'test_json_from_function.json')

prob_json_alt = load_off_design(
    'test_json_from_function.json', ProblemType.ALTERNATE, phase_info, 30000, 3000, 150000)

# run problem setup
prob_json_alt.check_and_preprocess_inputs()
prob_json_alt.add_pre_mission_systems()
prob_json_alt.add_phases()
prob_json_alt.add_post_mission_systems()
prob_json_alt.link_phases()
prob_json_alt.add_driver("IPOPT", max_iter=100)
prob_json_alt.add_design_variables()
prob_json_alt.add_objective()
prob_json_alt.setup()
prob_json_alt.set_initial_guesses()

# Execute the problem
prob_json_alt.run_aviary_problem()

print("Alternate Results from json")
print("Alternate Mission.Summary.RANGE                         ",
      prob_json_alt.get_val(Mission.Summary.RANGE, units='nmi'))
print("Alternate Aircraft.Design.OPERATING_MASS                ",
      prob_json_alt.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'))
print("Alternate Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS   ",
      prob_json_alt.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm'))
print("Alternate Mission.Summary.TOTAL_FUEL_MASS               ",
      prob_json_alt.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'))
print("Alternate Mission.Design.GROSS_MASS                     ",
      prob_json_alt.get_val(Mission.Design.GROSS_MASS, units='lbm'))
print("Alternate Mission.Summary.GROSS_MASS                    ",
      prob_json_alt.get_val(Mission.Summary.GROSS_MASS, units='lbm'))
print("Operating + Payload + Fuel =                            ", prob_json_alt.get_val(Aircraft.Design.OPERATING_MASS, units='lbm') +
      prob_json_alt.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm')+prob_json_alt.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm'))
print()
print("Alternate Mission.Objectives.FUEL                       ",
      prob_json_alt.get_val(Mission.Objectives.FUEL, units='unitless'))
print("Alternate Landing Ground Distance                       ",
      prob_json_alt.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'))
print("Alternate Mission.Landing.TOUCHDOWN_MASS                ",
      prob_json_alt.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'))


prob_json_fall = load_off_design(
    'test_json_from_function.json', ProblemType.FALLOUT, phase_info, 30000, 1000, 150000)

# run problem setup
prob_json_fall.check_and_preprocess_inputs()
prob_json_fall.add_pre_mission_systems()
prob_json_fall.add_phases()
prob_json_fall.add_post_mission_systems()
prob_json_fall.link_phases()
prob_json_fall.add_driver("IPOPT", max_iter=100)
prob_json_fall.add_design_variables()
prob_json_fall.add_objective()
prob_json_fall.setup()
prob_json_fall.set_initial_guesses()

# Execute the problem
prob_json_fall.run_aviary_problem()

print("Fallout Results from json")
print("Fallout Mission.Summary.RANGE                         ",
      prob_json_fall.get_val(Mission.Summary.RANGE, units='nmi'))
print("Fallout Aircraft.Design.OPERATING_MASS                ",
      prob_json_fall.get_val(Aircraft.Design.OPERATING_MASS, units='lbm'))
print("Fallout Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS   ",
      prob_json_fall.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm'))
print("Fallout Mission.Summary.TOTAL_FUEL_MASS               ",
      prob_json_fall.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'))
print("Fallout Mission.Design.GROSS_MASS                     ",
      prob_json_fall.get_val(Mission.Design.GROSS_MASS, units='lbm'))
print("Fallout Mission.Summary.GROSS_MASS                    ",
      prob_json_fall.get_val(Mission.Summary.GROSS_MASS, units='lbm'))
print("Operating + Payload + Fuel =                          ", prob_json_fall.get_val(Aircraft.Design.OPERATING_MASS, units='lbm') +
      prob_json_fall.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm')+prob_json_fall.get_val(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm'))
print()
print("Fallout Mission.Objectives.FUEL                       ",
      prob_json_fall.get_val(Mission.Objectives.FUEL, units='unitless'))
print("Fallout Landing Ground Distance                       ",
      prob_json_fall.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'))
print("Fallout Mission.Landing.TOUCHDOWN_MASS                ",
      prob_json_fall.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'))
