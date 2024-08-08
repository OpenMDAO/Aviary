import aviary.api as av
from easy_phase_info_max import phase_info


prob = av.AviaryProblem()
prob.load_inputs("c5_models/c5_maxpayload.csv", phase_info)
prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()
prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()

prob.setup()

prob.set_initial_guesses()
# prob.final_setup()
prob.run_aviary_problem()
