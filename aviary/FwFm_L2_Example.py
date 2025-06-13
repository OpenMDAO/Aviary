import aviary.api as av
from aviary.interface.default_phase_info.height_energy import phase_info

csv_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'

prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs(csv_path, phase_info)

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver('IPOPT', max_iter=50)

prob.add_design_variables()

prob.add_objective()

prob.setup()

prob.set_initial_guesses()

# prob.run_model()
# prob.model.list_vars(units=True, print_arrays=True)
prob.run_aviary_problem()

# prob.model.list_vars()
prob.list_driver_vars()
