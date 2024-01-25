from copy import deepcopy

from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.multi_engine_single_aisle.multi_engine_single_aisle_data import inputs

# Build problem
local_phase_info = deepcopy(phase_info)
prob = AviaryProblem()

prob.load_inputs(inputs, local_phase_info)

# Have checks for clashing user inputs
# Raise warnings or errors depending on how clashing the issues are
prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver("SNOPT", max_iter=50, use_coloring=True)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem("dymos_solution.db")
