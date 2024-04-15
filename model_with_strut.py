from copy import deepcopy

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.variable_info.enums import Verbosity
from aviary.interface.default_phase_info.two_dof import phase_info as two_dof_phase_info


prob = AviaryProblem()

local_phase_info = deepcopy(two_dof_phase_info)
prob.load_inputs('aviary/models/test_aircraft/converter_configuration_test_data_GwGm.csv', local_phase_info)

prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()
prob.add_driver("SLSQP", max_iter=0, verbosity=Verbosity.QUIET)
prob.add_design_variables()
prob.add_objective()
prob.setup()
prob.set_initial_guesses()

prob.recording_options['record_inputs'] = True
prob.recording_options['record_outputs'] = True

prob.run_aviary_problem("dymos_solution.db", make_plots=False)

