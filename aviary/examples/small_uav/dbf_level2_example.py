"""
Define an aviary mission with an NPSS defined engine. During pre-mission the engine is designed and
an engine deck is made. During the mission the deck is used for performance. Weight is estimated
using the default Aviary method. The engine model was developed using NPSS v3.2.
"""
from copy import deepcopy
import numpy as np
import aviary.api as av
from aviary.examples.small_uav.dbf_example_phase import phase_info
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder

rc_prop = RCBuilder()

phase_info = deepcopy(phase_info)

phase_info.pop('climb')
# phase_info.pop('cruise')
phase_info.pop('descent')

phase_info['pre_mission']['external_subsystems'] = [DBFMassBuilder()]
# phase_info['climb']['external_subsystems'] = [CustomAeroBuilder()]

# phase_info['climb']['subsystem_options']['core_aerodynamics'] = {
#     'method': 'external',
# }
phase_info['cruise']['external_subsystems'] = [CustomAeroBuilder()]

phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {
    'method': 'external',
}
# phase_info['cruise']['subsystem_options']['core_mass'] = {
#     'method': 'external',
# }
prob = av.AviaryProblem(verbosity=1)

prob.options['group_by_pre_opt_post'] = True

# Load aircraft and options data from user
# Allow for user overrides here
# add engine builder
prob.load_inputs(
    'models/aircraft/test_aircraft/small_scale_uav.csv',
    phase_info,
    engine_builders=[rc_prop],
)

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()
prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver('IPOPT')
# prob.add_driver('SLSQP')
# prob.driver.options["debug_print"] = ["desvars", "nl_cons", "objs"]

prob.add_design_variables()

# prob.model.add_design_var(av.Mission.Design.GROSS_MASS, lower=0.5, upper=40, units='kg', scaler=1)
# prob.model.add_design_var(av.Mission.Summary.GROSS_MASS, lower=0.5, upper=40, units='kg', scaler=1)
# prob.model.add_design_var('traj.cruise.timeseries.input_values:throttle', lower=0, upper=1)
prob.add_objective('time')

prob.setup()

prob.set_solver_print(level=0)

# prob.model.set_val(av.Mission.Design.GROSS_MASS, 6, units='kg')
# prob.model.set_val('traj.cruise.timeseries.input_values:throttle', 1.0, units='unitless')
prob.set_initial_guesses()

prob.run_aviary_problem(suppress_solver_print= False)
with open("aviary\examples\small_uav\level2_newvars.txt", "w") as f:
        prob.model.list_vars(print_arrays=True,out_stream=f, units=True)
# try:
#     prob.run_aviary_problem(suppress_solver_print= False)
# except: 
#     with open("aviary\examples\small_uav\level2_vars.txt", "w") as f:
#         prob.model.list_vars(print_arrays=True,out_stream=f, units=True)
#         # f.write(str(names))