from aviary.interface.methods_for_level2 import AviaryProblem

from aviary.examples.external_subsystems.battery.battery_builder import BatteryBuilder
from aviary.examples.external_subsystems.battery.battery_variable_meta_data import ExtendedMetaData
from aviary.api import default_height_energy_phase_info as phase_info
from aviary.utils.functions import get_aviary_resource_path


battery_builder = BatteryBuilder(include_constraints=False)

phase_info['pre_mission']['external_subsystems'] = [battery_builder]
phase_info['climb']['external_subsystems'] = [battery_builder]
phase_info['cruise']['external_subsystems'] = [battery_builder]
phase_info['descent']['external_subsystems'] = [battery_builder]

prob = AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
input_file = get_aviary_resource_path('models/test_aircraft/aircraft_for_bench_FwFm.csv')
prob.load_inputs(input_file, phase_info, meta_data=ExtendedMetaData)

# Preprocess inputs
prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver("SLSQP")

prob.add_design_variables()

prob.add_objective('mass')
# prob.model.add_objective(
#     f'traj.climb.states:{Mission.Battery.STATE_OF_CHARGE}', index=-1, ref=-1)

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
