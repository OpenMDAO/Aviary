from aviary.interface.methods_for_level2 import AviaryProblem
import os
import pkg_resources

from aviary.examples.external_subsystems.battery.battery_builder import BatteryBuilder
from aviary.examples.external_subsystems.battery.battery_variables import Aircraft
from aviary.api import default_height_energy_phase_info as phase_info


battery_builder = BatteryBuilder(include_constraints=False)

phase_info['pre_mission']['external_subsystems'] = [battery_builder]
phase_info['climb']['external_subsystems'] = [battery_builder]
phase_info['cruise']['external_subsystems'] = [battery_builder]
phase_info['descent']['external_subsystems'] = [battery_builder]

up_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
input_file = pkg_resources.resource_filename(
    "aviary", "models/test_aircraft/aircraft_for_bench_FwFm.csv")

prob = AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs(input_file, phase_info)


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
