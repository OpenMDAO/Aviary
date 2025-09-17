"""Example mission using the a detailed battery model."""

from aviary.api import default_height_energy_phase_info as phase_info
from aviary.examples.external_subsystems.battery.battery_builder import BatteryBuilder
from aviary.examples.external_subsystems.battery.battery_variable_meta_data import ExtendedMetaData
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.utils.functions import get_aviary_resource_path

battery_builder = BatteryBuilder(include_constraints=False)

prob = AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
input_file = get_aviary_resource_path('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv')
prob.load_inputs(input_file, phase_info, meta_data=ExtendedMetaData)

prob.load_external_subsystems([battery_builder])

prob.check_and_preprocess_inputs()

prob.build_model()

prob.add_driver('SLSQP')

prob.add_design_variables()

prob.add_objective('mass')
# prob.model.add_objective(
#     f'traj.climb.states:{Dynamic.Battery.STATE_OF_CHARGE}', index=-1, ref=-1)

prob.setup()

prob.run_aviary_problem()
