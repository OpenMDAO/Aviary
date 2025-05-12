from copy import deepcopy

import aviary.api as av
from aviary.examples.external_subsystems.battery.battery_builder import BatteryBuilder
from aviary.examples.external_subsystems.battery.battery_variable_meta_data import ExtendedMetaData
from aviary.examples.external_subsystems.battery.battery_variables import Dynamic

battery_builder = BatteryBuilder()

phase_info = deepcopy(av.default_height_energy_phase_info)
# Here we just add the simple weight system to only the pre-mission
phase_info['pre_mission']['external_subsystems'] = [battery_builder]
phase_info['climb']['external_subsystems'] = [battery_builder]


if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(
        'models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info, meta_data=ExtendedMetaData
    )

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver('SLSQP')

    prob.add_design_variables()

    prob.model.add_objective(
        f'traj.climb.states:{Dynamic.Battery.STATE_OF_CHARGE}', index=-1, ref=-1
    )

    prob.setup()

    prob.set_initial_guesses()

    prob.run_aviary_problem(suppress_solver_print=True)
