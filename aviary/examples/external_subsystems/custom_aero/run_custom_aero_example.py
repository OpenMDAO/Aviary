"""Run the a mission with a simple external component that computes aircraft lift and drag."""

from copy import deepcopy

import aviary.api as av
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder

phase_info = deepcopy(av.default_height_energy_phase_info)

# Just do cruise in this example.
phase_info.pop('climb')
phase_info.pop('descent')

# Add custom aero.
# TODO: This API for replacing aero will be changed an upcoming release.
phase_info['cruise']['external_subsystems'] = [CustomAeroBuilder()]

# Disable internal aero
# TODO: This API for replacing aero will be changed an upcoming release.
phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {
    'method': 'external',
}


if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    # Note, SLSQP has trouble here.
    prob.add_driver('IPOPT')

    prob.add_design_variables()

    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()

    prob.run_aviary_problem(suppress_solver_print=True)

    print('done')
