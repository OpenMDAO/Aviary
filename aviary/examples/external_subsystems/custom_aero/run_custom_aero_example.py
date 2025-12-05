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

# Start cruise at t=0.
del phase_info['cruise']['user_options']['time_initial_bounds']
phase_info['cruise']['user_options']['time_initial'] = (0.0, 'min')


if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(
        'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv', phase_info
    )

    prob.check_and_preprocess_inputs()

    prob.build_model()

    # Note, SLSQP has trouble here.
    prob.add_driver('IPOPT')

    prob.add_design_variables()

    prob.add_objective()

    prob.setup()

    prob.run_aviary_problem(suppress_solver_print=True)

    print('done')
