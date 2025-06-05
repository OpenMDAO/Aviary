"""
This is a slightly more complex Aviary example of running a coupled aircraft design-mission optimization.
It runs the same mission as the `level1_example.py` script, but it uses the AviaryProblem class to set up the problem.
This exposes more options and flexibility to the user and uses the "Level 2" API within Aviary.

We define a `phase_info` object, which tells Aviary how to model the mission.
Here we have climb, cruise, and descent phases.
We then call the correct methods in order to set up and run an Aviary optimization problem.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""

from copy import deepcopy

import aviary.api as av
from aviary.examples.example_phase_info import phase_info

phase_info = deepcopy(phase_info)

# Copy the current cruise phase, then make it a reserve phase
reserve_phase_0 = deepcopy(phase_info['cruise'])
reserve_phase_0['user_options']['reserve'] = True
reserve_phase_0['user_options']['target_distance'] = (200, 'km')
# remove the climb from the original cruise
reserve_phase_0['user_options']['altitude_final'] = (32000.0, 'ft')
# This cruise is much shorter so we need to revise the duration_bounds for this phase
reserve_phase_0['user_options']['time_duration_bounds'] = ((0, 120.0), 'min')

# Add the reserve phase to phase_info
phase_info.update({'reserve_cruise': reserve_phase_0})

if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver('SLSQP', max_iter=100)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()

    prob.run_aviary_problem(record_filename='reserve_mission_fixedrange.db')
