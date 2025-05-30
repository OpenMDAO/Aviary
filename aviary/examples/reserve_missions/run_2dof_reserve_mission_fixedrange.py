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
from aviary.interface.default_phase_info.two_dof import phase_info

phase_info = deepcopy(phase_info)

# Add reserve phase(s)
reserve_cruise = deepcopy(phase_info['cruise'])
reserve_cruise['user_options']['reserve'] = True
reserve_cruise['user_options']['target_distance'] = (200, 'km')
reserve_cruise['initial_guesses']['initial_distance'] = (3700, 'nmi')

phase_info.update({'reserve_cruise': reserve_cruise})

if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('models/test_aircraft/aircraft_for_bench_GwGm.csv', phase_info)

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver('SNOPT', max_iter=50, verbosity=2)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()

    prob.run_aviary_problem(record_filename='2dof_reserve_mission_fixedrange.db')
