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
reserve_climb1 = deepcopy(phase_info['climb1'])
reserve_climb1['user_options']['reserve'] = True
reserve_climb1['user_options']['distance_upper'] = (5000, 'NM')
reserve_climb1['initial_guesses']['distance'] = ([3675, 3700], 'nmi')

reserve_climb2 = deepcopy(phase_info['climb2'])
reserve_climb2['user_options']['reserve'] = True
reserve_climb2['user_options']['altitude_final'] = (25e3, 'ft')
reserve_climb2['user_options']['distance_upper'] = (5000, 'NM')
reserve_climb2['initial_guesses']['altitude'] = ([10e3, 25e3], 'ft')
reserve_climb2['initial_guesses']['distance'] = ([3700, 3725], 'nmi')

distance_cruise1 = deepcopy(phase_info['cruise'])
distance_cruise1['user_options']['reserve'] = True
distance_cruise1['user_options']['alt_cruise'] = (25e3, 'ft')
distance_cruise1['user_options']['target_distance'] = (100, 'nmi')
distance_cruise1['initial_guesses']['altitude'] = (25e3, 'ft')
distance_cruise1['initial_guesses']['initial_distance'] = (3725, 'nmi')

duration_cruise1 = deepcopy(phase_info['cruise'])
duration_cruise1['user_options']['reserve'] = True
duration_cruise1['user_options']['alt_cruise'] = (25e3, 'ft')
duration_cruise1['user_options']['time_duration'] = (30, 'min')
duration_cruise1['user_options']['time_initial_bounds'] = ((149.5, 448.5), 'min')
duration_cruise1['initial_guesses']['altitude'] = (25e3, 'ft')
duration_cruise1['initial_guesses']['initial_distance'] = (3825, 'nmi')

distance_cruise2 = deepcopy(phase_info['cruise'])
distance_cruise2['user_options']['reserve'] = True
distance_cruise2['user_options']['alt_cruise'] = (25e3, 'ft')
distance_cruise2['user_options']['target_distance'] = (75, 'nmi')
distance_cruise2['initial_guesses']['altitude'] = (25e3, 'ft')
distance_cruise2['initial_guesses']['initial_distance'] = (3900, 'nmi')

reserve_descent1 = deepcopy(phase_info['desc1'])
reserve_descent1['user_options']['reserve'] = True
reserve_descent1['initial_guesses']['altitude'] = ([25e3, 10e3], 'ft')
reserve_descent1['initial_guesses']['distance'] = ([3900, 3925], 'nmi')

reserve_descent2 = deepcopy(phase_info['desc2'])
reserve_descent2['user_options']['reserve'] = True
reserve_descent2['initial_guesses']['distance'] = ([3925, 3950], 'nmi')

phase_info.update(
    {
        'reserve_climb1': reserve_climb1,
        'reserve_climb2': reserve_climb2,
        'reserve_cruise_fixed_range': distance_cruise1,
        'reserve_cruise_fixed_time': duration_cruise1,
        'reserve_cruise_fixed_range_2': distance_cruise2,
        'reserve_desc1': reserve_descent1,
        'reserve_desc2': reserve_descent2,
    }
)

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

    prob.add_driver('SNOPT', max_iter=50)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()

    prob.run_aviary_problem(record_filename='2dof_reserve_mission_multiphase.db')
