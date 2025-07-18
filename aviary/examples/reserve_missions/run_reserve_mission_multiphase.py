"""
This is a slightly more complex Aviary example of sizing an aircraft whose design mission includes
a reserve segment. It is the same basic problem as the `level2_example.py` script, but with the
addition  of a reserve mission.

The pre-existing phase_info data is imported, and a reserve mission with a climb, 2 cruise, and a
descent segment are added. The first cruise has a fixed range, while the second cruise has a fixed
time (i.e. a loiter) to demonstrate how different reserve missions can be created.

We then call the correct methods in order to set up and run an Aviary problem using the level 2
interface.
"""

import aviary.api as av
from aviary.examples.example_phase_info import phase_info

#####################
# Define Phase Info #
#####################

# Add reserve phases to existing phase_info
phase_info.update(
    {
        'reserve_climb': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'reserve': True,
                'num_segments': 5,
                'order': 3,
                'mach_optimize': False,
                'mach_polynomial_order': 1,
                'mach_initial': (0.36, 'unitless'),
                'mach_final': (0.72, 'unitless'),
                'altitude_optimize': False,
                'altitude_polynomial_order': 1,
                'altitude_initial': (0.0, 'ft'),
                'altitude_final': (32000.0, 'ft'),
                'throttle_enforcement': 'path_constraint',
                'time_initial_bounds': ((0.0, 0.0), 'min'),
                'time_duration_bounds': ((30.0, 192.0), 'min'),
            },
            'initial_guesses': {
                'time': ([0, 128], 'min'),
            },
        },
        'reserve_cruise_fixed_range': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'reserve': True,
                # Distance traveled in this phase
                'target_distance': (300, 'km'),
                'num_segments': 5,
                'order': 3,
                'mach_optimize': False,
                'mach_polynomial_order': 1,
                'mach_initial': (0.72, 'unitless'),
                'mach_final': (0.72, 'unitless'),
                'altitude_optimize': False,
                'altitude_polynomial_order': 1,
                'altitude_initial': (32000.0, 'ft'),
                'altitude_final': (32000.0, 'ft'),
                'throttle_enforcement': 'boundary_constraint',
                'time_initial_bounds': ((149.5, 448.5), 'min'),
                'time_duration_bounds': ((0, 300), 'min'),
            },
            'initial_guesses': {
                'time': ([30, 120], 'min'),
            },
        },
        'reserve_cruise_fixed_time': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'reserve': True,
                # Time length of this phase
                'time_duration': (30, 'min'),
                'num_segments': 5,
                'order': 3,
                'distance_solve_segments': False,
                'mach_optimize': False,
                'mach_polynomial_order': 1,
                'mach_initial': (0.72, 'unitless'),
                'mach_final': (0.72, 'unitless'),
                'altitude_optimize': False,
                'altitude_polynomial_order': 1,
                'altitude_initial': (32000.0, 'ft'),
                'altitude_final': (32000.0, 'ft'),
                'throttle_enforcement': 'boundary_constraint',
                'time_initial_bounds': ((60, 448.5), 'min'),
            },
        },
        'reserve_descent': {
            'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
            'user_options': {
                'reserve': True,
                'num_segments': 5,
                'order': 3,
                'mach_optimize': False,
                'mach_polynomial_order': 1,
                'mach_initial': (0.72, 'unitless'),
                'mach_final': (0.36, 'unitless'),
                'altitude_optimize': False,
                'altitude_polynomial_order': 1,
                'altitude_initial': (32000.0, 'ft'),
                'altitude_final': (500.0, 'ft'),
                'throttle_enforcement': 'path_constraint',
                'time_initial_bounds': ((120.5, 361.5), 'min'),
                'time_duration_bounds': ((29.0, 87.0), 'min'),
            },
            'initial_guesses': {
                'time': ([60, 500], 'min'),
            },
        },
    }
)

######################
# Run Aircraft Model #
######################

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

prob.add_driver('SLSQP')

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
