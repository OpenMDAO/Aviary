"""
This is a slightly more complex Aviary example of running a coupled aircraft design-mission optimization.
It runs the same mission as the `run_basic_aviary_example.py` script, but it uses the AviaryProblem class to set up the problem.
This exposes more options and flexibility to the user and uses the "Level 2" API within Aviary.

We define a `phase_info` object, which tells Aviary how to model the mission.
Here we have climb, cruise, and descent phases.
We then call the correct methods in order to set up and run an Aviary optimization problem.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""
import aviary.api as av
from aviary.interface.default_phase_info.two_dof import phase_info
from copy import deepcopy

phase_info = deepcopy(phase_info)

# Add reserve phase(s)
phase_info.update({
    'reserve_cruise': {
        'user_options': {
            'reserve': True,
            "target_distance": (200, 'km'),
            'alt_cruise': (37.5e3, 'ft'),
            'mach_cruise': 0.8,
        },
        'initial_guesses': {
            # [Initial mass, delta mass] for special cruise phase.
            'mass': ([171481., -35000], 'lbm'),
            'initial_distance': (200.e3, 'ft'),
            'initial_time': (1516., 's'),
            'altitude': (37.5e3, 'ft'),
            'mach': (0.8, 'unitless'),
        }
    }})

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

prob.add_driver("SNOPT", max_iter=50)
# prob.add_driver("SLSQP", max_iter=100)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
