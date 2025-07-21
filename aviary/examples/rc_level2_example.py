"""
This is an example of running a coupled aircraft design-mission optimization in Aviary using the
"level 2" API. It runs the same aircraft and mission as the `level1_example.py` script, but it uses
the AviaryProblem class to set up the problem. This exposes more options and flexibility to the user.

The same ".csv" file is used to define the aircraft, but now the phase_info dictionary is directly
imported from the file and passed as an argument. It is common for level 2 scripts to modify
existing phase_info, but here it is used as-is here to match the level 1 example.

We then call the correct methods in order to set up and run an Aviary optimization problem. Most
methods have optional arguments, but none are necessary here. The selection of the SLSQP optimizer
limited to 50 iterations are included to demonstrate of how those common settings are set.
"""
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from example_phase_info import phase_info

import aviary.api as av

prob = av.AviaryProblem()

# Load aircraft and options data from provided sources
prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info, engine_builders=[RCBuilder()])

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

prob.link_phases()

# optimizer and iteration limit are optional provided here
prob.add_driver('SLSQP', max_iter=100)

prob.add_design_variables()

prob.add_objective()

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
