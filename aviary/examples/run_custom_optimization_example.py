"""
This is an example of running constrained optimization in Aviary using the "level 2" API. It runs
the same aircraft and mission as the `level1_example.py` script, but it uses the AviaryProblem class
to set up the problem.

The same ".csv" file is used to define the aircraft, but wing area and engine scale factor are added
as design variables. Then, wing loading and thrust-to-weight ratio are constrained to arbitrary
limits. If this example is run without these constraints, wing area is increased to its upper bound
and engine scale factor is reduced to its lower bound.
"""

from aviary.models.missions.height_energy_default import phase_info

import aviary.api as av

# Suppress outputs
prob = av.AviaryProblem(verbosity=0)

# Load aircraft and options data from provided sources
prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

prob.link_phases()

# Optimizer and iteration limit are optional provided here
prob.add_driver('SLSQP', max_iter=20)

# Add the default design variables needed to size the aircraft
prob.add_design_variables()

# Add wing area and engine scaling as additional design variables
prob.model.add_design_var(av.Aircraft.Engine.SCALE_FACTOR, lower=0.8, upper=1.2, ref=1)
prob.model.add_design_var(av.Aircraft.Wing.AREA, lower=1200, upper=1800, units='ft**2', ref=1400)

prob.add_objective()

# Constrain wing loading and thrust-to-weight ratio
prob.model.add_constraint(av.Aircraft.Design.WING_LOADING, lower=120, units='lbf/ft**2')
prob.model.add_constraint(av.Aircraft.Design.THRUST_TO_WEIGHT_RATIO, lower=0.35)

prob.setup()

prob.run_aviary_problem()

print(f'\nTakeoff Gross Weight = {prob.get_val(av.Mission.Summary.GROSS_MASS, units="lbm")} lbm')
print('\nDesign Variables\n---------------')
print(f'Engine Scale Factor (started at 1) = {prob.get_val(av.Aircraft.Engine.SCALE_FACTOR)}')
print(f'Wing Area (started at 1370) = {prob.get_val(av.Aircraft.Wing.AREA, units="ft**2")} ft^2')
print('\nConstraints\n-----------')
print(f'Wing Loading = {prob.get_val(av.Aircraft.Design.WING_LOADING, units="lbf/ft**2")} lbf/ft^2')
print(f'Thrust/Weight Ratio = {prob.get_val(av.Aircraft.Design.THRUST_TO_WEIGHT_RATIO)}')
