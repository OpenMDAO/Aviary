"""
NOTES:
Includes:
Takeoff, Climb, Cruise, Descent, Landing
Computed Aero
advanced_single_aisle data
"""

import dymos as dm
import openmdao.api as om
import scipy.constants as _units
import numpy as np
import aviary.api as av
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_model_options
from aviary.utils.aviary_values import AviaryValues


prob = om.Problem(model=om.Group())
driver = prob.driver = om.pyOptSparseDriver()
driver.options['optimizer'] = 'IPOPT'
driver.opt_settings['max_iter'] = 1000
driver.opt_settings['tol'] = 1e-3
driver.opt_settings['print_level'] = 5

########################################
# Aircraft Input Variables and Options #
########################################

# aviary_inputs = get_flops_inputs('AdvancedSingleAisle')
aviary_inputs = AviaryValues()
aviary_inputs.set_val(av.Aircraft.Engine.NUM_ENGINES, 2)
# aviary_inputs.set_val(av.Mission.Landing.LIFT_COEFFICIENT_MAX, 2.4, units='unitless')
# aviary_inputs.set_val(av.Mission.Takeoff.LIFT_COEFFICIENT_MAX, 2.0, units='unitless')
# aviary_inputs.set_val(av.Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, val=0.0175, units='unitless')

# takeoff_fuel_burned = 577  # lbm
# takeoff_thrust_per_eng = 24555.5  # lbf
# takeoff_L_over_D = 17.35

# aviary_inputs.set_val(av.Mission.Takeoff.FUEL_SIMPLE, takeoff_fuel_burned, units='lbm')
# aviary_inputs.set_val(av.Mission.Takeoff.LIFT_OVER_DRAG, takeoff_L_over_D, units='unitless')
# aviary_inputs.set_val(av.Mission.Design.THRUST_TAKEOFF_PER_ENG, takeoff_thrust_per_eng, units='lbf')

alt_airport = 0  # ft
cruise_mach = 0.79

alt_i_climb = 0 * _units.foot  # m
alt_f_climb = 35000.0 * _units.foot  # m
mass_i_climb = 131000 * _units.lb  # kg
mass_f_climb = 126000 * _units.lb  # kg
# initial mach set to lower value so it can intersect with takeoff end mach
# mach_i_climb = 0.3
mach_i_climb = 0.2
mach_f_climb = cruise_mach
range_i_climb = 0 * _units.nautical_mile  # m
range_f_climb = 160.3 * _units.nautical_mile  # m
t_i_climb = 2 * _units.minute  # sec
t_f_climb = 26.20 * _units.minute  # sec
t_duration_climb = t_f_climb - t_i_climb

alt_i_cruise = 35000 * _units.foot  # m
alt_f_cruise = 35000 * _units.foot  # m
alt_min_cruise = 35000 * _units.foot  # m
alt_max_cruise = 35000 * _units.foot  # m
mass_i_cruise = 126000 * _units.lb  # kg
mass_f_cruise = 102000 * _units.lb  # kg
cruise_mach = cruise_mach
range_i_cruise = 160.3 * _units.nautical_mile  # m
range_f_cruise = 3243.9 * _units.nautical_mile  # m
t_i_cruise = 26.20 * _units.minute  # sec
t_f_cruise = 432.38 * _units.minute  # sec
t_duration_cruise = t_f_cruise - t_i_cruise

alt_i_descent = 35000 * _units.foot
# final altitude set to 35 to ensure landing is feasible point
# alt_f_descent = 0*_units.foot
alt_f_descent = 35 * _units.foot
mach_i_descent = 0.79
mach_f_descent = 0.3
mass_i_descent = 102000 * _units.pound
mass_f_descent = 101000 * _units.pound
distance_i_descent = 3243.9 * _units.nautical_mile
distance_f_descent = 3378.7 * _units.nautical_mile
t_i_descent = 432.38 * _units.minute
t_f_descent = 461.62 * _units.minute
t_duration_descent = t_f_descent - t_i_descent

engines = [RCBuilder()]#[av.build_engine_deck(aviary_inputs)]
av.preprocess_options(aviary_inputs, engine_models=engines)

# define subsystems
# aero = av.CoreAerodynamicsBuilder(code_origin=av.LegacyCode('FLOPS'))
# geom = av.CoreGeometryBuilder(code_origin=av.LegacyCode('FLOPS'))
# mass = av.CoreMassBuilder(code_origin=av.LegacyCode('FLOPS'))
prop = av.CorePropulsionBuilder(engine_models=engines)

premission_subsystems = [prop] #aviary/examples/ipopt.opt
mission_subsystems = [prop] #aero, 

####################
# Design Variables #
####################

# Nudge it a bit off the correct answer to verify that the optimize takes us there.
# aviary_inputs.set_val(av.Mission.Design.GROSS_MASS, 135000.0, units='lbm')
# aviary_inputs.set_val(av.Mission.Summary.GROSS_MASS, 135000.0, units='lbm')

prob.model.add_design_var(
    av.Mission.Design.GROSS_MASS, units='lbm', lower=100000.0, upper=200000.0, ref=135000
)
prob.model.add_design_var(
    av.Mission.Summary.GROSS_MASS, units='lbm', lower=100000.0, upper=200000.0, ref=135000
)

takeoff_options = av.HeightEnergyTakeoffPhaseBuilder(
    airport_altitude=alt_airport,  # ft
    # no units
    num_engines=aviary_inputs.get_val(av.Aircraft.Engine.NUM_ENGINES),
)

#################
# Define Phases #
#################
num_segments_climb = 6
num_segments_cruise = 1
num_segments_descent = 5

climb_seg_ends, _ = dm.utils.lgl.lgl(num_segments_climb + 1)
descent_seg_ends, _ = dm.utils.lgl.lgl(num_segments_descent + 1)

transcription_climb = dm.Radau(
    num_segments=num_segments_climb, order=3, compressed=True, segment_ends=climb_seg_ends
)
transcription_cruise = dm.Radau(num_segments=num_segments_cruise, order=3, compressed=True)
transcription_descent = dm.Radau(
    num_segments=num_segments_descent, order=3, compressed=True, segment_ends=descent_seg_ends
)

climb_options = av.HeightEnergyPhaseBuilder(
    'test_climb',
    user_options={
        'altitude_optimize': (False, 'unitless'),
        'altitude_initial': (alt_i_climb, 'm'),
        'altitude_final': (alt_f_climb, 'm'),
        'mach_optimize': (False, 'unitless'),
        'mach_initial': (mach_i_climb, 'unitless'),
        'mach_final': (mach_f_climb, 'unitless'),
    },
    core_subsystems=mission_subsystems,
    subsystem_options={'core_aerodynamics': {'method': 'computed'}},
    transcription=transcription_climb,
)

cruise_options = av.HeightEnergyPhaseBuilder(
    'test_cruise',
    user_options={
        'altitude_optimize': (False, 'unitless'),
        'altitude_polynomial_order': 3,
        'altitude_initial': (alt_min_cruise, 'm'),
        'altitude_final': (alt_max_cruise, 'm'),
        'mach_optimize': (False, 'unitless'),
        'mach_polynomial_order': 3,
        'mach_initial': (cruise_mach, 'unitless'),
        'mach_final': (cruise_mach, 'unitless'),
        'required_available_climb_rate': (300, 'ft/min'),
    },
    core_subsystems=mission_subsystems,
    subsystem_options={'core_aerodynamics': {'method': 'computed'}},
    transcription=transcription_cruise,
)

descent_options = av.HeightEnergyPhaseBuilder(
    'test_descent',
    user_options={
        'altitude_optimize': (False, 'unitless'),
        'altitude_final': (alt_f_descent, 'm'),
        'altitude_initial': (alt_i_descent, 'm'),
        'mach_optimize': (False, 'unitless'),
        'mach_initial': (mach_i_descent, 'unitless'),
        'mach_final': (mach_f_descent, 'unitless'),
    },
    core_subsystems=mission_subsystems,
    subsystem_options={'core_aerodynamics': {'method': 'computed'}},
    transcription=transcription_descent,
)

landing_options = av.HeightEnergyLandingPhaseBuilder(
    ref_wing_area=aviary_inputs.get_val(av.Aircraft.Wing.AREA, units='ft**2'),
    Cl_max_ldg=aviary_inputs.get_val(av.Mission.Landing.LIFT_COEFFICIENT_MAX),  # no units
)

# Upstream pre-mission analysis for aero
prob.model.add_subsystem(
    'pre_mission',
    av.CorePreMission(aviary_options=aviary_inputs, subsystems=premission_subsystems),
    promotes_inputs=['aircraft:*', 'mission:*'],
    promotes_outputs=['aircraft:*', 'mission:*'],
)

# directly connect phases (strong_couple = True), or use linkage constraints (weak
# coupling / strong_couple=False)
strong_couple = False

takeoff = takeoff_options.build_phase(False)

climb = climb_options.build_phase(aviary_options=aviary_inputs)
climb.set_state_options('mass', fix_initial=False, input_initial=True)
climb.set_state_options('distance', fix_initial=False, input_initial=True)
climb.set_control_options('altitude', fix_initial=False)
climb.set_control_options('mach', fix_initial=False)

cruise = cruise_options.build_phase(aviary_options=aviary_inputs)
cruise.set_state_options('mass', fix_initial=False, input_initial=False)
cruise.set_state_options('distance', fix_initial=False, input_initial=False)
cruise.set_control_options('mach', fix_initial=False)

descent = descent_options.build_phase(aviary_options=aviary_inputs)
descent.set_state_options('mass', fix_initial=False, input_initial=False)
descent.set_state_options('distance', fix_initial=False, input_initial=False)
descent.set_control_options('mach', fix_initial=False)

landing = landing_options.build_phase(False)

prob.model.add_subsystem(
    'takeoff', takeoff, promotes_inputs=['aircraft:*', 'mission:*'], promotes_outputs=['mission:*']
)

traj = prob.model.add_subsystem('traj', dm.Trajectory())

# if fix_initial is false, can we always set input_initial to be true for
# necessary states, and then ignore if we use a linkage?
climb.set_time_options(
    fix_initial=True,
    fix_duration=False,
    units='s',
    duration_bounds=(t_duration_climb * 0.5, t_duration_climb * 2),
    duration_ref=t_duration_climb,
)
cruise.set_time_options(
    fix_initial=False,
    fix_duration=False,
    units='s',
    duration_bounds=(t_duration_cruise * 0.5, t_duration_cruise * 2),
    duration_ref=t_duration_cruise,
    initial_bounds=(t_duration_climb * 0.5, t_duration_climb * 2),
)
descent.set_time_options(
    fix_initial=False,
    fix_duration=False,
    units='s',
    duration_bounds=(t_duration_descent * 0.5, t_duration_descent * 2),
    duration_ref=t_duration_descent,
    initial_bounds=(
        (t_duration_cruise + t_duration_climb) * 0.5,
        (t_duration_cruise + t_duration_climb) * 2,
    ),
)

traj.add_phase('climb', climb)

traj.add_phase('cruise', cruise)

traj.add_phase('descent', descent)

prob.model.add_subsystem(
    'landing', landing, promotes_inputs=['aircraft:*', 'mission:*'], promotes_outputs=['mission:*']
)

###############
# link phases #
###############
phases = ['climb', 'cruise', 'descent']
traj.link_phases(
    phases, ['time', av.Dynamic.Vehicle.MASS, av.Dynamic.Mission.DISTANCE], connected=strong_couple
)

# loop through phases and get all subsystem parameters
external_parameters = {}
for phase_name in phases:
    external_parameters[phase_name] = {}
    for subsystem in mission_subsystems:
        parameter_dict = subsystem.get_parameters(phase_info={}, aviary_inputs=aviary_inputs)
        for parameter in parameter_dict:
            external_parameters[phase_name][parameter] = parameter_dict[parameter]

traj = av.setup_trajectory_params(
    prob.model, traj, aviary_inputs, phases, external_parameters=external_parameters
)

##################################
# Connect in Takeoff and Landing #
##################################
prob.model.connect(av.Mission.Takeoff.FINAL_MASS, 'traj.climb.initial_states:mass')
prob.model.connect(av.Mission.Takeoff.GROUND_DISTANCE, 'traj.climb.initial_states:distance')

prob.model.connect('traj.descent.states:mass', av.Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])
prob.model.connect(
    'traj.descent.control_values:altitude', av.Mission.Landing.INITIAL_ALTITUDE, src_indices=[-1]
)

###############
# Constraints #
###############

ecomp = om.ExecComp(
    'fuel_burned = initial_mass - descent_mass_final',
    initial_mass={'units': 'lbm', 'shape': 1},
    descent_mass_final={'units': 'lbm', 'shape': 1},
    fuel_burned={'units': 'lbm', 'shape': 1},
)

prob.model.add_subsystem(
    'fuel_burn',
    ecomp,
    promotes_inputs=[('initial_mass', av.Mission.Design.GROSS_MASS)],
    promotes_outputs=['fuel_burned'],
)

prob.model.connect('traj.descent.states:mass', 'fuel_burn.descent_mass_final', src_indices=[-1])

ecomp = om.ExecComp(
    'overall_fuel = fuel_burned + fuel_reserve',
    fuel_burned={'units': 'lbm', 'shape': 1},
    fuel_reserve={'units': 'lbm', 'val': 2173.0},
    overall_fuel={'units': 'lbm'},
)
prob.model.add_subsystem(
    'fuel_calc', ecomp, promotes_inputs=['fuel_burned'], promotes_outputs=['overall_fuel']
)

ecomp = om.ExecComp(
    'mass_resid = operating_empty_mass + overall_fuel + payload_mass - initial_mass',
    operating_empty_mass={'units': 'lbm'},
    overall_fuel={'units': 'lbm'},
    payload_mass={'units': 'lbm'},
    initial_mass={'units': 'lbm'},
    mass_resid={'units': 'lbm'},
)

prob.model.add_subsystem(
    'mass_constraint',
    ecomp,
    promotes_inputs=[
        ('operating_empty_mass', av.Aircraft.Design.OPERATING_MASS),
        'overall_fuel',
        ('payload_mass', av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS),
        ('initial_mass', av.Mission.Design.GROSS_MASS),
    ],
    promotes_outputs=['mass_resid'],
)

prob.model.add_constraint('mass_resid', equals=0.0, ref=1.0)

prob.model.add_subsystem(
    'gtow_constraint',
    om.EQConstraintComp(
        'GTOW',
        eq_units='lbm',
        normalize=True,
        add_constraint=True,
    ),
    promotes_inputs=[
        ('lhs:GTOW', av.Mission.Design.GROSS_MASS),
        ('rhs:GTOW', av.Mission.Summary.GROSS_MASS),
    ],
)

##########################
# Add Objective Function #
##########################

# This is an example of a overall mission objective
# create a compound objective that minimizes climb time and maximizes final mass
# we are maxing final mass b/c we don't have an independent value for fuel_mass yet
# we are going to normalize these (making each of the sub-objectives approx = 1 )
prob.model.add_subsystem(
    'regularization',
    om.ExecComp(
        'reg_objective = fuel_mass/1500',
        reg_objective=0.0,
        fuel_mass={'units': 'lbm', 'shape': 1},
    ),
    promotes_outputs=['reg_objective'],
)
# connect the final mass from cruise into the objective
prob.model.connect(av.Mission.Design.FUEL_MASS, 'regularization.fuel_mass')

prob.model.add_objective('reg_objective', ref=1)

# Set initial default values for all LEAPS aircraft variables.
varnames = [
    av.Aircraft.Engine.SCALE_FACTOR,
    av.Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
    av.Aircraft.Wing.SWEEP,
    av.Aircraft.Wing.TAPER_RATIO,
    av.Aircraft.Wing.THICKNESS_TO_CHORD,
    av.Mission.Design.GROSS_MASS,
    av.Mission.Summary.GROSS_MASS,
]
av.set_aviary_input_defaults(prob.model, varnames, aviary_inputs)

av.setup_model_options(prob, aviary_inputs)

prob.setup()

phase = prob.model.traj.phases.cruise
phase.nonlinear_solver = om.NonlinearRunOnce()
phase.linear_solver = om.LinearRunOnce()
if isinstance(phase.indep_states, om.ImplicitComponent):
    phase.indep_states.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    phase.indep_states.linear_solver = om.DirectSolver(rhs_checking=True)

phase = prob.model.traj.phases.descent
phase.nonlinear_solver = om.NonlinearRunOnce()
phase.linear_solver = om.LinearRunOnce()
if isinstance(phase.indep_states, om.ImplicitComponent):
    phase.indep_states.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    phase.indep_states.linear_solver = om.DirectSolver(rhs_checking=True)

av.set_aviary_initial_values(prob, aviary_inputs)

############################################
# Initial Settings for States and Controls #
############################################

prob.set_val('traj.climb.t_initial', t_i_climb, units='s')
prob.set_val('traj.climb.t_duration', t_duration_climb, units='s')

prob.set_val(
    'traj.climb.controls:altitude',
    climb.interp(av.Dynamic.Mission.ALTITUDE, ys=[alt_i_climb, alt_f_climb]),
    units='m',
)
prob.set_val(
    'traj.climb.controls:mach',
    climb.interp(av.Dynamic.Atmosphere.MACH, ys=[mach_i_climb, mach_f_climb]),
    units='unitless',
)
prob.set_val(
    'traj.climb.states:mass',
    climb.interp(av.Dynamic.Vehicle.MASS, ys=[mass_i_climb, mass_f_climb]),
    units='kg',
)
prob.set_val(
    'traj.climb.states:distance',
    climb.interp(av.Dynamic.Mission.DISTANCE, ys=[range_i_climb, range_f_climb]),
    units='m',
)

prob.set_val('traj.cruise.t_initial', t_i_cruise, units='s')
prob.set_val('traj.cruise.t_duration', t_duration_cruise, units='s')

prob.set_val(
    'traj.cruise.controls:altitude',
    cruise.interp(av.Dynamic.Mission.ALTITUDE, ys=[alt_i_cruise, alt_f_cruise]),
    units='m',
)
prob.set_val(
    'traj.cruise.controls:mach',
    cruise.interp(av.Dynamic.Atmosphere.MACH, ys=[cruise_mach, cruise_mach]),
    units='unitless',
)
prob.set_val(
    'traj.cruise.states:mass',
    cruise.interp(av.Dynamic.Vehicle.MASS, ys=[mass_i_cruise, mass_f_cruise]),
    units='kg',
)
prob.set_val(
    'traj.cruise.states:distance',
    cruise.interp(av.Dynamic.Mission.DISTANCE, ys=[range_i_cruise, range_f_cruise]),
    units='m',
)

prob.set_val('traj.descent.t_initial', t_i_descent, units='s')
prob.set_val('traj.descent.t_duration', t_duration_descent, units='s')

prob.set_val(
    'traj.descent.controls:altitude',
    descent.interp(av.Dynamic.Mission.ALTITUDE, ys=[alt_i_descent, alt_f_descent]),
    units='m',
)
prob.set_val(
    'traj.descent.controls:mach',
    descent.interp(av.Dynamic.Atmosphere.MACH, ys=[mach_i_descent, mach_f_descent]),
    units='unitless',
)
prob.set_val(
    'traj.descent.states:mass',
    descent.interp(av.Dynamic.Vehicle.MASS, ys=[mass_i_descent, mass_f_descent]),
    units='kg',
)
prob.set_val(
    'traj.descent.states:distance',
    descent.interp(av.Dynamic.Mission.DISTANCE, ys=[distance_i_descent, distance_f_descent]),
    units='m',
)

# Turn off solver printing so that the optimizer output is readable.
prob.set_solver_print(level=0)

dm.run_problem(
    prob,
    simulate=False,
    make_plots=False,
    solution_record_file='N3CC_sizing.db',
)
prob.record('final')
prob.cleanup()

times_climb = prob.get_val('traj.climb.timeseries.time', units='s')
altitudes_climb = prob.get_val('traj.climb.timeseries.altitude', units='m')
masses_climb = prob.get_val('traj.climb.timeseries.mass', units='kg')
ranges_climb = prob.get_val('traj.climb.timeseries.distance', units='m')
velocities_climb = prob.get_val('traj.climb.timeseries.velocity', units='m/s')
thrusts_climb = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')
times_cruise = prob.get_val('traj.cruise.timeseries.time', units='s')
altitudes_cruise = prob.get_val('traj.cruise.timeseries.altitude', units='m')
masses_cruise = prob.get_val('traj.cruise.timeseries.mass', units='kg')
ranges_cruise = prob.get_val('traj.cruise.timeseries.distance', units='m')
velocities_cruise = prob.get_val('traj.cruise.timeseries.velocity', units='m/s')
thrusts_cruise = prob.get_val('traj.cruise.timeseries.thrust_net_total', units='N')
times_descent = prob.get_val('traj.descent.timeseries.time', units='s')
altitudes_descent = prob.get_val('traj.descent.timeseries.altitude', units='m')
masses_descent = prob.get_val('traj.descent.timeseries.mass', units='kg')
ranges_descent = prob.get_val('traj.descent.timeseries.distance', units='m')
velocities_descent = prob.get_val('traj.descent.timeseries.velocity', units='m/s')
thrusts_descent = prob.get_val('traj.descent.timeseries.thrust_net_total', units='N')


print('-------------------------------')
print(f'times_climb: {times_climb[-1]} (s)')
print(f'altitudes_climb: {altitudes_climb[-1]} (m)')
print(f'masses_climb: {masses_climb[-1]} (kg)')
print(f'ranges_climb: {ranges_climb[-1]} (m)')
print(f'velocities_climb: {velocities_climb[-1]} (m/s)')
print(f'thrusts_climb: {thrusts_climb[-1]} (N)')
print(f'times_cruise: {times_cruise[-1]} (s)')
print(f'altitudes_cruise: {altitudes_cruise[-1]} (m)')
print(f'masses_cruise: {masses_cruise[-1]} (kg)')
print(f'ranges_cruise: {ranges_cruise[-1]} (m)')
print(f'velocities_cruise: {velocities_cruise[-1]} (m/s)')
print(f'thrusts_cruise: {thrusts_cruise[-1]} (N)')
print(f'times_descent: {times_descent[-1]} (s)')
print(f'altitudes_descent: {altitudes_descent[-1]} (m)')
print(f'masses_descent: {masses_descent[-1]} (kg)')
print(f'ranges_descent: {ranges_descent[-1]} (m)')
print(f'velocities_descent: {velocities_descent[-1]} (m/s)')
print(f'thrusts_descent: {thrusts_descent[-1]} (N)')
print('-------------------------------')