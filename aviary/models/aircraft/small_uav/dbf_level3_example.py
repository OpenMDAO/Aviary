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
from aviary.variable_info.enums import LegacyCode


prob = om.Problem(model=om.Group())
driver = prob.driver = om.pyOptSparseDriver()
driver.options['optimizer'] = 'IPOPT'
driver.opt_settings['max_iter'] = 1000
driver.opt_settings['tol'] = 1e-3
driver.opt_settings['print_level'] = 5

########################################
# Aircraft Input Variables and Options #
########################################

aviary_inputs = get_flops_inputs('AdvancedSingleAisle')
# aviary_inputs = AviaryValues()
aviary_inputs.set_val(av.Aircraft.Engine.NUM_ENGINES, 2)
aviary_inputs.set_val(av.Settings.MASS_METHOD, LegacyCode.FLOPS)
aviary_inputs.set_val(av.Aircraft.Wing.AREA, 3.17750636, units = 'ft**2')
aviary_inputs.set_val(av.Mission.Landing.LIFT_COEFFICIENT_MAX, 1.3)
aviary_inputs.set_val(av.Aircraft.Battery.MASS, 0.707, units='kg')
aviary_inputs.set_val(av.Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
aviary_inputs.set_val(av.Aircraft.Battery.VOLTAGE, 22.2, units='V')
aviary_inputs.set_val(av.Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
aviary_inputs.set_val(av.Aircraft.Engine.Motor.MAX_CONT_CURRENT, 70, units='A')
aviary_inputs.set_val(av.Aircraft.Engine.Motor.MASS, 0.288, units='kg')
aviary_inputs.set_val(av.Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
aviary_inputs.set_val(av.Aircraft.Engine.Propeller.PITCH, 10, units='inch')
# aviary_inputs.set_val(av.Aircraft.Wing.)

engines = [RCBuilder()]
av.preprocess_options(aviary_inputs, engine_models=engines)

# define subsystems
aero = av.CoreAerodynamicsBuilder(code_origin=av.LegacyCode('FLOPS'))
geom = av.CoreGeometryBuilder(code_origin=av.LegacyCode('FLOPS'))
# mass = av.CoreMassBuilder(code_origin=av.LegacyCode('FLOPS'))
prop = av.CorePropulsionBuilder(engine_models=engines)

premission_subsystems = [prop, geom, aero,] #geom, aero, mass
mission_subsystems = [aero, prop] #aero, 

#################
# Define Phases #
#################
num_segments_climb = 6
# num_segments_cruise = 1
# num_segments_descent = 5

climb_seg_ends, _ = dm.utils.lgl.lgl(num_segments_climb + 1)

transcription_climb = dm.Radau(
    num_segments=num_segments_climb, order=3, compressed=True, segment_ends=climb_seg_ends
)


climb_options = av.HeightEnergyPhaseBuilder(
    'test_climb',
    user_options={
        'altitude_optimize': (False, 'unitless'),
        'altitude_initial': (0, 'm'),
        'altitude_final': (200, 'ft'),
        'mach_optimize': (False, 'unitless'),
        'mach_initial': (0.01, 'unitless'),
        'mach_final': (0.03108, 'unitless'),
    },
    core_subsystems=mission_subsystems,
    subsystem_options={'core_aerodynamics': {'method': 'computed'}},
    transcription=transcription_climb,
)

# Upstream pre-mission analysis for aero
prob.model.add_subsystem(
    'pre_mission',
    av.CorePreMission(aviary_options=aviary_inputs, subsystems=premission_subsystems),
    promotes_inputs=['aircraft:*'],
    promotes_outputs=['aircraft:*', 'mission:*'],
)

# directly connect phases (strong_couple = True), or use linkage constraints (weak
# coupling / strong_couple=False)
strong_couple = False

climb = climb_options.build_phase(aviary_options=aviary_inputs)
# climb.set_state_options('mass', fix_initial=False, input_initial=True)
climb.set_state_options('distance', fix_initial=False, input_initial=True)
climb.set_control_options('altitude', fix_initial=False)
climb.set_control_options('mach', fix_initial=False)


traj = prob.model.add_subsystem('traj', dm.Trajectory())

# if fix_initial is false, can we always set input_initial to be true for
# necessary states, and then ignore if we use a linkage?
climb.set_time_options(
    fix_initial=True,
    fix_duration=False,
    units='s',
    duration_bounds=(1, 10),
    duration_ref=1.0,
)

traj.add_phase('climb', climb)

###############
# link phases #
###############
phases = ['climb']

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

climb.add_objective('time', loc='final', ref=10)
# prob.model.add_objective('reg_objective', ref=1)

varnames = [
    av.Aircraft.Wing.AREA,
    av.Aircraft.Battery.VOLTAGE,
    av.Aircraft.Engine.Motor.PEAK_CURRENT, 
    av.Aircraft.Engine.Motor.IDLE_CURRENT,
    av.Aircraft.Wing.TAPER_RATIO, 
    av.Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
    av.Aircraft.Wing.THICKNESS_TO_CHORD,
    av.Aircraft.Wing.SWEEP,

]
av.set_aviary_input_defaults(prob.model, varnames, aviary_inputs)

av.setup_model_options(prob, aviary_inputs)

prob.setup()

av.set_aviary_initial_values(prob, aviary_inputs)

############################################
# Initial Settings for States and Controls #
############################################

prob.set_val('traj.climb.t_initial', 0, units='s')
prob.set_val('traj.climb.t_duration', 10, units='s')

prob.set_val(
    'traj.climb.controls:altitude',
    climb.interp(av.Dynamic.Mission.ALTITUDE, ys=[0, 200]),
    units='ft',
)
prob.set_val(
    'traj.climb.controls:mach',
    climb.interp(av.Dynamic.Atmosphere.MACH, ys=[0.01, 0.03108]),
    units='unitless',
)
#May need to delete
# prob.set_val(
#     'traj.climb.states:mass',
#     climb.interp(av.Dynamic.Vehicle.MASS, ys=[mass_i_climb, mass_f_climb]),
#     units='kg',
# )
prob.set_val(
    'traj.climb.states:distance',
    climb.interp(av.Dynamic.Mission.DISTANCE, ys=[0, 50]),
    units='ft',
)


# Turn off solver printing so that the optimizer output is readable.
prob.set_solver_print(level=0)

dm.run_problem(
    prob,
    simulate=False,
    make_plots=False,
    solution_record_file='dbf_sizing.db',
)
prob.record('final')
prob.cleanup()

times_climb = prob.get_val('traj.climb.timeseries.time', units='s')
altitudes_climb = prob.get_val('traj.climb.timeseries.altitude', units='m')
masses_climb = prob.get_val('traj.climb.timeseries.mass', units='kg')
ranges_climb = prob.get_val('traj.climb.timeseries.distance', units='m')
velocities_climb = prob.get_val('traj.climb.timeseries.velocity', units='m/s')
thrusts_climb = prob.get_val('traj.climb.timeseries.thrust_net_total', units='N')



print('-------------------------------')
print(f'times_climb: {times_climb[-1]} (s)')
print(f'altitudes_climb: {altitudes_climb[-1]} (m)')
print(f'masses_climb: {masses_climb[-1]} (kg)')
print(f'ranges_climb: {ranges_climb[-1]} (m)')
print(f'velocities_climb: {velocities_climb[-1]} (m/s)')
print(f'thrusts_climb: {thrusts_climb[-1]} (N)')
