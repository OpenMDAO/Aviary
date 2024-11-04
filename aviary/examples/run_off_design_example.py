"""
This is a slightly more complex Aviary example of running a coupled aircraft design-mission optimization.
It runs the same mission as the `run_basic_aviary_example.py` script, but it uses the AviaryProblem class to set up the problem.
This exposes more options and flexibility to the user and uses the "Level 2" API within Aviary.

We define a `phase_info` object, which tells Aviary how to model the mission.
Here we have climb, cruise, and descent phases.
We then call the correct methods in order to set up and run an Aviary optimization problem.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""
from openmdao.utils.general_utils import env_truthy
import aviary.api as av

from aviary.interface.default_phase_info.height_energy import phase_info_parameterization
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.variables import Mission

phase_info = {
    "pre_mission": {"include_takeoff": True, "optimize_mass": True},
    "climb": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            'fix_initial': False,
            'input_initial': True,
            "optimize_mach": True,
            "optimize_altitude": True,
            "use_polynomial_control": False,
            "num_segments": 6,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.3, "unitless"),
            "final_mach": (0.79, "unitless"),
            "mach_bounds": ((0.1, 0.8), "unitless"),
            "initial_altitude": (35., "ft"),
            "final_altitude": (35000.0, "ft"),
            "altitude_bounds": ((0.0, 35000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 2.0), "min"),
            "duration_bounds": ((5.0, 50.0), "min"),
            "no_descent": False,
            "add_initial_mass_constraint": False,
        },
        "initial_guesses": {"time": ([0, 40.0], "min")},
    },
    "cruise": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": True,
            "optimize_altitude": True,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 1,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.79, "unitless"),
            "final_mach": (0.79, "unitless"),
            "mach_bounds": ((0.79, 0.79), "unitless"),
            "initial_altitude": (35000.0, "ft"),
            "final_altitude": (35000.0, "ft"),
            "altitude_bounds": ((35000.0, 35000.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((64.0, 192.0), "min"),
            "duration_bounds": ((60.0, 720.0), "min"),
        },
        "initial_guesses": {"time": ([128, 113], "min")},
    },
    "descent": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": True,
            "optimize_altitude": True,
            "use_polynomial_control": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.79, "unitless"),
            "final_mach": (0.3, "unitless"),
            "mach_bounds": ((0.2, 0.8), "unitless"),
            "initial_altitude": (35000.0, "ft"),
            "final_altitude": (35.0, "ft"),
            "altitude_bounds": ((0.0, 35000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((120., 800.), "min"),
            "duration_bounds": ((5.0, 35.0), "min"),
            "no_climb": True,
        },
        "initial_guesses": {"time": ([241, 30], "min")},
    },
    "post_mission": {
        "include_landing": True,
        "constrain_range": True,
        "target_range": (3375.0, "nmi"),
    },
}

##################
# Sizing Mission #
##################
prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

# Preprocess inputs
prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases(phase_info_parameterization=phase_info_parameterization)
prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()
if env_truthy("TESTFLO_RUNNING"):
    prob.add_driver('SLSQP', max_iter=100)
else:
    prob.add_driver('SNOPT', max_iter=100)
prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective()
prob.setup()
prob.set_initial_guesses()
prob.run_aviary_problem()
prob.save_sizing_to_json()

# Fallout Mission
prob_fallout = prob.fallout_mission()

# Alternate Mission
prob_alternate = prob.alternate_mission()

print('--------------')
print('Sizing Results')
print('--------------')
print(f'Design Range = {prob.get_val(av.Mission.Design.RANGE)}')
print(f'Summary Range = {prob.get_val(av.Mission.Summary.RANGE)}')
print(f'Fuel mass = {prob.get_val(av.Mission.Design.FUEL_MASS)}')
print(f'Total fuel mass = {prob.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)}')
print(f'Empty mass = {prob.get_val(av.Aircraft.Design.OPERATING_MASS)}')
print(f'Payload mass = {prob.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)}')
print(f'Design Gross mass = {prob.get_val(av.Mission.Design.GROSS_MASS)}')
print(f'Summary Gross mass = {prob.get_val(av.Mission.Summary.GROSS_MASS)}')

print('---------------')
print('Fallout Results')
print('---------------')
print(f'Design Range = {prob_fallout.get_val(av.Mission.Design.RANGE)}')
print(f'Summary Range = {prob_fallout.get_val(av.Mission.Summary.RANGE)}')
print(f'Fuel mass = {prob_fallout.get_val(av.Mission.Design.FUEL_MASS)}')
print(f'Total fuel mass = {prob_fallout.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)}')
print(f'Empty mass = {prob_fallout.get_val(av.Aircraft.Design.OPERATING_MASS)}')
print(
    f'Payload mass = {prob_fallout.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)}')
print(f'Design Gross mass = {prob_fallout.get_val(av.Mission.Design.GROSS_MASS)}')
print(f'Summary Gross mass = {prob_fallout.get_val(av.Mission.Summary.GROSS_MASS)}')

print('---------------')
print('Alternate Results')
print('---------------')
print(f'Design Range = {prob_alternate.get_val(av.Mission.Design.RANGE)}')
print(f'Summary Range = {prob_alternate.get_val(av.Mission.Summary.RANGE)}')
print(f'Fuel mass = {prob_alternate.get_val(av.Mission.Design.FUEL_MASS)}')
print(f'Total fuel mass = {prob_alternate.get_val(av.Mission.Summary.TOTAL_FUEL_MASS)}')
print(f'Empty mass = {prob_alternate.get_val(av.Aircraft.Design.OPERATING_MASS)}')
print(
    f'Payload mass = {prob_alternate.get_val(av.Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS)}')
print(f'Design Gross mass = {prob_alternate.get_val(av.Mission.Design.GROSS_MASS)}')
print(f'Summary Gross mass = {prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)}')
