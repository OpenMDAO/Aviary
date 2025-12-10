import warnings

import dymos as dm
import openmdao.api as om

import aviary.api as av
from aviary.api import Mission
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import inputs
from aviary.utils.preprocessors import preprocess_options


# Note, only creating this aviary problem so that it can read the aircraft csv for us.
prob = AviaryProblem()
prob.load_inputs("models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv", verbosity=0)
aviary_options = prob.aviary_inputs.deepcopy()

# These inputs aren't in the aircraft file yet.
aviary_options.set_val(Mission.Takeoff.AIRPORT_ALTITUDE, 0.0, 'ft')
aviary_options.set_val(Mission.Takeoff.DRAG_COEFFICIENT_MIN, 0.05, 'unitless')
aviary_options.set_val(Mission.Takeoff.LIFT_COEFFICIENT_MAX, 3.0, 'unitless')
aviary_options.set_val(Mission.Takeoff.OBSTACLE_HEIGHT, 35.0, 'ft')
aviary_options.set_val(Mission.Takeoff.ANGLE_OF_ATTACK_RUNWAY, 0.0, 'deg')
aviary_options.set_val(Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, 0.0175)
aviary_options.set_val(Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT, 0.35)
aviary_options.set_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT, 0.085000)
aviary_options.set_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT, -0.810000)
aviary_options.set_val(Mission.Takeoff.THRUST_INCIDENCE, 0.0, 'deg')
aviary_options.set_val(Mission.Takeoff.FUEL_SIMPLE, 577.0, 'lbm')
aviary_options.set_val(Mission.Design.GROSS_MASS, 160000, units='lbm')

# This builder can be used for both takeoff and landing phases
aero_builder = av.CoreAerodynamicsBuilder(
    name='low_speed_aero', code_origin=av.LegacyCode.FLOPS
)

# fmt: off
takeoff_subsystem_options = {
    'low_speed_aero': {
        'method': 'low_speed',
        'ground_altitude': 0.0,  # units='m'
        'angles_of_attack': [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],  # units='deg'
        'lift_coefficients': [
            0.5178, 0.6, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,
            1.35, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95,
        ],
        'drag_coefficients': [
            0.0674, 0.065, 0.065, 0.07, 0.072, 0.076, 0.084, 0.09,
            0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20,
        ],
        'lift_coefficient_factor': 1.0,
        'drag_coefficient_factor': 1.0,
    }
}
# fmt: off

# when using spoilers, add a few more options
takeoff_spoiler_subsystem_options = {
    'low_speed_aero': {
        **takeoff_subsystem_options['low_speed_aero'],
        'use_spoilers': True,
        'spoiler_drag_coefficient': inputs.get_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT),
        'spoiler_lift_coefficient': inputs.get_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT),
    }
}

# We also need propulsion analysis for takeoff and landing. No additional configuration
# is needed for this builder
engines = [av.build_engine_deck(aviary_options)]
preprocess_options(aviary_options, engine_models=engines)
prop_builder = av.CorePropulsionBuilder(engine_models=engines)

balanced_field_user_options = av.AviaryValues()

from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
from aviary.variable_info.functions import setup_model_options

takeoff_trajectory_builder = av.BalancedFieldTrajectoryBuilder('balanced_field_traj',
                                                               core_subsystems=[aero_builder, prop_builder],
                                                               subsystem_options=takeoff_subsystem_options,
                                                               user_options=balanced_field_user_options)

test_problem = om.Problem()

# default subsystems
default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

# Upstream pre-mission analysis for aero
test_problem.model.add_subsystem(
    'core_subsystems',
    av.CorePreMission(
        aviary_options=aviary_options,
        subsystems=default_premission_subsystems,
    ),
    promotes_inputs=['*'],
    promotes_outputs=['*'],
)

# Instantiate the trajectory and add the phases
traj = takeoff_trajectory_builder.build_trajectory(aviary_options=aviary_options, model=test_problem.model)

varnames = [
    av.Aircraft.Wing.AREA,
    av.Aircraft.Wing.ASPECT_RATIO,
    av.Aircraft.Wing.SPAN,
]
av.set_aviary_input_defaults(test_problem.model, varnames, aviary_options)

setup_model_options(test_problem, aviary_options)

# suppress warnings:
# "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
with warnings.catch_warnings():
    warnings.simplefilter('ignore', om.PromotionWarning)
    test_problem.setup(check=False)

av.set_aviary_initial_values(test_problem, aviary_options)

takeoff_trajectory_builder.apply_initial_guesses(test_problem, 'traj')

test_problem.final_setup()


import dymos
dymos.run_problem(test_problem, run_driver=False, make_plots=True)

print(test_problem.get_reports_dir())

