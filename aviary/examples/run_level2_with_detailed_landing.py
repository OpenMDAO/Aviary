import openmdao.api as om

import aviary.api as av

# fmt: off
subsystem_options = {
    'core_aerodynamics': {
        'method': 'low_speed',
        'ground_altitude': 0.0,  # units='ft'
        'angles_of_attack': [
            -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            6.0,  7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],  # units='deg'
        'lift_coefficients': [
            0.01, 0.1, 0.2, 0.3, 0.4, 0.5178, 0.6, 0.75, 0.85, 0.95, 1.05,
            1.15, 1.25, 1.35, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95,
        ],
        'drag_coefficients': [
            0.04, 0.02, 0.01, 0.02, 0.04, 0.0674, 0.065, 0.065, 0.07, 0.072,
            0.076, 0.084, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20,
        ],
        'lift_coefficient_factor': 2.0,
        'drag_coefficient_factor': 2.0,
    }
}
# fmt: on

subsystem_options_landing = subsystem_options.copy()
subsystem_options_landing['core_aerodynamics']['drag_coefficient_factor'] = 3.0

mach_optimize = False
altitude_optimize = False

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': False},
    'GH': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'ground_roll': False,
            'clean': False,
            'time_initial': (0.0, 'ft'),
            'time_duration_ref': (2750.0, 'ft'),
            'time_duration_bounds': ((500.0, 5.0e3), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.1, 0.5), 'unitless'),
            'mach_initial': (0.15, 'unitless'),
            'mach_final': (0.15, 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_initial': (500.0, 'ft'),
            'altitude_final': (394.0, 'ft'),
            'altitude_bounds': ((0.0, 1000.0), 'ft'),
            'mass_initial': (120.0e3, 'lbm'),
            'throttle_enforcement': 'bounded',
            'rotation': False,
            'constraints': {
                'flight_path_angle': {
                    'equals': -3.0,
                    'loc': 'initial',
                    'units': 'deg',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(0.0e3, 2.0e3), 'ft'],
            'time': [(0.0, 12.0), 's'],
            'mass': [(120.0e3, 119.8e3), 'lbm'],
        },
    },
    'HI': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'ground_roll': False,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((0.0, 16.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((500.0, 15.0e3), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_initial': (0.15, 'unitless'),
            'mach_final': (0.15, 'unitless'),
            'mach_bounds': ((0.1, 0.5), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_initial': (394.0, 'ft'),
            'altitude_final': (50.0, 'ft'),
            'altitude_bounds': ((0.0, 1000.0), 'ft'),
            'throttle_enforcement': 'bounded',
            'rotation': False,
            'constraints': {
                'flight_path_angle': {
                    'equals': -3.0,
                    'loc': 'final',
                    'units': 'deg',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(2.0e3, 6.5e3), 'ft'],
            'time': [(12.0, 50.0), 's'],
            'mass': [(119.8e3, 119.7e3), 'lbm'],
        },
    },
    'IJ': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'ground_roll': False,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((0.0, 30.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((500.0, 15.0e3), 'ft'),
            'mach_optimize': False,
            'mach_polynomial_order': 2,
            'mach_bounds': ((0.1, 0.5), 'unitless'),
            'mach_initial': (0.15, 'unitless'),
            'mach_final': (0.15, 'unitless'),
            'altitude_optimize': True,
            'altitude_polynomial_order': 2,
            #'altitude_initial': (50.0, 'ft'),
            #'altitude_final': (0.0, 'ft'),
            'altitude_bounds': ((0.0, 1000.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'rotation': False,
            'constraints': {},
        },
        'subsystem_options': subsystem_options_landing,
        'initial_guesses': {
            'altitude': [(50.0, 0.0), 'ft'],
            'distance': [(8.5e3, 2.0e3), 'ft'],
            'time': [(50.0, 60.0), 's'],
            'mass': [(119.7e3, 119.67e3), 'lbm'],
        },
    },
    'post_mission': {
        'include_landing': False,
        'constrain_range': False,
    },
}

prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_solved2dof.csv', phase_info)

# Preprocess inputs
prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver('SLSQP', max_iter=100)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective('mass')

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem(record_filename='detailed_landing.db')

try:
    loc = prob.get_outputs_dir()
    cr = om.CaseReader(f'{loc}/detailed_landing.db')
except:
    cr = om.CaseReader('detailed_landing.db')

cases = cr.get_cases('problem')
case = cases[0]

output_data = {}

point_name = 'P3'
phase_name = 'GH'
output_data[point_name] = {}
output_data[point_name]['thrust_fraction'] = (
    case.get_val(f'traj.{phase_name}.rhs_all.thrust_net', units='N')[-1][0]
    / case.get_val(f'traj.{phase_name}.rhs_all.thrust_net_max', units='N')[-1][0]
)
output_data[point_name]['true_airspeed'] = case.get_val(
    f'traj.{phase_name}.timeseries.velocity', units='kn'
)[-1][0]
output_data[point_name]['angle_of_attack'] = case.get_val(
    f'traj.{phase_name}.timeseries.angle_of_attack', units='deg'
)[-1][0]
output_data[point_name]['flight_path_angle'] = case.get_val(
    f'traj.{phase_name}.timeseries.flight_path_angle', units='deg'
)[-1][0]
output_data[point_name]['altitude'] = case.get_val(
    f'traj.{phase_name}.timeseries.altitude', units='ft'
)[-1][0]
output_data[point_name]['distance'] = case.get_val(
    f'traj.{phase_name}.timeseries.distance', units='ft'
)[-1][0]

print(output_data)
