import openmdao.api as om

import aviary.api as av

# fmt: off
subsystem_options = {
    'core_aerodynamics': {
        'method': 'low_speed',
        'ground_altitude': 0.0,  # units='ft'
        'angles_of_attack': [
            -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ],  # units='deg'
        'lift_coefficients': [
            0.01, 0.1, 0.2, 0.3, 0.4, 0.5178, 0.6, 0.75, 0.85, 0.95,
            1.05, 1.15, 1.25, 1.35, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95,
        ],
        'drag_coefficients': [
            0.04, 0.02, 0.01, 0.02, 0.04, 0.0674, 0.065, 0.065, 0.07, 0.072,
            0.076, 0.084, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20,
        ],
        'lift_coefficient_factor': 1.0,
        'drag_coefficient_factor': 1.0,
    }
}
# fmt: on

mach_optimize = True
altitude_optimize = True
optimizer = 'SLSQP'

phase_info = {
    'pre_mission': {'include_takeoff': False, 'optimize_mass': False},
    'AB': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'ground_roll': True,
            'time_duration_ref': (100.0, 'kn'),
            'time_duration_bounds': ((100.0, 500.0), 'kn'),
            'time_initial': (0.0, 'kn'),
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(0.0, 2.0e3), 'ft'],
            'time': [(0.0, 20.0), 's'],
            'velocity': [(1.0, 120.0), 'kn'],
            'mass': [(175.0e3, 174.85e3), 'lbm'],
        },
    },
    'rotate': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'ground_roll': True,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((1.0e3, 3.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((200.0, 2.0e3), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'rotation': True,
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.18, 0.2), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (0.0, 'ft'),
            'altitude_final': (0.0, 'ft'),
            'constraints': {
                'normal_force': {
                    'equals': 0.0,
                    'loc': 'final',
                    'units': 'lbf',
                    'type': 'boundary',
                    'ref': 10.0e5,
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(2.0e3, 1.0e3), 'ft'],
            'time': [(20.0, 25.0), 's'],
            'mach': [(0.18, 0.2), 'unitless'],
            'mass': [(174.85e3, 174.84e3), 'lbm'],
            'angle_of_attack': [(0.0, 12.0), 'deg'],
        },
    },
    'BC': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((1.0, 16.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((500.0, 1500.0), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.2, 0.22), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_bounds': ((0.0, 250.0), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'rotation': False,
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(3.0e3, 1.0e3), 'ft'],
            'time': [(25.0, 35.0), 's'],
            'mach': [(0.2, 0.22), 'unitless'],
            'altitude': [(0.0, 50.0), 'ft'],
            'mass': [(174.84e3, 174.82e3), 'lbm'],
        },
    },
    'CD_to_P2': {
        'user_options': {
            'num_segments': 4,
            'order': 3,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((1.0e3, 20.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((3.0e3, 20.0e3), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.22, 0.3), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_initial': (50.0, 'ft'),
            'altitude_final': (985.0, 'ft'),
            'altitude_bounds': ((0.0, 985.0), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'constraints': {
                'altitude': {
                    'equals': 985.0,
                    'loc': 'final',
                    'units': 'ft',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(4.0e3, 10.0e3), 'ft'],
            'time': [(35.0, 60.0), 's'],
            'mach': [(0.22, 0.3), 'unitless'],
            'mass': [(174.82e3, 174.8e3), 'lbm'],
        },
    },
    'P2_to_DE': {
        'user_options': {
            'num_segments': 4,
            'order': 3,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((1.0e3, 20.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((3.0e3, 20.0e3), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.22, 0.3), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_bounds': ((985.0, 1100.0), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'constraints': {
                'distance': {
                    'upper': 19.0e3,
                    'ref': 20.0e3,
                    'loc': 'final',
                    'units': 'ft',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(10.0e3, 14.0e3), 'ft'],
            'time': [(60.0, 80.0), 's'],
            'mach': [(0.22, 0.3), 'unitless'],
            'altitude': [(985.0, 1100.0), 'ft'],
            'mass': [(174.8e3, 174.5e3), 'lbm'],
        },
    },
    'DE': {
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((500.0, 30.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((50.0, 5000.0), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 2,
            'mach_bounds': ((0.24, 0.32), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 2,
            'altitude_bounds': ((985.0, 1.5e3), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'constraints': {
                'flight_path_angle': {
                    'equals': 4.0,
                    'loc': 'final',
                    'units': 'deg',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(18.0e3, 2.0e3), 'ft'],
            'mass': [(174.5e3, 174.4e3), 'lbm'],
            'mach': [(0.3, 0.3), 'unitless'],
            'altitude': [(1100.0, 1200.0), 'ft'],
            'time': [(80.0, 85.0), 's'],
        },
    },
    'EF_to_P1': {
        'user_options': {
            'num_segments': 3,
            'order': 3,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((500.0, 50.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((1.0e3, 20.0e3), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.24, 0.32), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_bounds': ((1.1e3, 1.2e3), 'ft'),
            'throttle_enforcement': 'path_constraint',
            'constraints': {
                'distance': {
                    'equals': 21325.0,
                    'units': 'ft',
                    'type': 'boundary',
                    'loc': 'final',
                    'ref': 30.0e3,
                },
                'flight_path_angle': {
                    'equals': 4.0,
                    'loc': 'final',
                    'units': 'deg',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(20.0e3, 1325.0), 'ft'],
            'mass': [(174.4e3, 174.3e3), 'lbm'],
            'mach': [(0.3, 0.3), 'unitless'],
            'altitude': [(1100.0, 1200.0), 'ft'],
            'time': [(85.0, 90.0), 's'],
        },
    },
    'EF_past_P1': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'clean': False,
            'time_initial_ref': (1.0e3, 'ft'),
            'time_initial_bounds': ((20.0e3, 50.0e3), 'ft'),
            'time_duration_ref': (1.0e3, 'ft'),
            'time_duration_bounds': ((100.0, 50.0e3), 'ft'),
            'mach_optimize': mach_optimize,
            'mach_polynomial_order': 1,
            'mach_bounds': ((0.24, 0.32), 'unitless'),
            'altitude_optimize': altitude_optimize,
            'altitude_polynomial_order': 1,
            'altitude_bounds': ((1.0e3, 3.0e3), 'ft'),
            'throttle_enforcement': 'boundary_constraint',
            'constraints': {
                'flight_path_angle': {
                    'equals': 4.0,
                    'loc': 'final',
                    'units': 'deg',
                    'type': 'boundary',
                },
                'distance': {
                    'equals': 30.0e3,
                    'units': 'ft',
                    'type': 'boundary',
                    'loc': 'final',
                    'ref': 30.0e3,
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(21325.0, 50.0e3), 'ft'],
            'mass': [(174.3e3, 174.2e3), 'lbm'],
            'mach': [(0.3, 0.3), 'unitless'],
            'altitude': [(1200.0, 2000.0), 'ft'],
            'time': [(90.0, 180.0), 's'],
        },
    },
    'post_mission': {
        'include_landing': False,
        'constrain_range': False,
    },
}


if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_solved2dof.csv', phase_info)

    prob.check_and_preprocess_inputs()

    prob.build_model()

    prob.add_driver(optimizer, max_iter=25)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective('mass')

    prob.setup()

    prob.run_aviary_problem(suppress_solver_print=True)

    try:
        loc = prob.get_outputs_dir()
        cr = om.CaseReader(f'{loc}/problem_history.db')
    except:
        cr = om.CaseReader('problem_history.db')

    cases = cr.get_cases('problem')
    case = cases[0]

    output_data = {}

    for point_name, phase_name in [['P1', 'EF_to_P1'], ['P2', 'CD_to_P2']]:
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
