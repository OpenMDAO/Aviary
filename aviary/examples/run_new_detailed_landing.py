import aviary.api as av


subsystem_options = {'core_aerodynamics':
                     {'method': 'low_speed',
                      'ground_altitude': 0.,  # units='ft'
                      'angles_of_attack': [
                          -5.0, -4.0, -3.0, -2.0, -1.0,
                          0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                          6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                          12.0, 13.0, 14.0, 15.0],  # units='deg'
                      'lift_coefficients': [
                          0.01, 0.1, 0.2, 0.3, 0.4,
                          0.5178, 0.6, 0.75, 0.85, 0.95, 1.05,
                          1.15, 1.25, 1.35, 1.5, 1.6, 1.7,
                          1.8, 1.85, 1.9, 1.95],
                      'drag_coefficients': [
                          0.04, 0.02, 0.01, 0.02, 0.04,
                          0.0674, 0.065, 0.065, 0.07, 0.072, 0.076,
                          0.084, 0.09, 0.10, 0.11, 0.12, 0.13,
                          0.15, 0.16, 0.18, 0.20],
                      'lift_coefficient_factor': 2.,
                      'drag_coefficient_factor': 2.}}
subsystem_options_landing = subsystem_options.copy()
subsystem_options_landing['core_aerodynamics']['drag_coefficient_factor'] = 3.

optimize_mach = False
optimize_altitude = False

phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": False},
    'GH': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'fix_initial': True,
            'ground_roll': False,
            'clean': False,
            'initial_ref': (1.e3, 'ft'),
            'initial_bounds': ((0., 16.e3), 'ft'),
            'duration_ref': (1.e3, 'ft'),
            'duration_bounds': ((500., 5.e3), 'ft'),
            'mach_bounds': ((0.1, 0.5), 'unitless'),
            'initial_mach': (0.15, 'unitless'),
            'final_mach': (0.15, 'unitless'),
            'initial_altitude': (500., 'ft'),
            'final_altitude': (394., 'ft'),
            'altitude_bounds': ((0., 1000.), 'ft'),
            'polynomial_control_order': 1,
            'throttle_enforcement': 'bounded',
            'optimize_mach': optimize_mach,
            'optimize_altitude': optimize_altitude,
            'rotation': False,
            'constraints': {
                'flight_path_angle': {
                    'equals': -3.,
                    'loc': 'initial',
                    'units': 'deg',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(0.e3, 2.e3), 'ft'],
            'time': [(0., 12.), 's'],
            'mass': [(120.e3, 119.8e3), 'lbm'],
        },
    },
    'HI': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'fix_initial': False,
            'ground_roll': False,
            'clean': False,
            'initial_ref': (1.e3, 'ft'),
            'initial_bounds': ((0., 16.e3), 'ft'),
            'duration_ref': (1.e3, 'ft'),
            'duration_bounds': ((500., 15.e3), 'ft'),
            'mach_bounds': ((0.1, 0.5), 'unitless'),
            'altitude_bounds': ((0., 1000.), 'ft'),
            'initial_mach': (0.15, 'unitless'),
            'final_mach': (0.15, 'unitless'),
            'initial_altitude': (394., 'ft'),
            'final_altitude': (50., 'ft'),
            'polynomial_control_order': 1,
            'throttle_enforcement': 'bounded',
            'optimize_mach': optimize_mach,
            'optimize_altitude': optimize_altitude,
            'rotation': False,
            'constraints': {
                'flight_path_angle': {
                    'equals': -3.,
                    'loc': 'final',
                    'units': 'deg',
                    'type': 'boundary',
                },
            },
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(2.e3, 6.5e3), 'ft'],
            'time': [(12., 50.), 's'],
            'mass': [(119.8e3, 119.7e3), 'lbm'],
        },
    },
    'IJ': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'fix_initial': False,
            'ground_roll': False,
            'clean': False,
            'initial_ref': (1.e3, 'ft'),
            'initial_bounds': ((0., 30.e3), 'ft'),
            'duration_ref': (1.e3, 'ft'),
            'duration_bounds': ((500., 15.e3), 'ft'),
            'mach_bounds': ((0.1, 0.5), 'unitless'),
            'altitude_bounds': ((0., 1000.), 'ft'),
            'initial_mach': (0.15, 'unitless'),
            'final_mach': (0.15, 'unitless'),
            'initial_altitude': (50., 'ft'),
            'final_altitude': (0., 'ft'),
            'polynomial_control_order': 2,
            'throttle_enforcement': 'path_constraint',
            'optimize_mach': False,
            'optimize_altitude': True,
            'rotation': False,
            'constraints': {
            },
        },
        'subsystem_options': subsystem_options_landing,
        'initial_guesses': {
            'distance': [(8.5e3, 2.e3), 'ft'],
            'time': [(50., 60.), 's'],
            'mass': [(119.7e3, 119.67e3), 'lbm'],
        },
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": False,
    },
}


prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/test_aircraft/aircraft_for_bench_solved2dof.csv', phase_info)

# Preprocess inputs
prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()

prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver("SLSQP", max_iter=100)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective('mass')

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()

# prob.model.list_inputs(units=True, print_arrays=True)
# prob.model.list_outputs(units=True, print_arrays=True)
