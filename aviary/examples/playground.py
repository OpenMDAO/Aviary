import aviary.api as av


throttle_max = 1.0
throttle_climb = 0.956
throttle_cruise = 0.930
throttle_idle = 0.0


subsystem_options = {'core_aerodynamics':
                     {'method': 'low_speed',
                      'ground_altitude': 0.,  # units='m'
                      'angles_of_attack': [
                          0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                          6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                          12.0, 13.0, 14.0, 15.0],  # units='deg'
                      'lift_coefficients': [
                          0.5178, 0.6, 0.75, 0.85, 0.95, 1.05,
                          1.15, 1.25, 1.35, 1.5, 1.6, 1.7,
                          1.8, 1.85, 1.9, 1.95],
                      'drag_coefficients': [
                          0.0674, 0.065, 0.065, 0.07, 0.072, 0.076,
                          0.084, 0.09, 0.10, 0.11, 0.12, 0.13,
                          0.15, 0.16, 0.18, 0.20],
                      'lift_coefficient_factor': 1.,
                      'drag_coefficient_factor': 1.}}

phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": True},
    'groundroll': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'fix_initial': True,
            'throttle_setting': throttle_max,
            'input_speed_type': av.SpeedType.TAS,
            'ground_roll': True,
            'clean': False,
            'initial_ref': (1.e3, 'm'),
            'initial_bounds': ((0., 0.), 'm'),
            'duration_ref': (1.e3, 'm'),
            'duration_bounds': ((50., 2000.), 'm'),
            'control_order': 1,
            'opt': True,
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'times': [(0., 1000.), 'm'],
            'TAS': [(1., 100.), 'kn'],
            'mass': [(175.e3, 175.e3), 'lbm'],
            # 'altitude': [(0., 0.), 'ft'],
        },
    },
    'rotation': {
        'user_options': {
            'num_segments': 5,
            'order': 3,
            'fix_initial': False,
            'throttle_setting': throttle_max,
            'input_speed_type': av.SpeedType.TAS,
            'ground_roll': True,
            'clean': False,
            'initial_ref': (1.e3, 'm'),
            'initial_bounds': ((1., 500.), 'm'),
            'duration_ref': (1.e3, 'm'),
            'duration_bounds': ((50., 2000.), 'm'),
            'control_order': 1,
            'opt': True,
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'times': [(50., 1000.), 'm'],
            'TAS': [(1., 100.), 'kn'],
            'mass': [(175.e3, 175.e3), 'lbm'],
            # 'altitude': [(0., 0.), 'ft'],
        },
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": False,
        "target_range": (1906, "nmi"),
    },
}


prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/test_aircraft/playground.csv', phase_info)


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
