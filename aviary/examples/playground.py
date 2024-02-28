import aviary.api as av


throttle_max = 1.0
throttle_climb = 0.956
throttle_cruise = 0.7
throttle_idle = 0.0

subsystem_options = {'core_aerodynamics':
                     {'method': 'low_speed',
                      'ground_altitude': 0.,  # units='ft'
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
                      'drag_coefficient_factor': .4}}

phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": False},
    'cruise': {
        'user_options': {
            'num_segments': 20,
            'order': 3,
            'fix_initial': True,
            'throttle_setting': throttle_max,
            'balance_throttle': True,
            'ground_roll': False,
            'clean': False,
            'initial_ref': (1.e3, 'ft'),
            'initial_bounds': ((0.e3, 0.e3), 'ft'),
            'duration_ref': (100., 'nmi'),
            'duration_bounds': ((1000., 1000.), 'nmi'),
            'mach_bounds': ((0.72, 0.8), 'unitless'),
            'altitude_bounds': ((32.e3, 36.e3), 'ft'),
            'control_order': 2,
            'optimize_mach': True,
            'optimize_altitude': True,
            'opt': True,
            'constraints': {
                'flight_path_angle': {
                    'equals': 0.,
                    'loc': 'final',
                    'units': 'deg',
                    'type': 'boundary',
                },
            }
        },
        'subsystem_options': subsystem_options,
        'initial_guesses': {
            'distance': [(0., 1000.), 'nmi'],
            'mach': [(0.72, 0.72), 'unitless'],
            'mass': [(175.e3, 174.e3), 'lbm'],
            'altitude': [(32.e3, 35.e3), 'ft'],
            # 'time': [(0., 1200.), 's'],
        },
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": False,
    },
}


driver = "SNOPT"

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

prob.add_driver(driver, max_iter=50)

prob.add_design_variables()

# Load optimization problem formulation
# Detail which variables the optimizer can control
prob.add_objective('mass')

prob.setup()

prob.set_initial_guesses()

# import openmdao.api as om
# om.n2(prob)

prob.run_aviary_problem()

# prob.check_partials(compact_print=True)

# prob.model.list_inputs(units=True, print_arrays=True)
# prob.model.list_outputs(units=True, print_arrays=True)
