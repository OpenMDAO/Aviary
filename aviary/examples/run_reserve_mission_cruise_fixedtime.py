"""
This is a slightly more complex Aviary example of running a coupled aircraft design-mission optimization.
It runs the same mission as the `run_basic_aviary_example.py` script, but it uses the AviaryProblem class to set up the problem.
This exposes more options and flexibility to the user and uses the "Level 2" API within Aviary.

We define a `phase_info` object, which tells Aviary how to model the mission.
Here we have climb, cruise, and descent phases.
We then call the correct methods in order to set up and run an Aviary optimization problem.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""
import aviary.api as av


phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": True},
    "climb": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "reserve": False,
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.2, "unitless"),
            "final_mach": (0.72, "unitless"),
            "mach_bounds": ((0.18, 0.74), "unitless"),
            "initial_altitude": (0.0, "ft"),
            "final_altitude": (32000.0, "ft"),
            "altitude_bounds": ((0.0, 34000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": True,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 0.0), "min"),
            "duration_bounds": ((64.0, 192.0), "min"),
        },
        "initial_guesses": {"times": ([0, 128], "min")},
    },
    "cruise": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "reserve": False,
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.72, "unitless"),
            "final_mach": (0.72, "unitless"),
            "mach_bounds": ((0.7, 0.74), "unitless"),
            "initial_altitude": (32000.0, "ft"),
            "final_altitude": (34000.0, "ft"),
            "altitude_bounds": ((23000.0, 38000.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((64.0, 192.0), "min"),
            "duration_bounds": ((56.5, 169.5), "min"),
        },
        "initial_guesses": {"times": ([128, 113], "min")},
    },
    "descent": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "reserve": False,
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.72, "unitless"),
            "final_mach": (0.36, "unitless"),
            "mach_bounds": ((0.34, 0.74), "unitless"),
            "initial_altitude": (34000.0, "ft"),
            "final_altitude": (500.0, "ft"),
            "altitude_bounds": ((0.0, 38000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((120.5, 361.5), "min"),
            "duration_bounds": ((29.0, 87.0), "min"),
        },
        "initial_guesses": {"times": ([241, 58], "min")},
    },
    "post_mission": {
        "include_landing": False,
        "target_range": (1906, "nmi"),
    },
    "cruise_reserve": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "reserve": True,
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.72, "unitless"),
            "final_mach": (0.72, "unitless"),
            "mach_bounds": ((0.7, 0.74), "unitless"),
            "initial_altitude": (32000.0, "ft"),
            "final_altitude": (32000.0, "ft"),
            "altitude_bounds": ((23000.0, 38000.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": True,
            "initial_bounds": ((149.5, 448.5), "min"),
            "duration_bounds": ((60, 60), "min"),  # set this for fixed duration phases
        },
        "initial_guesses": {"times": ([60, 60], "min")},
    },
}


prob = av.AviaryProblem()

# Load aircraft and options data from user
# Allow for user overrides here
prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)


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
prob.add_objective()

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()
