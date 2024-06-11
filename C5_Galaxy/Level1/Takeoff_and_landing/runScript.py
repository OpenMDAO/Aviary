#Chris Psenica
#Aviary Test For A C5 Galaxy
#Level 1

#---------- Imports ----------
import aviary.api as av

#---------- Phase Info ----------
phase_info = {
    "pre_mission": {"include_takeoff": True, "optimize_mass": True},
    "climb_1": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 2,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.2, "unitless"),
            "final_mach": (0.77, "unitless"),
            "mach_bounds": ((0.18, 0.79), "unitless"),
            "initial_altitude": (0.0, "ft"),
            "final_altitude": (34000.0, "ft"),
            "altitude_bounds": ((0.0, 34500.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": True,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 0.0), "min"),
            "duration_bounds": ((48.0, 144.0), "min"),
        },
        "initial_guesses": {"time": ([0, 96], "min")},
    },
    "cruise_1": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 2,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.77, "unitless"),
            "final_mach": (0.77, "unitless"),
            "mach_bounds": ((0.75, 0.79), "unitless"),
            "initial_altitude": (34000.0, "ft"),
            "final_altitude": (34000.0, "ft"),
            "altitude_bounds": ((33500.0, 34500.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((48.0, 144.0), "min"),
            "duration_bounds": ((51.5, 154.5), "min"),
        },
        "initial_guesses": {"time": ([96, 103], "min")},
    },
    "descent_1": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "polynomial_control_order": 1,
            "use_polynomial_control": True,
            "num_segments": 2,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.77, "unitless"),
            "final_mach": (0.1, "unitless"),
            "mach_bounds": ((0.095, 0.79), "unitless"),
            "initial_altitude": (34000.0, "ft"),
            "final_altitude": (0.0, "ft"),
            "altitude_bounds": ((500.0, 34500.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((99.5, 298.5), "min"),
            "duration_bounds": ((51.0, 153.0), "min"),
        },
        "initial_guesses": {"time": ([199, 102], "min")},
    },
    "post_mission": {
        "include_landing": True,
        "constrain_range": True,
        "target_range": (2271, "nmi"),
    },
}

#---------- Run Aviary ----------
prob = av.run_aviary('define_C5.csv' , phase_info , optimizer = "SLSQP" , make_plots = True , max_iter = 400)

#---------- Assumptions/Approximations/Info ----------
'''
Engine: CF6-6 for engine mass data


'''

#---------- Mission ----------
'''
Polynomial control order: 1
Number of segments: 2
Duration: 300 minutes
Optimize Mach: None
Optimize Altitude: None

----------------------------------------------------
|| Pt. number ||   1   ||   2   ||   3   ||   4   ||
----------------------------------------------------
|| Altitude   ||   0   || 34000 || 34000 || 1000  ||
----------------------------------------------------
|| Mach       ||  0.2  ||  0.77 ||  0.77 ||  0.2  ||  
----------------------------------------------------

Exit: condition: Exit Mode 0
Error: N/A
Notes: An unusual amount of drag is being produced with this model. The drag has been reduced by 80% via scalars.

'''