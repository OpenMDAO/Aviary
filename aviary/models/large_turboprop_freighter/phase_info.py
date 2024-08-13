from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Dynamic, Mission
from aviary.variable_info.enums import LegacyCode

GASP = LegacyCode.GASP

prop = CorePropulsionBuilder('core_propulsion', BaseMetaData)
mass = CoreMassBuilder('core_mass', BaseMetaData, GASP)
aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, GASP)
geom = CoreGeometryBuilder('core_geometry', BaseMetaData, GASP)

default_premission_subsystems = [prop, geom, aero, mass]
default_mission_subsystems = [aero, prop]


phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": True},
    "climb": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.1, "unitless"),
            "final_mach": (0.475, "unitless"),
            "mach_bounds": ((0.05, 0.48), "unitless"),
            "initial_altitude": (0.0, "ft"),
            "final_altitude": (21_000.0, "ft"),
            "altitude_bounds": ((0.0, 33_000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": True,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((0.0, 0.0), "min"),
            "duration_bounds": ((15.0, 192.0), "min"),
            "add_initial_mass_constraint": False,
        },
    },
    "cruise": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.475, "unitless"),
            "final_mach": (0.72, "unitless"),
            "mach_bounds": ((0.7, 0.74), "unitless"),
            "initial_altitude": (21_000.0, "ft"),
            "final_altitude": (21_000.0, "ft"),
            "altitude_bounds": ((21_000.0, 21_000.0), "ft"),
            "throttle_enforcement": "boundary_constraint",
            "fix_initial": False,
            "constrain_final": False,
            "fix_duration": False,
            "initial_bounds": ((30.0, 192.0), "min"),
            "duration_bounds": ((30.5, 169.5), "min"),
        },
    },
    "descent": {
        "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
        "user_options": {
            "optimize_mach": False,
            "optimize_altitude": False,
            "num_segments": 5,
            "order": 3,
            "solve_for_distance": False,
            "initial_mach": (0.72, "unitless"),
            "final_mach": (0.36, "unitless"),
            "mach_bounds": ((0.34, 0.74), "unitless"),
            "initial_altitude": (21_000.0, "ft"),
            "final_altitude": (0.0, "ft"),
            "altitude_bounds": ((0.0, 33_000.0), "ft"),
            "throttle_enforcement": "path_constraint",
            "fix_initial": False,
            "constrain_final": True,
            "fix_duration": False,
            "initial_bounds": ((120.5, 361.5), "min"),
            "duration_bounds": ((15.0, 87.0), "min"),
        },
    },
    "post_mission": {
        "include_landing": False,
        "constrain_range": True,
        "target_range": (1906.0, "nmi"),
    },
}
