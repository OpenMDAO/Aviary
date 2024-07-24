"""
Curated list of aviary inputs that are promoted as parameters in the mission.
"""
from aviary.variable_info.variables import Aircraft, Mission


core_mission_inputs = [
    Aircraft.Wing.INCIDENCE,
    Mission.Takeoff.FINAL_MASS,
]
