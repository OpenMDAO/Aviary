phase_info = {
    "pre_mission": {"include_takeoff": False, "optimize_mass": True},
    "post_mission": {
        "include_landing": False,
        "constrain_range": True,
        "target_range": (0.0, "nmi"),
    },
}
