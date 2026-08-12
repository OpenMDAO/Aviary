"""Energy-state mission definition for the minimum single-aisle example."""

from copy import deepcopy

from aviary.models.missions.energy_state_default import phase_info as default_phase_info


# Keep a local copy so examples can safely modify this dictionary without changing the shared default.
phase_info = deepcopy(default_phase_info)
