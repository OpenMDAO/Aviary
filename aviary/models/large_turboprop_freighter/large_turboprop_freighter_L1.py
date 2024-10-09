import aviary.api as av
from aviary.models.large_turboprop_freighter.phase_info import two_dof_phase_info

# aviary run_mission large_turboprop_freighter.csv --phase_info phase_info.py
# python3 large_turboprop_freighter_L1.py

av.run_aviary(
    aircraft_filename='large_turboprop_freighter.csv',
    phase_info=two_dof_phase_info,
)
