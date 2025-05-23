"""
This is a straightforward and basic example of running a coupled aircraft design-mission optimization in Aviary.
This uses the "level 1" API within Aviary.

We use the pre-defined single aisle commercial transport aircraft definition and use a pre-defined phase_info object
to describe the mission optimization problem to Aviary.
This mission consists of climb, cruise, and descent phases.
We then call the `run_aviary` function, which takes in the path to the aircraft model, the phase info, and some other options.
This performs a coupled design-mission optimization and outputs the results from Aviary into the `reports` folder.
"""

import aviary.api as av

prob = av.run_aviary(
    'models/test_aircraft/aircraft_for_bench_FwFm.csv',
    av.default_height_energy_phase_info,
    optimizer='SLSQP',
    make_plots=True,
)
