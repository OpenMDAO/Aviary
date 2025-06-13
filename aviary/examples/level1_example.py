"""
This is a basic example of running a coupled aircraft design-mission optimization in Aviary using
the "level 1" API.

We use a pre-existing ".csv" file that describes the properties of a single-aisle commercial
transport and a pre-existing "phase_info" object that describes the mission to be flown.
This mission consists of climb, cruise, and descent phases.
The aircraft sizing problem is ran by calling the `run_aviary` function, which takes in the path to
the aircraft model, the phase info, and some other options.

This performs a coupled design-mission optimization and outputs the results from Aviary into the
`reports` folder.
"""

import aviary.api as av

prob = av.run_aviary(
    aircraft_data='models/N3CC/N3CC_FLOPS.csv',
    phase_info=av.default_height_energy_phase_info,
    optimizer='SLSQP',
    make_plots=True,
)
