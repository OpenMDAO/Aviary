"""
This is a basic example of running a coupled aircraft design-mission optimization in Aviary using
the "level 1" API.

The properties of the aircraft are defined in a pre-existing ".csv" file - in this case it describes
a conventional single-aisle commercial transport. The mission is defined using a "phase_info" file,
which consists of a climb, cruise, and descent phase.

The aircraft sizing problem is ran by calling the `run_aviary` function, which takes in the path to
the aircraft model, the phase info, and some other optional settings. This performs a coupled
design-mission optimization.
"""

import aviary.api as av

prob = av.run_aviary(
    aircraft_data='models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
    phase_info='examples/example_phase_info.py',
    optimizer='SLSQP',
    make_plots=True,
)
