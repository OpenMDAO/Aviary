# Reserve Mission Examples
The reserve mission examples are broken up into two sections: height-energy missions, and 2DOF missions. 
By defualt Aviary uses a height-energy formulation unless you have specified otherwise.
Single phase reserves for fixed-time and fixed-range will be presented as well as an example of a multiphase reserve mission.
An overview of how to formulate a reserve mission is presented in [Reserve Mission User Guide](../user_guide/reserve_missions.md). 

## Fixed Range Reserve

The simplest example of a reserve mission implementation can be seen in `run_reserve_mission_fixedrange.py`.
In this example we add a fixed range cruise to the `run_basic_aviary_example.py`.
This is performed by copying the cruise `phase_info` object from the basic example and adding `"reserve": True,` and `"target_distance": (200, 'km'),`.

The results of this study are shown in the `traj_results_report.html`.
The first thing we review is the altitude plot.
This plot clearly shows that a reserve cruise phase was created at the end of the regular mission.
There is no climb phase in that reserve; we jump to 32,000 ft and cruise for 200 km.

![Altitude](images/fixed_range_cruise_altitude.png "Altitude vs. Time for Fixed Range Cruise Reserve Example")

Additionally when looking at the mass and distance plots from the same study, we can see that the regular and reserve phases are connected properly.
There is no discontinuity between regular and reserve phases for either distance or mass.

![Distance](images/fixed_range_cruise_distance.png "Distance vs. Time for Fixed Range Cruise Reserve Example")
![Mass](images/fixed_range_cruise_mass.png "Mass vs. Time for Fixed Range Cruise Reserve Example")

## Fixed Time Reserve

An example of a fixed-time reserve mission implementation can be found in `run_reserve_mission_fixedtime.py`.
This example builds upon the basic flight defined in `run_basic_aviary_example.py`.
We make a copy of the cruise phase from `example_phase_info.py` and then modify it by adding `"reserve": True,` and `"target_duration": (30, 'min'),`.
We have also removed other references to time in the reserve phase definition.
We do not list `initial_guesses:times`, `user_options:duration_bounds`, or `user_options:fixed_duration`.
All of these are set automatically when we set `target_duration`.

## Multiphase Reserve Mission

The last example reserve mission includes five phases: climb, cruise for distance, cruise for time, cruise for distance, descend.
This example is contained in `run_reserve_mission_multiphase.py`.
This demonstrates how to create an arbitrarily ordered set of phases for reserve, including climb and descent phases.
All of these reserve phases are tagged with `"reserve": True,`.

Results from this mission can be seen in the figure below.
![Multiphase](images/multiphase_reserve.png "Distance vs. Time for Multiphase Reserve Example")
There are three distinct cruise segments and the added reserve climb and reserve descent show a more realistic flight profile for a diverted aircraft.
