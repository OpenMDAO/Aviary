# Reserve Mission

## Overview
Reserve missions are enabled for the following mission types:
* height_energy    (completed)
* 2ODF_collocation (in-progress)
* 2DOF_shooting    (future work)

Reserve missions can be used to create fixed-time or a fixed-distance phases that are appended to the last phase of your nominal mission. 
To do this, set the `"reserve": True,` flag on `phase_info.phase_name.user_options`. 
Additionally, if this is a fixed-time phase add `"target_duration` and if it's a fixed-distance phase add `"target_distance"`. 
Reserve phases must be added to `phase_info` after the regular phases. You are not allowed to intersperse regular and reserve. 

The reserve phase will start at the same range and mass as the last regular phases, but the altitudes are not automatically connected. 
Thus you can fly climb, cruise, descent for regular phases, and then immediatly jump to an altitude of 10 km for the reserve mission. 
Or if you wanted to make things more realistic attach a climb phase and then add your reserve cruise. Make sure both the reserve climb and the reserve cruise phases both have `reserve = True`.

You can chain together multiple reserve phases to make a complete reserve mission (i.e. climb to altitude, cruise for range, cruise for time, then descend). 
An example of this is shown in `run_reserve_mission_multiphase_time_and_range.py`. 
You cannot create a reserve mission that enforces time or range constraints over multiple phases (i.e specify the total range covered by a climb + cruise+ descent). 

## Examples
Examples of single-phase and multi-phase reserve missions are presented in [Reserve Mission Examples](../docs/examples/reserve_missions.md).

## Theory
When adding a reserve phase, `check_and_preprocess_inputs()` divides all the phases into two dictionaries, `regular_phases` which contain your normal flight, and `reserve_phases` which contains any phases with the `reserve = True` flag.

Only the final mission mass and range from `regular_phases` is automatically connected to the first point of the `reserve_phases`. 
Altitude, and other state variables are not automatically connected, allowing you to start the reserve mission at whatever altitude you want.

Setting optimize mach or altitude on regular or reserve phases will work, but it will only connect between their respective dictionaries. 
This is implemented in `_link_phases_helper_with_options()`. Optimize mach setting will not connect mach in a regular_phase to mach in a reserve_phase. 

It is essential that you run `check_and_preprocess_inputs()` after `prob.load_inputs()` to make sure that regular and reserve phases are separated via `phase_separator()`.

### Fuel Burn Calculations
Fuel burn during the regular mission (`Mission.Summary.FUEL_BURNED`) is calculated only based on `regular_phases`. 

Reserve fuel (`Mission.Design.RESERVE_FUEL`) is the sum of `Mission.Design.RESERVE_FUEL_ADDITIONAL`, `Mission.Design.RESERVE_FUEL_FRACTION`, and `Mission.Summary.RESERVE_FUEL_BURNED`. 
* `RESERVE_FUEL_ADDITIONAL` is a fixed value (i.e. 300kg)
* `RESERVE_FUEL_FRACTION` is based on a fraction of `Mission.Summary.FUEL_BURNED` 
* `RESERVE_FUEL_BURNED` is sum of fuel burn in all `reserve_phases`