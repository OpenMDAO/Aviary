# Reserve Mission

## Overview

Reserve missions are enabled for the following mission types:

* height_energy    (completed)
* 2ODF (collocation) (in-progress)
* 2DOF (shooting)    (future work)

Reserve missions can be used to create fixed-time or a fixed-distance phases that are appended to the last phase of your nominal mission.
To do this, set the `"reserve": True,` flag on within the `user_options` for your reserve phases within `phase_info`.
Additionally, if this is a fixed-time phase add `"target_duration"` and if it's a fixed-distance phase add `"target_distance"`.
Reserve phases must be added to `phase_info` after the regular phases.
You are not allowed to intersperse regular and reserve phases.

The reserve phases will start at the same range and mass as the last regular phases, but the altitudes are not automatically connected.
Thus you can fly climb, cruise, descent for regular phases and then immediately jump to an arbitrary altitude for the reserve mission.
Or if you wanted to make things more realistic you could attach a climb phase and then add your reserve cruise.
Make sure both the reserve climb and the reserve cruise phases both have the `reserve` flag within `phase_info` set to `True`.

You can chain together multiple reserve phases to make a complete reserve mission (i.e. climb to altitude, cruise for range, cruise for time, then descend).
An example of this is shown in the example file `run_reserve_mission_multiphase_time_and_range.py`.
You cannot create a reserve mission that enforces time or range constraints over multiple phases (i.e specify the total range covered by a climb + cruise + descent).

## Examples

Examples of single-phase and multi-phase reserve missions are presented in [Reserve Mission Examples](../examples/reserve_missions.md).

## Theory

When adding a reserve phase, `check_and_preprocess_inputs()` divides all the phases into two dictionaries: `regular_phases` which contain your nominal phases and `reserve_phases` which contains any phases with the `reserve` flag set to `True`.

Only the final mission mass and range from `regular_phases` is automatically connected to the first point of the `reserve_phases`.
Altitude and other state variables are not automatically connected, allowing you to start the reserve mission at whatever altitude you want.

Setting `optimize_mach` or `optimize_altitude` to `True` on regular or reserve phases will work.
To be clear, the Mach and altitude values (even if set to be optimized) will not be connected between regular and reserve phases.

It is essential that you run `prob.check_and_preprocess_inputs()` after `prob.load_inputs()` to make sure that regular and reserve phases are separated via `phase_separator()`.

### Fuel Burn Calculations

Fuel burn during the regular mission (`Mission.Summary.FUEL_BURNED`) is calculated only based on `regular_phases`.

Reserve fuel (`Mission.Design.RESERVE_FUEL`) is the sum of `Mission.Design.RESERVE_FUEL_ADDITIONAL`, `Mission.Design.RESERVE_FUEL_FRACTION`, and `Mission.Summary.RESERVE_FUEL_BURNED`.

* `RESERVE_FUEL_ADDITIONAL` is a fixed value (i.e. 300kg)
* `RESERVE_FUEL_FRACTION` is based on a fraction of `Mission.Summary.FUEL_BURNED`
* `RESERVE_FUEL_BURNED` is sum of fuel burn in all `reserve_phases`
