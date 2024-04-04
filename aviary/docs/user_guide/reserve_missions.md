# Reserve Mission

## Overview

Reserve missions are enabled for the following mission types:

* height_energy    (completed)
* 2ODF (collocation) (complete)
* 2DOF (shooting)    (in-progress)
* height_energy (shooting)    (future work)

A reserve mission can be created by appending one or more reserve phases to `phase_info` after the last phase of the regular mission.
To create a simple reserve mission, copy your current cruise phase which is located in `phase_info`.

```{note}
You may need to revise some of your assumptions for the copied phase if you are making a reserve phase that is radically different than the original (i.e. original phase was to travel 3000km but reserve phase is 100km).
```

Append that phase to the end of `phase_info`, name it `reserve_cruise` and add `"reserve": True,` to `user_options` for this phase.
There are two optional flags that can now be added to `user_options`.
The `"target_duration"` option creates a phase requiring the aircraft to fly for a specific amount of time.
The `"target_distance"` option creates a phase requiring the aircraft to fly for a specific distance.
Avoid using the optional flag if you have a reserve phase (i.e climb or descent) where you just want that phase to be completed as quickly as possible.
The optional flags should not be combined as they will create overlapping constraints creating an infeasible problem.

You can chain together multiple reserve phases to make a complete reserve mission (i.e. climb to altitude, cruise for range, cruise for time, then descend).
Examples of this are shown in `run_reserve_mission_multiphase.py` and `run_2dof_reserve_mission_multiphase.py`.

The first reserve phase will start at the same range and mass as the last regular phase, but all other states (i.e. altitude, Mach number) are not automatically connected.
Thus you can fly climb, cruise, descent for regular phases and then immediately jump to an arbitrary altitude for the reserve mission.
Or if you wanted to make things more realistic you could attach a climb phase and then add your reserve cruise.
Make sure both the reserve climb and the reserve cruise phases both have `"reserve": True,` flag.

### Examples

Examples of single-phase and multi-phase reserve missions are presented in [Reserve Mission Examples](../examples/reserve_missions.md).

### Caveats when using 2DOF

If you are using 2DOF equations of motion (EOM) in your problem (i.e. `settings:equations_of_motion,2DOF`) there are some additional things you need to be aware of.
The name of the reserve phase should include one of the keywords to indicate which EOM from 2DOF will be selected and the prefix `reserve_`.
Valid keywords include: `accel`, `ascent`, `climb1`, `climb2`, `cruise`, `desc1`, `desc2`.
This is because 2DOF uses different EOMs for different phases and we need to let `methods_for_level2.py` know which method to select.
This is why in the example in the first paragraph above, the phase was named `reserve_cruise`.
Cruise phases can have additional information in suffixes, but this isn't necessary.
Do not worry about phase naming if you are using Height-Energy EOM as all those EOMs are the same for every phase.

## Theory

When adding a reserve phase, `check_and_preprocess_inputs()` divides all the phases into two dictionaries: `regular_phases` which contain your nominal phases and `reserve_phases` which contains any phases with the `reserve` flag set to `True`.
Additionally, `check_and_preprocess_inputs()` will add the `"analytic"` flag to each phase.
This is used to indicate if a phase is an analytic phase (i.e. Breguet range) or a ordinary differential equation (ODE).

Only the final mission mass and range from `regular_phases` are automatically connected to the first point of the `reserve_phases`.
All other state variables (i.e. altitude, mach) are not automatically connected, allowing you to start the reserve mission at whatever altitude you want.

The `"analytic"` flag helps to properly connect phases for 2DOF missions.
2DOF `cruise` missions are analytic because they use a Breguet range calculation instead of integrating an EOM. 
Analytic phases have a slightly different naming convention in order to access state/timeseries variables like distance, mass, and range compared with their non-analytic counterparts.

You cannot create a reserve mission that enforces time or range constraints over multiple phases (i.e specify the total range covered by a climb + cruise + descent).
This is because each constraint `"target_distance"` or `"target_time"` is only enforced on a single phase.

It is essential that you run `prob.check_and_preprocess_inputs()` after `prob.load_inputs()` to make sure that regular and reserve phases are separated via `phase_separator()`.

### Advanced Users and Target Duration Phases

For advanced users, instead of just copying a phase you used before, you might completely specify a new phase from scratch. 
When creating a `"target_duration"` reserve phase there are a number of values inside of `phase_info['user_options']` that are overwritten in `check_and_preprocess_inputs()`. 
Specifically, `duration_bounds`, `fixed_duration`, and `"initial_guesses": {"time"}` will be over-written. 
That is because if `"target_duration"` is specified, Aviary already knows what these other three values need to be: `target_duration = duration_bounds = "initial_guesses": {"time"}`, and `fixed_duration = True`.

### Fuel Burn Calculations

Fuel burn during the regular mission (`Mission.Summary.FUEL_BURNED`) is calculated only based on `regular_phases`.

Reserve fuel (`Mission.Design.RESERVE_FUEL`) is the sum of `Mission.Design.RESERVE_FUEL_ADDITIONAL`, `Mission.Design.RESERVE_FUEL_FRACTION`, and `Mission.Summary.RESERVE_FUEL_BURNED`.

* `RESERVE_FUEL_ADDITIONAL` is a fixed value (i.e. 300kg)
* `RESERVE_FUEL_FRACTION` is based on a fraction of `Mission.Summary.FUEL_BURNED`
* `RESERVE_FUEL_BURNED` is sum of fuel burn in all `reserve_phases`
