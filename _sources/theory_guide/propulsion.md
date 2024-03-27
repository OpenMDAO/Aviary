# Core Propulsion Subsystem

Propulsion is unique among the core Aviary subsystems for its level of flexibility and compatibility with custom and external code. Aviary supports an arbitrary number of engines on a vehicle which can be modeled using the built-in method (engine decks) alongside any user-provided models.

Aviary propulsion can be thought of as having two levels: system-level propulsion and individual engines. The top-level propulsion subsystem is model-agnostic and exists to organize engines and sum relevant outputs into system-level totals that other subsystems can use. Modeling of each type of engine on the aircraft is handled with `EngineModels`.

This page details how the core propulsion subsystem works when using `EngineDecks`. External propulsion subsystems (when wrapped by an `EngineModel`) are treated the same way by Aviary, but the details of what is computed and where will differ by model.

## Preprocessing

Before analysis is performed, Aviary's propulsion preprocessor function is used to ensure input consistency. These consistency checks serve two functions.
First, the inputs and options used to initialize individual `EngineModels` and the vehicle-level variable hierarchy need to match. The vehicle-level hierarchy's engine-related inputs must be vectorized to include information for all individual engines being modeled.
Second, a select number of inputs are checked for physical consistency with each other. For example, the sum of wing- and fuselage-mounted engines must match the total number of engines on the vehicle.

Because engine-related variables are expected to be vectorized with a value for each unique `EngineModel`, if a variable is defined for one engine, it must also be defined for all others so a vector of consistent size can be created. The propulsion preprocessor handles this automatically. Values of variables defined in an `EngineModel` are given highest priority and always used if available. Otherwise, if a value for that variable is present in the vehicle-level variable hierarchy, it is treated as a 'user-provided default' value to assign to any engine that doesn't explicitly define it. Finally, if neither of these values can be found, the default value pulled from Aviary variable metadata is used.

`EngineModels` objects have input preprocessing steps performed during initialization to handle internal consistency of inputs and options within that individual engine.

## Pre-Mission Analysis

Aviary propulsion organizes all pre-mission analysis into a propulsion group that is added to the 'pre_mission_analysis' group. Pre-mission propulsion calls the pre-mission builders for each provided `EngineModel`, adding each created subsystem to the propulsion group. Aviary also includes a component which calculates system-level SLS (sea-level static) thrust and another component which calculates distributed propulsion factors based on total thrust and total number of engines. These distributed factors are used by some Aviary core mass estimation components.

The `EngineDeck` pre-mission builder includes a `EngineScaling` component that calculates a performance scaling factor based on the percent difference in desired target thrust versus the unscaled SLS thrust present in the data file.

## Mission Analysis

Similar to pre-mission, during mission analysis Aviary propulsion creates a group that calls the mission builder for each `EngineModel`. Engine performance data is [muxed](https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/mux_comp.html) together and then summed into vehicle-level totals using an additional component. Both single engine and vehicle-level propulsion performance is promoted within the propulsion group although only vehicle-level totals are used by other Aviary core subsystems.

`EngineDeck` mission builders produce a more complicated group that interpolates and scales performance values based on flight condition. To provide maximum thrust conditions (as needed by the [energy-state approximation](energy-method)), a duplicate set of interpolation components are created and run at max throttle setting to always produce max thrust for a given flight condition. This performance data is then scaled with an additional component. Only the scaled data is exposed to the greater propulsion group. Unscaled engine data is not promoted outside that `EngineDeck's` mission group, and is therefore generally unavailable to other Aviary components.

## Post-Mission Analysis

Aviary currently does not support post-mission propulsion analysis. A future update will include a propulsion-level group that iteratively calls `EngineModel` post-mission builders similar to how pre-mission is performed.

## External Subsystems
Because the actual computation of propulsion-related variables is done within `EngineModels`, in most cases the propulsion subsystem builder does not need to be replaced when adding new propulsion analysis to Aviary.