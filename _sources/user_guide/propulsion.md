# Propulsion

The propulsion subsystem in Aviary organizes and executes models for engine sizing and performance.

Aviary does not natively model gas-turbine or electric motor performance, and instead relies on user-provided data and/or custom performance models to perform propulsion analysis.

Aviary supports an arbitrary number of propulsor models on a vehicle, each with their own unique properties such as performance characteristics, scaling behaviors, and number of propulsors for that given type. 
<!-- A diagram would be helpful here, showing X propulsors of type A and Y propulsors of type B on an outline of an aircraft -->
Each unique type of engine is referred to as an engine model<!-- Link to wherever this class is described - theory guide? -->. In Aviary, an engine model contains information on how to size and estimate performance for a single instance of an engine of that type. During analysis, Aviary handles summing performance data to a system level. This way, information is available on the performance of both a single instance of an engine, as well as aircraft-level totals, for other Aviary subsystems to utilize.

## Engine Decks
<!--parts of this are probably theory?-->
The built-in way Aviary handles engine performance is by interpolating tabular data from a user-defined file that describes performance characteristics for a given engine. Engines modeled in this manner are called engine decks<!-- Link to wherever this class is described - theory guide? -->. Engine decks are a type of engine model - they use the same basic interface, but have additional functionality to handle reading and processing data files.

### Formatting
An engine deck data file requires specific formatting for Aviary to correctly interpret. These files must follow the [Aviary data file format](input_files).An example of a properly-formatted engine deck can be found [here](https://github.com/OpenMDAO/Aviary/blob/main/aviary/models/engines/turbofan_22k.deck).

### Variables

The following engine performance parameters are supported natively by Aviary for use in engine decks. If units are not specified, the default units are assumed. A variety of alternate names for these variables are understood by Aviary, but it is recommended to use the official names given here. A column with a header not recognized by Aviary will be ignored, with a warning raised at runtime. This allows for variables not used by Aviary to still be included in a data file, either for reference or compatibility with another analysis tool.

<!-- default variables and their units are not finalized -->
| Variable | Default Units | Required? |
| :--- | :--- | :---: |
| `Mach Number` | unitless | &#x2714; |
| `Altitude` | ft | &#x2714; |
| `Throttle` | unitless | &#x2714; |
| `Hybrid Throttle` | unitless | &#x2718; |
| `Net Thrust` | lbf | &#x2714;* |
| `Gross Thrust` | lbf | &#x2718; |
| `Ram Drag` | lbf | &#x2718; |
| `Fuel Flow Rate` | lbm/h | &#x2718; |
| `Electric Power` | kW | &#x2718; |
| `NOx Rate` | lbm/h | &#x2718; |
| `T4 Temperature` | degR | &#x2718; |

**`Net Thrust` (defined as `Gross Thrust` - `Ram Drag`) is not required if both of those variables are provided for calculation*

`Mach Number`, `Altitude`, and the two throttle parameters are independent variables required to describe the operating conditions of the engine. `Hybrid Throttle` is optional, and is intended for use as a second degree of control for engines using independently controllable fuel- and electric-based power. The remaining variables are dependent on the operating conditions and are therefore typically optional.

Engine decks without headers are assumed to contain only the required variable set, in the order specified by the table (`Mach`, `Altitude`, `Throttle`, and `Net Thrust`), and with default units.

Comments may be added to an engine deck data file by using a '`#`' symbol preceding the comment. Anything after this symbol on that line is ignored by Aviary, allowing the user to temporarily remove data points or add in-line comments with context for the data. It is good practice to include comments at the start of the file to explain what kind of engine the data represents, and where it came from.

## Setting Up Propulsion Analysis

### Beginner Guide

To add an engine deck to Aviary, the minimum set of variables to describe it must be provided in your input file. This list includes:
<!-- This list is for EngineDecks specifically, does that need clarification? No other engine model is currently planned to be supported with a level1 interface right? -->
* `Aircraft.Engine.DATA_FILE`
* `Aircraft.Engine.SCALE_PERFORMANCE`
* `Aircraft.Engine.IGNORE_NEGATIVE_THRUST`
* `Aircraft.Engine.GEOPOTENTIAL_ALT`
* `Aircraft.Engine.GENERATE_FLIGHT_IDLE`
* `Aircraft.Engine.NUM_WING_ENGINES` and/or `Aircraft.Engine.NUM_FUSELAGE_ENGINES`

<!--At a system level, there a few variables you must include that affect all engines. This list includes: -->

<!-- Additional variables are required but will depend on preprocessor behavior, which is not finalized. NUM_WING_ENGINES & NUM_FUSELAGE_ENGINES vs NUM_ENGINES behavior, SCALE_FACTOR vs. SCALED_SLS_THRUST, etc. -->

If generating flight idle points is desired, the following variables are also required. More information on flight idle generation is available here <!--TODO link here-->.

* `Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION`
* `Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION`
* `Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION`

If you are missing any required variables, you will see a warning at runtime. Aviary will try using the default value for the missing variable, which may affect analysis results.

```bash
UserWarning: <aircraft:engine:scale_performance> is a required option for EngineDecks, but has not been specified for EngineDeck <example>. The default value will be used.
```

<!-- See !!!LINK HERE!!! for a complete list of all available options to define engine behavior -->

<!-- Section on setting up Propulsion-level variables, which ones are required? -->

### Intermediate Guide

Engine models are defined in Aviary using an `EngineModel` object. An `EngineModel` is responsible for handling many tasks required to prepare an engine for use in Aviary, such as reading engine data from a file in the case of an `EngineDeck` (which is a child class of `EngineModel`). <!-- link to theory guide? -->

An `EngineModel` (and classes inheriting it) can be manually created and added to the Aviary problem. This is extremely useful when setting up an aircraft with multiple engine types, each with unique properties, or using a custom engine model. An `EngineModel` requires an `AviaryValues` object containing the variables required for that engine (such as those outlined in the Beginner Guide example for `EngineDecks`)<!-- link to that subsection -->.

<!-- enforce uniform code style across documentation -->
```python
import aviary.api as av

engine_options = av.AviaryValues()
# Add relevant inputs and options to engine_options
# using engine_options.set_val(...)

engine_model = av.EngineModel(name='example',
                           options=engine_options)
```

Once an `EngineModel` has been created, it must be added to the Aviary analysis you want to perform. The simplest way to do this is to take advantage of the propulsion preprocessor utility. This preprocessor handles all of the details of getting data related to `EngineModels`, which may change during initialization, correctly set up in the `AviaryValues` object which is used to define the vehicle at the Aviary problem level.

```python
aviary_options = av.AviaryValues()
# It is assumed here that aviary_options is configured to have
# all inputs needed for analysis, except engine-level values

av.preprocess_propulsion(aviary_options=aviary_options,
                         engine_models=[engine_model])
```

In this example, *aviary_options* is modified in-place with updated values from *engine_model*, as well as properly configuring engine-related variables into vectors. When working with multiple engines, simply provide `preprocess_propulsion()` with a list of all `EngineModels`, like so:

```python
av.preprocess_propulsion(aviary_options=aviary_options,
                         engine_models=[engine_model_1,
                                        engine_model_2])
```

The propulsion preprocessor is also capable of creating an `EngineDeck` based on existing inputs in the provided `AviaryValues` object and add it to the analysis, which is what is done behind-the-scenes in the Level 1 interface. To use the preprocessor in this way, simply do not provide an `EngineModel` to the function:

```python
av.preprocess_propulsion(aviary_options=aviary_options)
```

### Advanced Guide

This section is a work in progress. Please check back later for more information.
