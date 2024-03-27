# Features and Functionalities

## Subsystems

At a minimum, an Aviary model needs to include the following subsystems:

- Geometry
- Mass
- Aerodynamics
- Propulsion
- Mission

Basically, these subsystems provide all the necessary forces acting on the aircraft which allows us to evaluate the performance of the aircraft across its mission.
Aviary does not explicitly support models with only some of these subsystems, but you are welcome to use parts of Aviary's code to build your own model.

The following sections will discuss each of these subsystems in more detail.

## Basic Assumptions

Because Aviary is a tool for conceptual design, we make a number of assumptions about the aircraft and mission that allow us to simplify the model.
These are often baked in to the underlying subsystem models, but it's important to understand what these assumptions are so that you can make sure they're appropriate for your problem.

For example, Aviary's computed aerodynamic models are only valid for subsonic flight.
Aviary itself is not limited to subsonic flight, but the aerodynamic models are.
So, you could use Aviary to model a supersonic aircraft, but you would need to provide your own aerodynamic model.

Other assumptions are largely dependent on which subsystems you include, the aircraft you're designing, and the mission you're evaluating.

<!-- TODO: add more detail here -->

## Mission Optimization

We'll now discuss mission optimization from a user's perspective.
What we mean by that is that we'll not go into details on the theory or math behind mission optimization, but instead focus on what it means for you to set up a mission optimization problem in Aviary.
We'll start by discussing the different phases of a mission that you can model in Aviary, and then we'll discuss how to set up constraints for your mission optimization problem.
Additionally, we'll dig into just how much flexibility you should give the optimizer when defining your mission.

### Basic definition of a mission

Throughout Aviary we use a series of terms when discussing mission optimization.

A "trajectory" is the full mission that the aircraft flies.
Usually this is from takeoff to landing, inclusive.
Sometimes you might want to model just a portion of the full aircraft trajectory; for example only the cruise portion.

A "phase" is a part of the trajectory that is defined by a single set of differential equations.
For example, a simple way of defining a full trajectory is to have climb, cruise, and descent phases.
Each of these phases can have different physics, subsystems, controls, and constraints.
For example, the climb phase for a hybrid-electric aircraft might be have electric motor assistance whereas that might not be needed for the cruise phase.

A "segment" is a sub-portion of a phase that is mostly used internally or when discussing the math behind the problem.
Users that are not defining custom phases will likely never need to worry about segments.

### Defining a mission

A mission is defined by a series of phases that the user chooses by specifying options in the `phase_info` dictionary.
The `phase_info` dictionary is a dictionary of dictionaries, where each key is the name of a phase and the value is a dictionary of options for that phase.

How you choose to define your phases is dependent on the aircraft you're modeling, the mission you're trying to evaluate, and the flexibility you want to give the optimizer.
For example, if you have a relatively conventional aircraft that is flying a straightforward mission, you might just need three phases: climb, cruise, and descent.
However, if you have a more complex aircraft or mission, you might need to define more phases.
For instance, if you're modeling a hybrid-electric aircraft with non-conventional propulsion systems that are controlled in different ways, you might want to define additional phases and prescribe different options based on which physics you want included at different stages in the flight.

In general, if you're familiar with the legacy tools FLOPS or GASP, you can use the corresponding default `phase_info` objects to start defining your mission.
FLOPS-based missions have three integrated phases: climb, cruise, and descent, as well as analytic takeoff and landing systems.
GASP-based missions have at least nine integrated phases: groundroll, rotation, ascent, accel, climb1, climb2, cruise, desc1, and desc2, as well landing systems.
GASP-based missions that are solved using SGM have additional phases.
The difference in the number of phases is due to the fact that GASP had more detailed requirements on the flight profile, especially in the early phases of a mission.

You can import a copy of the default `phase_info` dicts and then modify them as you need to for your own mission definition.

### Defining mission controls and constraints

How you choose to define your mission constraints depends on the aircraft your modeling, the equations of motion used, and which subsystems you're including in your model.
For example, if you're modeling a single-aisle commercial transport aircraft that will fly a relatively conventional mission, you might define your mission so that the aircraft can only climb in the first phase, cruises at a fixed altitude and Mach number, then descends in the final phase.
This would mimic the actual flight profile of this aircraft to a reasonable degree.

However, if you're modeling an urban air mobility aircraft that will fly a more complex mission, you might want to give the optimizer more flexibility in how it flies the mission.
Purposefully giving the optimizer the freedom to explore the trajectory design space at the same time it's designing the aircraft is a perfect example use case for Aviary.
This will result in a more complex optimization problem that might not converge well without some expert knowledge of the problem, but it will allow you to explore the design space more fully.

## Collocation and Shooting

Both [collocation](https://openmdao.org/dymos/docs/latest/getting_started/collocation.html) and shooting methods are included in Aviary for mission analysis as they each have something to offer.
Collocation methods are easily parallelizable and call the model ODE relatively few times.
This leads to significantly faster optimization times for large problems.

```{note}
Please see [this Dymos doc page](https://openmdao.org/dymos/docs/latest/getting_started/transcriptions.html#differences-between-collocation-and-explicit-shooting) for a better understanding of the similarities and differences between shooting and collocation methods.
```

Shooting (or Forward in Time Integration) methods provide physically valid trajectories at all iterations of the optimization. This means that even if an optimization fails to converge, the results are still physical and can be useful for debugging.
[This journal paper](https://link.springer.com/article/10.1007/s10957-023-02303-3) contains more information about the shooting method used in Aviary.

While collocation methods require a reasonably accurate estimation of the trajectory to be able to converge, shooting methods only require the initial state. This makes analyzing a new aircraft or mission easier for analysts as they do not need to produce accurate initial guesses.
One of the main advantages of shooting methods is the ability to dynamically order phases based on events. This means that different constraints, controls, or ODEs can be used depending on conditions during the trajectory. For example, the drag calculations change depending on aircraft configuration during takeoff; if the flaps are retracted when the aircraft reaches a certain speed, but the gear is retracted based on altitude, the two events could occur in either order.

Collocation results are presented as part of a fixed step-size timeseries. To improve performance, the shooting method uses an adaptive step size; this means that the resulting trajectories will not always have a consistent number of points. There are plans to add a post processing interpolation to the trajectory results to produce a consistent timeseries, but that has not been implemented yet.

```{note}
When using Aviary, the `AnalysisScheme` option is used to select the integration method. The default is `COLLOCATION`, but this can be changed to `SHOOTING` to use the shooting method.
```
