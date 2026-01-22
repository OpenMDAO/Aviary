# Getting Started
Aviary is an open-source conceptual aircraft design tool developed by NASA that allows users to:

- design aircraft and evaluate their mission performance
- perform gradient-based optimization for both the aircraft design and trajectory

Aviary can also be used as a framework, where users can bring their own tools and analysis methods and integrate them into the aircraft design process.
A key goal of Aviary is flexibility and extensibility.
Aviary has a large number of options that allow users to change how their aircraft is modeled and what kind of problem is being solved.

## Core Features

### Aircraft Design
Aviary is, at its core, a tool that executes the classic aircraft design problem (often referred to as "sizing" an aircraft).
This iterative approach to aircraft design is outlined in many popular textbooks such as those written by Raymer, Anderson, and Roskam. The way that Aviary does this under-the-hood is slightly more complicated, utilizing numerical methods and/or optimization to find a aircraft design that "closes", but the basic goal is the same.
In short, aircraft sizing helps aircraft designers find the value of "unknown" variables, such as the aircraft's gross weight, given certain design requirements, such as being able to fly a specific mission.
Aviary's implementation is as generalized as possible to generic vehicle design, but is biased towards aircraft through variable names, use of some industry jargon, and the included methods for estimating vehicle performance being aircraft-specific.
A knowledgeable user can use Aviary as a bare framework, replacing all aircraft-specific features with custom ones to model whatever kind of system they can think of.

As a standalone tool, Aviary is able to perform low-fidelity, conceptual aircraft design.
Users have a broad amount of flexibility to define their aircraft design problem, especially when using optimization.
What kind of aircraft you are modeling (regional jet, general aviation, blended-wing-body, etc.), what mission it is flying (specifying phases such as takeoff, climb, cruise, descent, landing, reserve), and what your goal is (such as minimize energy consumption, maximize range) are all highly configurable with Aviary.

As a framework, Aviary is able to integrate external tools and methods into the design loop. We refer to these user-provided models as "external subsystems".
This is extremely powerful and can help designers capture the effects of interacting (or "coupled") systems during optimization.
Please see the [external subsystems doc pages](./user_guide/subsystems) for more information on how to add your own subsystems to an Aviary model.


### Included Methods

Aviary includes a core suite of analysis models that are needed for aircraft design.
They are referred to as "subsystems".
Each subsystem is an independent piece of analysis that models a specific aspect of an aircraft or its performance.
While they are intended to be used inside Aviary, they may be useful in other contexts.
The basic subsystems included with Aviary are:

- atmosphere
- aerodynamics
- geometry
- mass
- propulsion
- energy
- performance
- mission analysis

For each of these subsystems, Aviary provides a few different methods that can be used.
In general, these are empirical models based on historical data, raw tabular data from other models or experiments, and some physics-based models.

The [Theory Guide](./theory_guide/intro) provides much more information on the specific methods used for each subsystem.

### Optimization
Aviary excels at designing aircraft through optimization.
Aircraft sizing problems are solved via gradient-based optimization, allowing for the design of the aircraft and trajectory analysis to be solved simultaneously.
For unconventional aircraft designs, this makes Aviary extremely useful for fully accounting for the effects of coupled systems.

Aviary's [Installation Guide](./getting_started/installation) provides instructions on how to get access to several different gradient-based optimizers that can be used with the tool.