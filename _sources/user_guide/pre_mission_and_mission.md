# Pre-Mission and Mission

Within an Aviary model there are two main types of systems: pre-mission and mission.

Pre-mission systems are those that are run before the mission analysis and do not vary throughout the mission.
Examples include aircraft geometry, masses of aircraft components, any pre-computed quantities needed for mission systems.
Any quantities within the pre-mission systems are assumed to be constant throughout the mission.

Mission systems are those that are run during the mission analysis and may vary throughout the mission.
Examples include aerodynamics, propulsion, current mass and velocity of the aircraft, etc.
Any quantities within the mission systems are allowed to vary throughout the mission.

A nominal diagram showing the pre-mission and mission systems is shown below.

```{note}
In other works the pre-mission systems are sometimes called "static" and the mission systems are called "dynamic".
Within Aviary we avoid the terms "static" and "dynamic" because they are confusing due to their use in other contexts, such as structural dynamics.
```

![Pre-mission vs mission](images/pre_mission_and_mission.svg)

## Pre-Mission Systems

Pre-mission systems are run before the mission analysis and are assumed to be constant throughout the mission.
The values in the pre-mission systems _can_ vary during the optimization process.
For example, gross takeoff weight (GTOW) is often a design variable in an aircraft optimization problem.
GTOW does not vary across the mission but it does vary during the optimization process.

## Mission Systems

Systems within the mission group of Aviary are the systems that vary during the aircraft's flight trajectory.
This means that the systems are evaluated at each analysis point within the mission analysis.
For example, the aerodynamics subsystem is evaluated at each analysis point to determine the aerodynamic forces and moments acting on the aircraft at that point.
The propulsion subsystem is evaluated at each point to determine the thrust and fuel flow of the propulsion system.

Systems within the mission group are often vectorized.
This is possible because the systems are evaluated at each analysis point independently of the other analysis points when using {term}`collocation integration methods`.
Within Aviary, the number of mission analysis points is called `num_nodes`.

## States, Controls, and Parameters

States, controls, and parameters are the three main types of variables within Aviary that are relevant to the mission analysis.
States are variables that are integrated over the mission analysis.
Controls are variables that are manipulated by the optimizer and are allowed to vary across the mission.
Parameters are variables that are allowed to be controlled by the optimizer (but don't have to be) and are assumed to be constant across the entire trajectory or a single phase, depending on how they're set up.

## The Bus System in Aviary

Within Aviary, you might want to connect a pre-mission system to a mission system.
Variables that begin with `'aircraft:'` are connected when you use the `get_parameters()` method within `SubsystemBuilderBase`.
However, you might want to connect a variable from a pre-mission system to a mission system that does not begin with `'aircraft:'`.
For example, you might have a subsystem that has some computations in the pre-mission system that you want to connect to the mission system, but you don't necessarily want to expose those variables to the rest of Aviary.
The bus system is also useful if you have variables that begin with `'aircraft:'` but you don't want them exposed to the rest of Aviary.

To do this, you can use the "bus" system.
The bus system allows you to connect variables from the pre-mission system to the mission system based on what you specify.
This is especially relevant when you're using external subsystems as core Aviary does not use the bus system internally.
The notion of the bus system is detailed more within the `SubsystemBuilderBase` docstrings.

## Post-Mission Systems

Post-mission systems are run after the pre-mission and mission analyses, as expected by the name.
These systems are used to compute any post-mission quantities that are needed, such as landing-related properties, economic models, mission postprocessing, etc.
These systems can use any of the outputs from the pre-mission and mission systems.
