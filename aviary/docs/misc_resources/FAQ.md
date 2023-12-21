# Frequently Asked Questions

## What are the goals and objectives of Aviary?

Aviary is a preliminary design tool for aircraft capable of modeling traditional and novel designs.
This tool is built on top of OpenMDAO and is open-source.
The tool incorporates a number of low-fidelity subsystems and provides streamlined capability for users to integrate their own higher fidelity subsystems or subsystems of additional disciplines.

## What are the capabilities of Aviary?

Aviary is an open-source preliminary aircraft design, analysis, and optimization tool capable of performing low-fidelity design, analysis, and optimization of traditional and novel aircraft configurations.
This includes subsystem analysis for aerodynamics, propulsion, weights, trajectory, and geometry.
External subsystems like [NPSS](https://www.swri.org/consortia/numerical-propulsion-system-simulation-npss), [pyCycle](https://github.com/OpenMDAO/pyCycle), [OpenAeroStruct](https://github.com/mdolab/OpenAeroStruct/), [VSPAero](https://openvsp.org/wiki/doku.php?id=vspaerotutorial), [ANOPP](https://software.nasa.gov/software/LAR-19861-1), electrical modeling, etc., can be connected to Aviary for medium- and high-fidelity analysis and optimization.
External subsystems can include disciplines already in Aviary as well as disciplines outside of Aviary.
Aviary can also perform both design and off-design analyses and optimizations.

## Why is Aviary open-source?

Aviary is open-source to provide the largest impact possible to the aviation community and to encourage a vibrant ecosystem of students, researchers, and industry practitioners.
One of NASA's guiding mottos is "research and technology for the benefit of all."
Aviary is built and shared with this in mind.

## What is the fidelity level of Aviary?

Aviary by itself provides low-fidelity analysis.
For medium- and high-fidelity analysis, users need to create those models themselves and link them to Aviary.
Aviary comes with [multiple examples of external subsystems](../user_guide/using_external_subsystems.md) that can be used to create medium- and high-fidelity models.

## Are you able to connect high-fidelity analysis to Aviary?

Yes, either as external subsystems, or by including their outputs in tabular form for supported disciplines.

## Are you able to connect external toolboxes to Aviary?

Yes, an external toolbox is called an external subsystem in Aviary, and connecting these is a standard feature.

## Can Aviary perform analysis or optimization?

Yes, Aviary can perform either analysis or optimization, depending on what the user chooses. Both capabilities are available.
Aviary uses OpenMDAO and gradient-based methods throughout to enable efficient optimization.

## How does Aviary perform aircraft design?

Aviary operates in two modes: design and off-design. (The off-design mode is currently under development and primarily works with the 2-degrees-of-freedom mission analysis method.)
An Aviary problem consists of three parts: pre-mission, mission, and post-mission.
In design mode, these parts pass data back and forth, and all three are involved in designing the aircraft.
In off-design, these modes pass data back and forth, and all three are involved in analyzing and/or optimizing the off-design flight.

## What is the difference between OpenMDAO and Aviary?

OpenMDAO is a generic gradient-based analysis and optimization framework for any sort of system (e.g. wind turbines, aircraft, CNC machining).
Aviary uses OpenMDAO for aircraft-specific analysis and has features like data hierarchies and template missions.

## Is Aviary verified or validated?

Validation ensures Aviary meets end-user needs whereas verification ensures correct calculations in Aviary.
Aviary has been verified against codes like FLOPS and GASP which have been independently verified and validated.
It contains a suite of unit tests and benchmark tests for verification.
Ongoing discussions with stakeholders ensure Aviary fits their needs, with ongoing work to meet changing features and needs.

## Who are the Aviary stakeholders?

Stakeholders include NASA projects, industry partners, academia, and more.
They all need system-level analysis and optimization of aircraft.

## What can Aviary offer that legacy codes cannot do and how does Aviary compare to legacy codes?

Aviary can perform analysis and optimization on electrified aircraft, mission optimization, and integrate different discipline tools.
It offers flexibility in modeling various aircraft types, unlike previous tools limited by their equations.
Specifically, Aviary enables coupled aircraft-mission design for complex systems with arbitrarily connected subsystems in a flexible manner. It also provides analytic gradients for all core-subsystems, a feature unique from the legacy codes.
