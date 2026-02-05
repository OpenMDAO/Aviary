# Underlying Tools
Aviary is built upon multiple existing NASA tools.
Knowledge of what these tools are, and why Aviary uses them, will help build understanding for how and why Aviary is structured the way that it is.

## Core Components
Several Python packages comprise the fundamental building blocks of the Aviary codebase. These tools are the way through which Aviary actually performs optimization.

### OpenMDAO
At the base of Aviary is [**OpenMDAO**](https://openmdao.org/newdocs/versions/latest/main.html),
a NASA-developed open-source modeling and optimization tool written in Python. OpenMDAO is a widely accepted and used tool both within NASA and in academia and industry.

OpenMDAO helps compartmentalize individual pieces of analyses and provides a number of tools to help users visualize how information flows through their model. It also provides easy access to state-of-the-art gradient-based optimization that can be used for your model.

Aviary is written from the ground up to use the OpenMDAO framework. In fact, Aviary can be thought of as a convenient way to set up, interact with, and run an extremely large and complicated OpenMDAO problem tailored to aircraft design.
This means that a huge portion of the behind-the-scenes computational processing is directly handled
by OpenMDAO, including numerical solvers, derivative information passing, and parallel processing.
Additionally, existing tools written in OpenMDAO can easily interface with Aviary

### Dymos
[**Dymos**](https://openmdao.github.io/dymos/) is another NASA developed open-source Python package that extends OpenMDAO's capabilities to handle optimization of dynamic systems, especially over time-based trajectories.
Aviary uses it for trajectory optimization. Its integration with OpenMDAO means that Aviary can optimize both the vehicle's design and its mission simultaneously.

Similar to OpenMDAO's deep integration into Aviary, mission analysis is built using Dymos. 

## Legacy Tools
For actual modeling of aircraft properties and performance, Aviary includes equations taken from two of NASA's existing conceptual aircraft design codes: FLOPS (Flight Optimization System) and GASP (General Aviation Synthesis Program).
These tools are fully-functional, standalone programs that were written in Fortran and used in largely similar ways.
Both utilized text-based input files and were accessed via the command line. Both codes used different methodologies to model various aspects of aircraft performance, such as mass estimation, aerodynamic performance, and mission modeling.
Aviary includes utilities for converting FLOPS and GASP input files into formats that can be used by Aviary.

### FLOPS

The Flight Optimization System (FLOPS), developed at NASA Langley Research Center, is a multidisciplinary system for conceptual and preliminary design and evaluation of advanced aircraft concepts.
It consists of six primary modules: weights, aerodynamics, propulsion data scaling and interpolation, mission performance, takeoff and landing, and program control.

The weights module uses statistical/empirical equations to predict the weight of each item in a group weight statement.
Aviary directly utilizes these equations.

The aerodynamics module provides drag polars for performance calculations. Alternatively, drag polars may be input and then scaled.
Aviary also directly utilizes the aerodynamics module equations.

The propulsion data scaling and interpolation module uses an engine deck (a file containing tabluar engine performance data), fills in any missing data, and uses linear or nonlinear scaling laws to scale the engine data to the desired thrust. It then provides any propulsion data requested by the
mission performance module or the takeoff and landing module.
Aviary's propulsion system is significantly more flexible and robust than FLOPS, but it supports the use of engine decks in a similar manner to FLOPS.

The mission performance module uses the calculated weights, aerodynamics, and propulsion system data to calculate performance based on energy considerations. Several options exist for specifying climb, cruise, and decent segments, as well as specifying a reserve mission.
Aviary uses the same fundamental equations of motion, but solves the trajectory using Dymos instead of a hard-coded numerical method.

The takeoff and landing module computes the all-engine takeoff field length, the balanced field length including one-engine-out takeoff and aborted takeoff, and the landing field length. The approach speed is also calculated, and the second segment climb gradient and the missed approach climb gradient criteria are evaluated. Insofar as possible with the available data, all FAR Part 25 or MIL-STD-1793 requirements are met. The module also has the capability to  generate a detailed takeoff and climbout profile for use in calculating noise footprints.
Aviary currently implements the FLOPS equations for computing takeoff and landing distance for the flown mission, as well as implementing the detailed takeoff and climbout profiles. For the detailed profiles, Dymos is used to solve the trajectory instead of the FLOPS numerical method. The additional field length calculations and other key metrics have not yet been implemented in Aviary.

For a more detailed description of FLOPS' capabilities, see its User's Guide[^FLOPSMAN], as well as the official documentation that accompanies the public distribution of the code[^FLOPS].

### GASP

NASA's Ames Research Center developed the General Aviation Synthesis Program, or GASP.
This program performs tasks generally associated with aircraft preliminary design and allows an analyst the capability of performing parametric studies in a rapid manner.

GASP was originally made to emphasize small fixed-wing aircraft employing propulsion systems varying from a single piston engine with fixed pitch propeller through twin turboprop/turbofan powered business or transport type aircraft. It now supports hybrid wing body, truss-braced wing, and other unconventional configurations.

The program is comprised of modules representing the various technical disciplines. Aviary implements many of these modules. Empirical equations for estimating aircraft weight and aerodynamic performance are directly implemented in Aviary. This includes GASP support for hybrid wing body and truss-braced wing aircraft. Propulsion analysis in GASP also utilizes the concept of engine decks. Aviary's Propulsion subsystem supports a similar style of analysis. The equations of motion used for mission analysis in GASP have been implemented in Aviary, utilizing Dymos to solve for the trajectory instead of the numerical methods implemented in GASP.

Specifics on the capabilities of GASP can be found in its User's Guide[^GASPMAN].

# References
[^FLOPS]: Flight Optimization System (FLOPS) Software v.9 (LAR-18934-1). NASA Technology
Transfer Program. URL: https://software.nasa.gov/software/LAR-18934-1 [retrieved
September 22, 2023]

[^FLOPSMAN]: McCullers, L., FLOPS User’s Guide, NASA Langley Research Center, Hampton,
Virginia, 2011 (available with public distribution of software).

[^GASPMAN]: Hague, D. S. et al., GASP - General Aviation Synthesis Program,
URL: https://ntrs.nasa.gov/api/citations/19810010562/downloads/19810010562.pdf [retrieved
December 6, 2023]