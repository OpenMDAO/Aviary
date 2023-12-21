# Tools That Aviary is Built Upon

Multiple tools are used by Aviary, including OpenMDAO, Dymos, and more.
Additionally, other existing NASA tools have been adapted and converted for use within Aviary,
including FLOPS and GASP.
This doc page details each of these tools, how they're used in Aviary, then goes into more detail about each tool.

## How and why each tool is used

At the base of Aviary is [**OpenMDAO**](https://openmdao.org/newdocs/versions/latest/main.html),
an open-source modeling and optimization tool written in Python.
All of Aviary is written within the OpenMDAO framework.
This means that a huge portion of the behind-the-scenes computational processing is handled
by OpenMDAO, including numerical solvers, derivative information passing, and parallel processing.
Additionally, existing tools written in OpenMDAO can easily interface with Aviary

[**Dymos**](https://openmdao.github.io/dymos/) is an open-source tool for the design of complex
dynamic systems which is especially useful for trajectory optimization.
Dymos is written within the OpenMDAO framework and provides time integration methods,
results postprocessing, and much more to enable Aviary's coupled mission design optimization capabilities.
In addition to Aviary's models being made within the OpenMDAO framework, Aviary uses
the concepts of Dymos' phases and trajectories when constructing mission optimization problems.

Aviary directly implements equations from earlier NASA aircraft design tools, specifically **FLOPS** and **GASP**.
Calculations from both of these tools were ported directly into Aviary across all parts of the code.
Painting in broad strokes, these tools were used to design conventional aircraft, especially at the commercial scale.

Now, let's talk about how these tools come together in a general sense.
Here's an image roughly showing the building blocks of Aviary.

![Aviary's toolstack](images/tool_stack.svg)

OpenMDAO is at the foundation in this figure because the entire Aviary tool is written in this framework.
Some portions of Aviary, specifically the models (think aerodynamics, weights, propulsion, etc)
are simply OpenMDAO systems.
This is why the lighter blue box for "Aviary: models" rests atop OpenMDAO's block.
Other parts of Aviary, specifically the infrastructure that brings all the models together and performs
the time integration and trajectory optimization, is written using Dymos.
This is why the "Aviary: platform and tool" box is on top of the Aviary models and Dymos blocks.

Equations and calculations from both FLOPS and GASP exist within Aviary, hence why those tools are shown in transparent boxes within the Aviary models block.
This is because those tools are not _directly_ used in Aviary, but rather they strongly contributed
to the calculations and models that exist within Aviary.

## OpenMDAO

[**OpenMDAO**](https://openmdao.org/newdocs/versions/latest/main.html) is the foundational framework for Aviary.
Developed by NASA, it's designed to enable and facilitate multidisciplinary system analysis and optimization.
One of the significant benefits of using OpenMDAO is its built-in tools to enable gradient-based optimization
of complex systems.
This means that Aviary can make use of optimization algorithms efficiently, primarily due to OpenMDAO's
handling of derivative calculation and aggregation.

Although we say that Aviary is written using OpenMDAO, you can think of it as being written in Python in a
way that OpenMDAO understands.
The models within Aviary are written as OpenMDAO components and groups.
Using OpenMDAO provides a large number of benefits, including [advanced debugging tools](https://openmdao.org/newdocs/versions/latest/features/debugging/debugging.html),
[model visualization](https://openmdao.org/newdocs/versions/latest/features/model_visualization/main.html),
[derivative handling](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_derivatives/main.html),
and much more.

OpenMDAO is a widely accepted and used tool within NASA and externally.
Its uses, published research, and industrial applications are too numerous to list here.
In short, OpenMDAO has been used to design aircraft, wind turbine blades, electric motors, spaceflight trajectories, composite structures, and much more.

## Dymos

[**Dymos**](https://openmdao.github.io/dymos/) extends the capabilities of OpenMDAO by focusing on the optimization
of dynamic systems, especially over time-based trajectories.
The challenges of designing complex systems with time-dependent performance (like aircraft trajectories or space mission profiles)
are addressed by Dymos.
It provides the ability to define phases of a mission, optimize control profiles over these phases, and ensure that the system's dynamics are correctly integrated over time.

For Aviary, Dymos brings in the ability to perform mission design optimization.
Its integration with OpenMDAO means that Aviary can optimize both the system's design and its mission simultaneously.
Dymos handles the complexities of time-based integration, allowing Aviary to focus on defining the system dynamics and objectives.

NASA has used Dymos in various projects, from optimizing the trajectory of space missions to ensuring
efficient electric aircraft flight profiles subject to thermal constraints.
These projects have benefited from Dymos's capabilities and Dymos has in turn grown as a more than capable tool, hence its usage in Aviary.

## Simupy

[**Simupy**](https://simupy.readthedocs.io/en/latest/) is a Python package for the simulation of dynamic systems.
It provides a framework for defining and solving differential equations, including the ability to handle
discontinuities and events.
As Aviary is further developed, Simupy will be available to handle the time integration of the system dynamics.
This capability is not currently fully integrated into Aviary.

## FLOPS

The following description has been adapted from the FLOPS User's Guide[^FLOPSMAN]. For
more detailed information, please see the documentation that accompanies FLOPS[^FLOPS].
Not all of the features from FLOPS have been implemented in Aviary.

The Flight Optimization System (FLOPS) is a multidisciplinary system of computer programs
for conceptual and preliminary design and evaluation of advanced aircraft concepts. It
consists of six primary modules: 1) weights, 2) aerodynamics, 3) propulsion data scaling
and interpolation, 4) mission performance, 5) takeoff and landing, and 6) program
control.

The weights module uses statistical/empirical equations to predict the weight of each
item in a group weight statement. In addition, a more analytical wing weight estimation
capability is available for use with more complex wing planforms.

The aerodynamics module provides drag polars for performance calculations. Alternatively,
drag polars may be input and then scaled with variations in wing area and engine
(nacelle) size.

The propulsion data scaling and interpolation module uses an engine deck that has been
input, fills in any missing data, and uses linear or nonlinear scaling laws to scale the
engine data to the desired thrust. It then provides any propulsion data requested by the
mission performance module or the takeoff and landing module.

The mission performance module uses the calculated weights, aerodynamics, and propulsion
system data to calculate performance based on energy considerations. Several options
exist for specifying climb, cruise, and decent segments. In addition, acceleration, turn,
refueling, payload release, and hold segments may be specified in any reasonable order.
Reserve calculations can include flight to an alternate airport and a specified hold
segment. Some support is provided for supersonic aircraft.

The takeoff and landing module computes the all-engine takeoff field length, the balanced
field length including one-engine-out takeoff and aborted takeoff, and the landing field
length. The approach speed is also calculated, and the second segment climb gradient and
the missed approach climb gradient criteria are evaluated. Insofar as possible with the
available data, all FAR Part 25 or MIL-STD-1793 requirements are met. The module also has
the capability to generate a detailed takeoff and climbout profile for use in calculating
noise footprints.

Through the program control module, FLOPS may be used to analyze a point design,
parametrically vary certain design variables, or optimize a configuration with respect to
these design variables (for minimum gross weight, minimum fuel burned, maximum range,  or
minimum NOx emissions) using nonlinear programming techniques. Several input options are
available, including design variables for both aircraft configuration and performance.

## GASP

The following description has been taken from the GASP User's Guide[^GASPMAN]:

NASA's Ames Research Center has developed the General Aviation Synthesis Program, GASP.
This computer program performs tasks generally associated with aircraft preliminary design and allows an analyst the capability of performing parametric studies in a rapid manner.
GASP was originally made to emphasize small fixed-wing aircraft employing propulsion systems varying from a single piston engine with fixed pitch propeller through twin turboprop/turbofan powered business or transport type aircraft.
It now support hybrid wing body, truss-braced wing, and other unconventional configurations.
The program may be operated from a computer terminal in either the "batch" or "interactive graphics" mode.

The program is comprised of modules representing the various technical disciplines integrated into a computational flow which ensures that the interacting effects of design variables are continuously accounted for in the aircraft sizing procedure.
The model is a useful tool for comparing configurations, assessing aircraft performance and economics, performing tradeoff and sensitivity studies, and assessing the impact of advanced technologies on aircraft performance and economics. By utilizing the computer model the impact of various aircraft requirements and design factors may be studied in a systematic manner with benefits measured in terms of overall aircraft performance and economics.

[^FLOPS]: Flight Optimization System (FLOPS) Software v.9 (LAR-18934-1). NASA Technology
Transfer Program. URL: https://software.nasa.gov/software/LAR-18934-1 [retrieved
September 22, 2023]

[^FLOPSMAN]: McCullers, L., FLOPS User’s Guide, NASA Langley Research Center, Hampton,
Virginia, 2011 (available with public distribution of software).

[^GASPMAN]: Hague, D. S. et al., GASP - General Aviation Synthesis Program,
URL: https://ntrs.nasa.gov/api/citations/19810010562/downloads/19810010562.pdf [retrieved
December 6, 2023]