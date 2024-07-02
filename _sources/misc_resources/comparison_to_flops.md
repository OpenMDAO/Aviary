# Comparison to FLOPS

Portions of Aviary are based on the publicly-released version of the Flight
Optimization System ([FLOPS](https://software.nasa.gov/software/LAR-18934-1)).
Many of the features are replicated directly and will produce the same results
whereas others are replicated using similar methods and will produce similar
results but not exactly the same. This section discusses the different features
of the public version of FLOPS that have been replicated in Aviary and how they
compare. Note that additional features existed in non-public versions of FLOPS,
but they are not included in this comparison.

## Geometry

Geometry definition in Aviary uses a combination of methods from FLOPS and GASP.
When using either (or both) of the FLOPS-based internal aerodynamics or mass
prediction methods, the geometry is defined using the same quantities used by
FLOPS.

Since the parametric variation and optimization capabilities of FLOPS are not
implemented in the same way within Aviary (see [Optimization](#Optimization and parametric variation)
below), only the baseline values for geometric quantities are entered.

## Mass equations

FLOPS weights equations are replicated very closely, except that in Aviary they
calculate mass in lbm instead of weight in lbf. Currently only the transport
mass equations and alternate mass equations are fully implemented. BWB (blended wing body) mass
equations are partially implemented and there are plans to complete them in the
future. Fighter mass equations are not implemented.

The detailed wing mass calculation in Aviary is based on the method in FLOPS,
but it was vectorized for efficiency. Note that LEAPS used a modified wing weight
method for very large numbers of wing-mounted engines; this modification is not
implemented in Aviary.

## Aerodynamics

Like FLOPS, Aviary allows the user to specify the aerodynamic characteristics of
an aircraft using an internal empirical analysis or table-based user input.
Aviary also allows the user to dynamically link an externally-generated aero
model. These three modes are discussed in the following sections.

### Empirical

The internal empirical aerodynamics calculation from FLOPS has been fully
implemented in Aviary and the user can choose to use it or another methodology
based on GASP. The FLOPS-based equations calculate only the drag polars (drag as
a function of Mach number, altitude, and lift coefficient), whereas the
GASP-based equations also calculate lift as a function of angle of attack.

The empirical aerodynamic analysis in Aviary is based directly on the
methodology from FLOPS, but there are some differences in the numerical
libraries used. The skin-friction analysis in FLOPS uses a two-level fixed-point
iteration for a fixed number of iterations, with no check on convergence; in its
place, Aviary uses Newton's method. Because of this difference, there may be
differences in the computed skin-fraction drag due to a different level of
convergence. In addition, table lookups for all other parts of the empirical
drag methodology use a slightly different interpolation method than FLOPS. This
difference seems to have the largest effect at Mach numbers higher than the
design Mach number. The data being interpolated show a discontinuity that the
methodology captures with just a few points, so the results will be dependent on
the change in interpolation method.

### Tabular input

Aviary has the option to use tabular input for aerodynamics using the same
format as FLOPS (zero-lift drag is given as a function of Mach number and
altitude, while lift-dependent drag is given as a function of Mach number and
lift coefficient). Currently there is not a method for reading in
FLOPS-formatted block-style aero tables; instead the TabularAeroBuilder is used
to build an equivalent table. There are tentative plans to add the ability to
read the FLOPS-formatted aero table directly from a file.

As with the empirical aerodynamics, the interpolation method used in Aviary may
differ from the built-in linear interpolation used by FLOPS, so interpolated
values may differ, even for the same tabular values.

### External model

In addition to the two previous methods for defining the aero characteristics,
Aviary also has the ability to dynamically link an external aerodynamics
analysis using the AeroBuilderBase class. This external analysis can be
implemented as a metamodel, or even as an on-the-fly analysis. The interface
between the mission analysis and aerodynamics is configurable, so in addition to
the traditional relationship between the flight condition and lift and drag, the
aerodynamics can be a function of any number of variable such as throttle
setting. This integration represents an improvement over FLOPS in the ability to
analyze aircraft with tightly-integrated aerodynamics and propulsion, or other
advanced configurations for which the aerodynamic characteristics are more
complex than for a traditional tube-and-wing configuration.

## Propulsion

FLOPS allows for the input of a maximum of two engine types, each with its own
engine deck, whereas in Aviary an unlimited number of engine types may be
defined. Aviary retains many of the relations used by FLOPS to scale engine mass and
performance as a function of change in target thrust. Aviary introduces the
capability to define these scaling relations on a per-engine-type basis, instead
of using the same scaling relations for all engine types.

### Engine deck input

Aviary uses a custom format for engine decks which is similar to FLOPS, but more
flexible than the fixed-column FLOPS format, but this means that FLOPS-formatted
engine decks cannot be used directly. Once the FLOPS engine deck is converted to
an Aviary-compatible format, however, the deck can be used to define the engine
performance in the same way. Instead of using power code to represent specific
engine throttle conditions like in FLOPS, Aviary's throttle is a continuous
engine control parameter (typically defined as a percentage of maximum net
thrust/power).
	
Aviary currently does not have the ability to fill in missing data points like
FLOPS does, but generation of flight idle points through extrapolation is
handled similarly.
	
In FLOPS, engine decks are interpolated linearly. Aviary allows the user to
specify the type of interpolation to use for the engine deck; one option is to
use linear interpolation, but other interpolation types are also available. If a
different method is used, the interpolated engine performance may be different
than in FLOPS, even for the exact same tabular data.

### Dynamic engine models

In addition to the traditional engine deck, Aviary also allows a pyCycle model
to be integrated directly into the analysis. This integration allows for the
optimization of both airframe and propulsion design variables simultaneously,
which is a capability not available in FLOPS. This direct integration can also
be combined with a more complex aerodynamics model to improve the ability to
optimize aircraft with tightly-integrated aerodynamics and propulsion.

## Mission

### Main mission segments

FLOPS uses a single type of mission analysis: height-energy, with an explicit
integration scheme. When full developed, Aviary will offer this type of
analysis, as well as a variety of other mission analysis types.

Currently there are two options for the equations of motion used: two degree of
freedom, and height-energy. The height-energy equations are the same as those
used in FLOPS, although the integration scheme is different. Prescribed mission
equations are also being planned.

Currently, integration of the equations of motion is done using a collocation
method. Implementation of an analytic shooting method is underway. Eventually
Aviary will also offer a simple explicit integration, similar to FLOPS. The
combination of the height-energy equations with the explicit integration should
give similar results to FLOPS, but they will not be exactly the same because of
differing numerical methods used to achieve convergence.

### Reserve mission

Reserve mission segments can be simulated in the same manner as is done in
FLOPS. Currently this would require manually setting up additional mission
segments and linking the starting conditions. In the future this will be handled
in a more automated fashion.

### Takeoff and landing

Aviary includes the ability to calculate takeoff and landing field lengths using
either type of method used by FLOPS (simple, or detailed), as well as a method
based on GASP.

The simple takeoff and landing equations from FLOPS are replicated in Aviary,
and should produce nearly identical results.

The detailed takeoff and landing analysis methods from FLOPS are also replicated
using the same equations of motion, but instead of a built-in explicit
integration scheme like that used by FLOPS, Aviary can use any number of
integration schemes (currently, only a collocation scheme is used). Not all of
the relevant performance requirements from FAR 25 are implemented, and in the
collocation scheme they are enforced indirectly. Because the OpenMDAO
collocation scheme is very different than the FLOPS integration, it is almost
certain that there will be differences in the results, but in general the
results should be similar. Both the detailed takeoff and landing capabilities
are still works in progress, and as they are improved they will likely produce
results that are closer to those from FLOPS.

Since the detailed takeoff and landing trajectories are built up from modular
sequences of phases, they are more flexible than the fixed sequence defined in
FLOPS and can be used to model more complex operations. For example, a
higher-fidelity aerodynamic model can be used, or rotation can be controlled
using thrust vectoring. 

Aviary also includes the ability to calculate extended takeoff and landing
profiles for use in noise analysis. This capability exists within the same set
of components as the performance calculations and merely requires extending the
performance calculation with additional trajectory segments.

### Off-design missions

Off-design, or economic, missions (defined in FLOPS using the $RERUN namelist)
are possible by manually setting up additional mission segments and linking the
starting conditions to the results of the design mission analysis. This process
will become more automated in the future, including the ability to assemble
payload-range diagrams.

### Optimization and parametric variation

Optimization of Aviary models is handled differently. FLOPS features a built-in
optimizer that allows users to optimize the model based on a hard-coded set of
output variables that can be used to build an objective function, and using a
hard-coded set of design variables. Aviary is built in the OpenMDAO framework
and models can be optimized by drawing on all of the optimization capabilities
inherent to that framework. Any input variable can be designated as a design
variable in the optimization, and any set of output variables can be used to
build a custom objective function. Almost all of Aviary's components use
analytic derivatives to greatly enhance gradient-based optimization schemes.

## FLOPS features not included

### Sonic boom

FLOPS features an approximate method for calculating sonic-boom overpressures
directly underneath the flight path. This feature is not implemented in Aviary
and there are currently no plans to include it.
