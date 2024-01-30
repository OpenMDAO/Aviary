# Overview of Aviary Functionality

The Aviary code is in part based on two legacy NASA aircraft analysis codes, GASP (General Aviation Synthesis Program) and FLOPS (Flight Optimization System). (Note: FLOPS also had a successor named LEAPS which implemented the same equations in python instead of Fortran. References for LEAPS will be substituted where references for FLOPS are sparse.) These two codes provided the equations for the empirical weight, geometry, and aerodynamic estimating relationships in the Aviary core subsystems, as well as a few equations for the mission and propulsion subsystems. The bottom of this page has several references to theory and user's manuals for GASP and FLOPS/LEAPS. These two legacy codes were and still are widely used at NASA and beyond. However, the Aviary code far outpaces them in flexibility, model integration capability, and optimization potential.

There are two different aspects to Aviary. The first aspect is computational: Aviary contains OpenMDAO models of several basic subsystems, and these subsystems can be combined in various fashions to build purely Aviary-based aircraft models. An example of a purely Aviary-based models is available [here](../examples/simple_mission_example). Below are the five subsystems which Aviary inherently provides.

## Geometry
The [geometry](./geometry) calculations that Aviary provides are made up of empirical equations and the basic equations of aircraft design. The empirical estimations in these equations are based on the conventional tube-and-wing aircraft design, and are well suited to this conventional configuration. For more novel aircraft configurations the Aviary team recommends implementing an external subsystem.

## Aerodynamics
The [aerodynamic](./aerodynamics) calculations that Aviary provides are also made up of empirical equations and are also based on the conventional tube-and-wing aircraft design. However, Aviary also provides the ability to read in tabular aerodynamic data in the form of a drag polar, lift curve, etc.

## Propulsion
The [propulsion](./propulsion) calculations that Aviary provides require an input engine deck. Aviary does not have the inherent ability to model a gas turbine engine cycle, and relies on outside data. The propulsion subsystem has the capability to interpolate and use this data, and also includes some basic calculations and checks, but it does not actually build its own engine model.

## Mass
The [mass](./mass) calculations that Aviary provides are similar to the geometry calculations. They use empirical equations based on the traditional tube-and-wing aircraft design, and work well for this case but break down in the case of unconventional configurations. For more novel aircraft configurations the Aviary team recommends implementing an external subsystem.

## Mission Analysis
The [mission analysis](./mission) calculations that Aviary provides are broader in scope than the other disciplines. Aviary provides two different types of equations of motion (2DOF and height energy), and also provides two different ways to integrate these EOMs (collocation and analytic shooting). The intended capability is that either EOM can be used with either integration technique.


The second aspect of Aviary, instead of being computational like the first, is about subsystem integration. In addition to the capability to build purely Aviary-based aircraft models, Aviary provides the ability to build mixed-origin aircraft models, which are aircraft models consisting partially or entirely of external user-provided subsystems. In the case of a mixed-origin model, instead of just selecting existing Aviary subsystems and combining them into an aircraft model, the user also provides the Aviary code with their own subsystems (these could be pre-existing codes such as [pyCycle](https://github.com/OpenMDAO/pyCycle), or they could be new subsystems the user has built themselves). An example of a mixed-origin model is the [OpenAeroStruct example case](../examples/OAS_subsystem). In this example, the built-in Aviary subsystem for mass is partially replaced by external subsystems.

### FLOPS/LEAPS References
1. [Here](https://ntrs.nasa.gov/api/citations/20190000442/downloads/20190000442.pdf) is an overview of the LEAPS code development.
2. [Here](https://ntrs.nasa.gov/api/citations/20200001143/downloads/20200001143.pdf) is an overview of performing aircraft analysis using LEAPS.
3. [Here](https://ntrs.nasa.gov/api/citations/20190000427/downloads/20190000427.pdf) is a detailed description of the mission analysis method used in LEAPS/FLOPS.
4. [Here](https://ntrs.nasa.gov/api/citations/20170005851/downloads/20170005851.pdf) is a detailed description of the aircraft weight calculating method used in LEAPS/FLOPS.
5. [Here](https://ntrs.nasa.gov/api/citations/20190000431/downloads/20190000431.pdf) is a comparison of the LEAPS/FLOPS weight calculating method to other similar weight calculating methods.

### GASP References
1. [Here](https://ntrs.nasa.gov/api/citations/19810010562/downloads/19810010562.pdf) is documentation on the theoretical development of the main program of GASP.
2. [Here](https://ntrs.nasa.gov/api/citations/19810010563/downloads/19810010563.pdf) is documentation on the theoretical development of the aircraft geometry calculating methods in GASP.
3. [Here](https://ntrs.nasa.gov/api/citations/19810010564/downloads/19810010564.pdf) is documentation on the theoretical development of the aerodynamics calculating methods in GASP.
4. [Here](https://ntrs.nasa.gov/api/citations/19810010565/downloads/19810010565.pdf) is documentation on the theoretical development of the propulsion calculation methods in GASP.
5. [Here](https://ntrs.nasa.gov/api/citations/19810010566/downloads/19810010566.pdf) is documentation on the theoretical development of the weight calculation methods in GASP.
6. [Here](https://ntrs.nasa.gov/api/citations/19810010567/downloads/19810010567.pdf) is documentation on the theoretical development of the performance calculation methods in GASP.
7. [Here](https://ntrs.nasa.gov/api/citations/19810010568/downloads/19810010568.pdf) is documentation on the theoretical development of the economic calculation methods in GASP. (Note, the economics calculations from GASP have not yet been implemented in Aviary.)