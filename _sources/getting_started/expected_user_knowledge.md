# Expected User Knowledge

This section details the recommend user knowledge to use Aviary for different purposes.
We'll reference the [three user interface levels](../user_guide/user_interface.md) throughout.

A graphical summary of the info presented below is shown in the figure.

![Aviary user knowledge](images/expected_user_knowledge.svg)

## Recommended background in aircraft design

One of the most important aspects of performing computational design and optimization is understanding the resulting designs, if they're realistic, and what those designs mean in the real world.
Thus, some knowledge of aircraft design is greatly beneficial when running Aviary.

At a basic level, the user is expected to understand how conventional aircraft fly simple missions (takeoff, climb, cruise, descent).
Additionally, a simplistic understanding of how forces act on the aircraft during flight is helpful, including aerodynamic lift and drag, propulsive thrust, and the weight of the aircraft.
Many of Aviary's components are based on these basic principles, which are often detailed in introductory aircraft design textbooks, such as [Raymer's Aircraft Design](https://arc.aiaa.org/doi/book/10.2514/4.104909).

````{margin}
```{note}
We are using Raymer's book as an example of an introductory aircraft design textbook, but there are many other good options available.
We do not explicitly endorse any one textbook or resource.
```
````

More advanced users will benefit by understanding how different mission definitions, constraints, and design variables affect the aircraft design and optimal trajectory.
Advanced users can also benefit from understanding how the different subsystems are modeled and how they interact with each other.
User-defined external subsystems generally require a bit more aircraft design knowledge to understand how to integrate them physically correctly into the Aviary platform.

## Recommended background in optimization

Aviary uses many concepts related to multidisciplinary design and optimization (MDO) to perform its design and optimization tasks.
At a simplistic level, knowledge of MDO is not needed to use Aviary.
Aviary's Level 1 interface is made with the goal of being accessible to users with no knowledge of MDO.

However, to derive the most benefit from Aviary, users should have a basic understanding of MDO concepts.
More advanced knowledge of MDO will be helpful when setting up more complex Aviary models and debugging their behavior.

The OpenMDAO dev team has a series called [Practical MDO](https://openmdao.github.io/PracticalMDO/intro.html) that is a great resource for learning about MDO.
It consists of short lesson videos and corresponding Python notebooks that teach the basics of MDO.
For a quick view into why we use gradient-based multidisciplinary optimization in Aviary, [check out this video](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/gradient_based_mdo.html).

There is also a [free textbook on MDO](http://mdobook.github.io/) written by Professors Joaquim R.R.A. Martins and Andrew Ning.
It is quite readable and available in pdf format.

````{margin}
```{note}
We also do not officially endorse this textbook.
```
````

## Required programming skills

The goal of Aviary is to make aircraft design and optimization accessible to a wide range of users while providing a flexible and powerful platform for advanced users.

This means that the simple user interface at level 1 does not strictly require any programming knowledge.
Users can interact with Aviary through input files and the command line, viewing the results in a web browser.

At level 2, users are expected to have a basic understanding of Python and object-oriented programming.
Aircraft and mission definition occurs in Python scripts, and users can add their own external subsystems to the Aviary platform.
Knowledge of [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) is not required at this level, but it is helpful to understand how the Aviary tool is built on top of OpenMDAO.

Level 3 is the most advanced level, and users are expected to have a strong understanding of Python and object-oriented programming.
Users have complete control over the Aviary model and how the methods are called.

## Familiarity with dependencies

As users dig deeper into the Aviary code, they will encounter the various dependencies that Aviary relies on.
The most important of these is OpenMDAO, which is the optimization framework that Aviary is built on top of. In addition to its [basic user guide](https://openmdao.org/newdocs/versions/latest/basic_user_guide/basic_user_guide.htmlAviary), users are encouraged to learn more about the following topics in the [OpenMDAO Advanced User Guide](https://openmdao.org/newdocs/versions/latest/advanced_user_guide/advanced_user_guide.html):
  - [Implicit Components](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/implicit_component.html)
  - [N<sup>2</sup> Diagram](https://openmdao.org/newdocs/versions/latest/features/model_visualization/n2_details/n2_details.html)
  - [Nonlinear and Linear Solvers](https://openmdao.org/newdocs/versions/latest/features/core_features/controlling_solver_behavior/set_solvers.html)
  - [BalanceComp](https://openmdao.org/newdocs/versions/latest/advanced_user_guide/models_implicit_components/implicit_with_balancecomp.html)
  - [Complex Step](https://openmdao.org/newdocs/versions/latest/advanced_user_guide/complex_step.html)
  - [Total Derivative Coloring](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_derivatives/simul_derivs.html)
  - [MetaModelStructuredComp](https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/metamodelstructured_comp.html)

Aviary also relies on [Dymos](https://github.com/OpenMDAO/Dymos) for its trajectory optimization capabilities. Users are recommended to read through the following sections:
  - [Optimal Control](https://openmdao.github.io/dymos/getting_started/optimal_control.html)
  - [Phases and Segments](https://openmdao.github.io/dymos/getting_started/intro_to_dymos/intro_segments.html)
  - [Aircraft Balanced Field Length Calculation Example](https://openmdao.github.io/dymos/examples/balanced_field/balanced_field.html)

Again, at levels 1 and 2 an understanding of these dependencies is not strictly required.
More knowledge about OpenMDAO and Dymos when using some portions of level 2 will be helpful.
For example, knowing how to scale variables in OpenMDAO or choosing a reasonable number of nodes in Dymos will be helpful when using the Aviary level 2 interface.

At level 3, users are expected to have a strong understanding of OpenMDAO and Dymos because they are working directly with these tools.
Starting with no knowledge of OpenMDAO or Dymos is not recommended at this level.

Starting out with the fantastic [OpenMDAO docs](https://openmdao.org/newdocs/versions/latest/main.html) and looking at some examples there is the best way to learn OpenMDAO.
The [Dymos docs](https://openmdao.github.io/dymos/) and examples are also useful to examine if you're digging more into the trajectory optimization side of Aviary.

<!-- TODO: Add mention to the level 3 onboarding document once it's ready. -->