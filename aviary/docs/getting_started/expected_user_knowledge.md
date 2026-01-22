# Expected User Knowledge

To fully utilize all of the functionality of Aviary, some background knowledge is required. This section details the recommend user knowledge to use Aviary for different purposes.

## Fundamentals of Aircraft Design

One of the most important aspects of performing computational design and optimization is understanding the resulting designs, if they're realistic, and what those designs mean in the real world.
Thus, some knowledge of aircraft design is greatly beneficial when running Aviary.

The user is expected to understand how conventional aircraft fly simple missions (takeoff, climb, cruise, descent).
Additionally, a understanding of how forces act on the aircraft during flight is helpful, including aerodynamic lift and drag, propulsive thrust, and the weight of the aircraft.
Many of Aviary's components are based on these basic principles, which are often detailed in introductory aircraft design textbooks.

More advanced users will benefit by understanding how different mission definitions, constraints, and design variables affect the aircraft design and optimal trajectory.
Advanced users can also benefit from understanding how the different subsystems are modeled and how they interact with each other.
User-defined external subsystems generally require a bit more aircraft design knowledge to understand how they will impact vehicle performance.

## Optimization

Aviary uses many concepts related to multidisciplinary optimization (MDO) to perform its design and optimization tasks.
Knowledge of MDO techniques and theory is not required to use Aviary, however it is strongly recommended even for basic users to help understand their models and debug issues.

The OpenMDAO dev team has a series called [Practical MDO](https://openmdao.github.io/PracticalMDO/intro.html) that is a great resource for learning about MDO.
It consists of short lesson videos and corresponding Python notebooks that teach the basics of MDO.
For a quick introduction to gradient-based multidisciplinary optimization, [check out this video](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/gradient_based_mdo.html) from the Practical MDO series.

There is also a [free textbook on MDO](http://mdobook.github.io/) available written by Professors Joaquim R.R.A. Martins and Andrew Ning.

````{margin}
```{note}
This textbook is cited as a reference only, and is not officially endorsed by the Aviary team or NASA.
```
````

## Programming in Python

The goal of Aviary is to make aircraft design and optimization accessible to a wide range of users while providing a flexible and powerful platform for advanced users.

Users can interact with Aviary without any programming knowledge through use of input files and the command line, and viewing results in a web browser.

Setting up and running Aviary models can also be scripted using the Python API.
For this kind of analysis, users are expected to have a basic understanding of Python and object-oriented programming.
Interacting with Aviary in this way unlocks most of its functionality.
Models and optimization setup are significantly more customizable, and external subsystems can be defined and added.

Knowledge of [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO) is not required at this level, but it is helpful to understand how the Aviary tool is built on top of OpenMDAO.

For advanced users, the Aviary API is just a suggestion.
The code is very modular and lends itself well to being utilized in pieces.
A problem can be completely scripted from beginning to end directly utilizing parts of Aviary where desired.
At this level, users need a strong understanding of Python, object-oriented programming, and the Aviary codebase in general.

### Familiarity With Dependencies

As users dig deeper into the Aviary code, they will encounter the various dependencies that Aviary relies on.
The most important of these is OpenMDAO, which is the optimization framework that Aviary is built on top of. In addition to its [basic user guide](https://openmdao.org/newdocs/versions/latest/basic_user_guide/basic_user_guide.html), users are encouraged to learn more about the following topics in the [OpenMDAO Advanced User Guide](https://openmdao.org/newdocs/versions/latest/advanced_user_guide/advanced_user_guide.html):
<!-- TODO: Review this list -->
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

For most Aviary models an understanding of these dependencies is not strictly required.
Experience with OpenMDAO and Dymos will be helpful when using Aviary's Python API and for diagnosing failed optimizations.
For example, knowing how to scale variables in OpenMDAO or choosing a reasonable number of nodes in Dymos can help solve optimization convergence issues.