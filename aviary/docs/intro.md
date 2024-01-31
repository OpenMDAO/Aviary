# Aviary Documentation

This is the landing page for all of Aviary's documentation, including a user's guide, developer's guide, and theory guide, as well as other resources. Welcome!

## What Aviary is

[Aviary](https://github.com/OpenMDAO/Aviary) is an aircraft analysis, design, and optimization tool built on top of the Python-based optimization framework [OpenMDAO](https://github.com/OpenMDAO/OpenMDAO).
Aviary provides a flexible and user-friendly optimization platform that allows the beginning aircraft modeler to build a useful model, the intermediate aircraft modeler to build an advanced model, and the advanced aircraft modeler to build any model they can imagine.

Features of Aviary include:

- included simple subsystem models for aerodynamics, propulsion, mass, geometry, and mission analysis
- ability to add user-defined subsystems
- gradient-based optimization capability
- analytical gradients for all included subsystems

## How to Read These Docs

The Aviary documentation is broken up into several sections, each of which is designed to teach a different aspect of Aviary.
Reading the entirety of the docs is highly recommended for new users, but please read through the Getting Started section at a minimum.

You can read through the documentation in order or you can jump to the sections that interest you the most.

```{note}
Use the interactive table of contents on the left side of the page to navigate through the documentation.
```

## User Guide

The [Aviary user interface](user_guide/user_interface.md) is under development and employs a 3-tiered approach that is broken up into 3 separate levels.
The user guide walks through how to use each of these levels in detail.

The actual finer points of aircraft design and what these input values should be set to are beyond the scope of this documentation.
We refer users to their aircraft design textbooks as well as their experienced coworkers for information in this area.
This user guide is simply designed to teach the basics of using Aviary for aircraft analysis.

## Examples

The Aviary code includes a [suite of built-in examples](examples/intro.md) which the Aviary team has developed to demonstrate the capability of the Aviary code.
These examples range in complexity and length from a Level 1 input file of a simple aircraft analysis including only Aviary core subsystems to a Level 3 input script where the user has added several external subsystems and manually controlled what variables are passed where.
The Aviary team recommends that the examples be used as as starting point for building your first few Aviary models until you have built up examples of your own.

## Theory Guide

The [theory guide](theory_guide/intro.md) details how the five core subsystems (aerodynamics, propulsion, mass, geometry, and mission analysis) work and how the integration capability combines them together.
The theory guide also gives a much deeper understanding of the equations and modeling assumptions behind Aviary.

## Developer Guide

This [developer guide](developer_guide/codebase_overview.md) walks through each aspect of the code from the perspective of a developer who would like to contribute code.

The Aviary development team is housed out of NASA but welcomes code input and pull requests from the public.
We are developing a formal review process, but at the moment each code contribution will be made as a pull request and reviewed by the development team.

## Miscellaneous Resources

There are some features of the Aviary code which are not addressed in the above documentation.
The [miscellaneous resources](misc_resources/FAQ.md) section includes documentation on these additional features, as well as other relevant information.

## Table of contents

```{tableofcontents}
```
