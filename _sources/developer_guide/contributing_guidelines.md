# Guidelines for Contributing Code

Welcome to our guide for contributing to the Aviary codebase. We are so glad you are interested in contributing to our software! There are a few guidelines that our developers follow in order to ensure that Aviary is well organized and uniform. If you would like to contribute to Aviary, please take a minute to read these guidelines:

## Coding Standards
In order to ensure that our code is readable for our contributors, we ask that you follow our coding standards located [here](coding_standards).

## Unit Testing
We require all code entering our codebase to be validated and regression tested. As part of this requirement, any new code that you contribute to our codebase must have an associated unit test. Our pull request approvers will check for this and will ask you to add a test if one is missing. For details on our unit testing structure see [Unit Tests](unit_tests).

## Thorough Documentation
Documentation is the backbone of the Aviary team's support for our user community. The goal of Aviary's documentation is to provide a way for Aviary users to learn the codebase and have their questions answered in an efficient manner. Thus, we monitor the documentation to ensure that changes in the code are reflected in the docs, and that new code features are documented as well. As a result of this, any pull request which alters a feature's behavior must also update the documentation for that feature, and any pull request which creates a new feature for use by a user must also document that feature. For a guide on writing documentation in Aviary visit [How to Contribute Docs](how_to_contribute_docs).

## Docstrings
The Aviary codebase is currently under active development and cleanup, including the addition of docstrings. Thus, not every function and class currently includes a docstring, however, we are slowly adding them. In order to move forwards instead of backwards we require that all added functions and classes include a docstring in the numpy format.

## Benchmark Tests
The Aviary codebase has several benchmark tests which test some of the baseline models included in Aviary. These tests supplement the unit test capability, and are tested frequently by the Aviary team. We encourage you to run these tests using our test runner located [here](https://github.com/OpenMDAO/Aviary/blob/main/aviary/run_all_benchmarks.py).

## Use of Issue Backlog
The Aviary team would like a chance to interact with and get community engagement in feature changes to the codebase. The primary place that this engagement happens is in the [issue backlog](https://github.com/OpenMDAO/Aviary/issues/new/choose) using the "feature or change request" section. In addition, we would like to be able to track bug fixes that come through the code. To support these goals we encourage users to create issues, and we encourage code contributors to link issues to their pull requests.