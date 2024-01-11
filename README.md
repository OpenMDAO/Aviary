# Aviary -- NASA's aircraft design tool

**Check out the Aviary documentation [here](https://openmdao.github.io/om-Aviary/intro.html).**

## Description

This repository is an [OpenMDAO](https://openmdao.org/)-based aircraft modeling tool that incorporates aircraft sizing and weight equations from its predecessors [GASP (General Aviation Synthesis Program)](https://ntrs.nasa.gov/api/citations/19810010563/downloads/19810010563.pdf) and [FLOPS (Flight Optimization System)](https://software.nasa.gov/software/LAR-18934-1).
It also incorporates aerodynamic calculations from GASP and FLOPS and has the capability to use an aerodynamics deck as well as an aircraft engine deck.
There are two options for the mission analysis portion of this code, a 2 degrees-of-freedom (2DOF) approach, and a height energy (HtEn) approach.
The user can select which type of mission analysis to use, as well as whether to use the FLOPS-based code or the GASP-based code for the weight, sizing, and aerodynamic relations.

## Installation

The simplest installation method for users is to install via pip.
Once you have cloned the Aviary repo, change directories into the top-level Aviary folder (not within the `aviary` folder) and run the following command:

    pip install .

If you want to also run the Aviary test suite (this is not strictly necessary), you can instead run:

    pip install .[test]

If you also want to install all packages used for the Aviary tests _and_ external subsystem examples, you can instead run:

    pip install .[all]

## Documentation

The Aviary documentation is located [here](https://openmdao.github.io/om-Aviary/intro.html).

Otherwise you can build the docs locally:

1. Install jupyter-book using instructions located [here](https://jupyterbook.org/en/stable/start/overview.html
)
2. Go to Aviary/aviary/docs
3. Run the command `sh build_book.sh` from your command prompt of choice
4. Navigate to the built html: `/Aviary/aviary/docs/\_build/html/intro.html`

## Visualization

To create XDSM visualizations of some of the Aviary code, you can run the `run_all.py` utility script within the `aviary/xdsm` directory.
This is not strictly necessary but is used in some of the unit tests.
This requires installing Aviary with the `[test]` or `[all]` options as described above.

## Validation

This code has been validated using output and data from the GASP and FLOPS codes themselves. The GASP-based weight calculations in this code include in their comments which versions of the GASP standalone weights module were used in validation. The aero and EOM subsystem validations were based on runs of the entire GASP and FLOPS code as they stood in the summer of 2021 and the summer of 2022 respectively.

### Quick testing

The repository installation can be tested using the command ``testflo .`` at the top-level Aviary folder. Assuming you have both SNOPT and IPOPT installed, the output should look something like this:

        OK

        Passed:  706
        Failed:  0
        Skipped: 3


        Ran 709 tests using 16 processes
        Wall clock time:   00:00:16.97

### Full testing

In addition to all of the quicker tests, we include multiple integration tests within Aviary.
These have also been known as "benchmarks".
Due to their length, these tests are not run when using the above command.
Instead, you can use the `run_all_benchmarks.py` file in the `Aviary/aviary` folder, which is just a light wrapper around the `testflo` call.
This will run all of the longer tests in parallel using all of your available CPU cores.

## Package versions

Information on the versions of the packages required for Aviary can be found in the most recent [GitHub Actions runs](https://github.com/OpenMDAO/Aviary/actions).
We have also provided a static version of the `environment.yml` at the top level of the Aviary repo.

## Planned future features

Aviary is in active development.
We plan to expand its capabilities and have provided a non-exhaustive [list of future features](https://openmdao.github.io/om-Aviary/misc_resources/planned_future_features.html).