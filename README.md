# Aviary -- NASA's aircraft design tool
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-17-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

**Check out the Aviary [documentation](https://openmdao.github.io/Aviary/intro.html).**

**[NASA's Aviary Takes Flight](https://ntrs.nasa.gov/citations/20240009217) (Presented at EAA AirVenture 2024)**

**Get in touch with the Aviary team at agency-aviary@mail.nasa.gov**

## Description

This repository is an [OpenMDAO](https://openmdao.org/)-based aircraft modeling tool that incorporates aircraft sizing and weight equations from its predecessors [GASP (General Aviation Synthesis Program)](https://ntrs.nasa.gov/api/citations/19810010563/downloads/19810010563.pdf) and [FLOPS (Flight Optimization System)](https://software.nasa.gov/software/LAR-18934-1).
It also incorporates aerodynamic calculations from GASP and FLOPS and has the capability to use an aerodynamics deck as well as an aircraft engine deck.
There are two options for the mission analysis portion of this code, a 2 degrees-of-freedom (2DOF) approach, and a energy-height approach.
The user can select which type of mission analysis to use, as well as whether to use the FLOPS-based code or the GASP-based code for the weight, sizing, and aerodynamic relations.

## Installation

The simplest installation method for users is to install via pip:

    pip install om-aviary

Please see the [installation doc page](https://openmdao.github.io/Aviary/getting_started/installation.html) for more detailed instructions.
The minimum supported Python version for Aviary is 3.9.

## Documentation

The Aviary documentation is located [here](https://openmdao.github.io/Aviary/intro.html).

## Validation

This code has been validated using output and data from the GASP and FLOPS codes themselves. The GASP-based weight calculations in this code include in their comments which versions of the GASP standalone weights module were used in validation. The aero and EOM subsystem validations were based on runs of the entire GASP and FLOPS code as they stood in the summer of 2021 and the summer of 2022 respectively.

### Quick testing

The repository installation can be tested using the command ``testflo .`` at the top-level Aviary folder. If you have both SNOPT and IPOPT installed the output should look something like this:

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
We plan to expand its capabilities and have provided a non-exhaustive [list of future features](https://openmdao.github.io/Aviary/misc_resources/planned_future_features.html).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chapman178"><img src="https://avatars.githubusercontent.com/u/2847218?v=4?s=100" width="100px;" alt="Jeff Chapman"/><br /><sub><b>Jeff Chapman</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=chapman178" title="Code">💻</a> <a href="#example-chapman178" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/crecine"><img src="https://avatars.githubusercontent.com/u/51181861?v=4?s=100" width="100px;" alt="crecine"/><br /><sub><b>crecine</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=crecine" title="Code">💻</a> <a href="#data-crecine" title="Data">🔣</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Acrecine" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dlcaldwelljr-ama-inc"><img src="https://avatars.githubusercontent.com/u/39774974?v=4?s=100" width="100px;" alt="dlcaldwelljr-ama-inc"/><br /><sub><b>dlcaldwelljr-ama-inc</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=dlcaldwelljr-ama-inc" title="Code">💻</a> <a href="#infra-dlcaldwelljr-ama-inc" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#mentoring-dlcaldwelljr-ama-inc" title="Mentoring">🧑‍🏫</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Adlcaldwelljr-ama-inc" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ehariton"><img src="https://avatars.githubusercontent.com/u/11527849?v=4?s=100" width="100px;" alt="Eliot Aretskin-Hariton"/><br /><sub><b>Eliot Aretskin-Hariton</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=ehariton" title="Code">💻</a> <a href="#fundingFinding-ehariton" title="Funding Finding">🔍</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Aehariton" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/erikdolsonva"><img src="https://avatars.githubusercontent.com/u/39806272?v=4?s=100" width="100px;" alt="Erik Olson"/><br /><sub><b>Erik Olson</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=erikdolsonva" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/errordynamicist"><img src="https://avatars.githubusercontent.com/u/109693657?v=4?s=100" width="100px;" alt="DP"/><br /><sub><b>DP</b></sub></a><br /><a href="#example-errordynamicist" title="Examples">💡</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gawrenn"><img src="https://avatars.githubusercontent.com/u/127416371?v=4?s=100" width="100px;" alt="gawrenn"/><br /><sub><b>gawrenn</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=gawrenn" title="Code">💻</a> <a href="#example-gawrenn" title="Examples">💡</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hschilling"><img src="https://avatars.githubusercontent.com/u/867557?v=4?s=100" width="100px;" alt="hschilling"/><br /><sub><b>hschilling</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=hschilling" title="Code">💻</a> <a href="#design-hschilling" title="Design">🎨</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://ixjlyons.com"><img src="https://avatars.githubusercontent.com/u/943602?v=4?s=100" width="100px;" alt="Kenneth Lyons"/><br /><sub><b>Kenneth Lyons</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=ixjlyons" title="Code">💻</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Aixjlyons" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jdgratz10"><img src="https://avatars.githubusercontent.com/u/46534043?v=4?s=100" width="100px;" alt="Jennifer Gratz"/><br /><sub><b>Jennifer Gratz</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=jdgratz10" title="Code">💻</a> <a href="https://github.com/OpenMDAO/Aviary/commits?author=jdgratz10" title="Documentation">📖</a> <a href="#projectManagement-jdgratz10" title="Project Management">📆</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Ajdgratz10" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jkirk5"><img src="https://avatars.githubusercontent.com/u/110835404?v=4?s=100" width="100px;" alt="Jason Kirk"/><br /><sub><b>Jason Kirk</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=jkirk5" title="Code">💻</a> <a href="#data-jkirk5" title="Data">🔣</a> <a href="https://github.com/OpenMDAO/Aviary/commits?author=jkirk5" title="Documentation">📖</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Ajkirk5" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/johnjasa"><img src="https://avatars.githubusercontent.com/u/16373529?v=4?s=100" width="100px;" alt="John Jasa"/><br /><sub><b>John Jasa</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=johnjasa" title="Code">💻</a> <a href="https://github.com/OpenMDAO/Aviary/commits?author=johnjasa" title="Documentation">📖</a> <a href="#example-johnjasa" title="Examples">💡</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3Ajohnjasa" title="Reviewed Pull Requests">👀</a> <a href="#video-johnjasa" title="Videos">📹</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.openmdao.org"><img src="https://avatars.githubusercontent.com/u/867917?v=4?s=100" width="100px;" alt="Kenneth Moore"/><br /><sub><b>Kenneth Moore</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=Kenneth-T-Moore" title="Code">💻</a> <a href="#infra-Kenneth-T-Moore" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/OpenMDAO/Aviary/pulls?q=is%3Apr+reviewed-by%3AKenneth-T-Moore" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/robfalck"><img src="https://avatars.githubusercontent.com/u/699809?v=4?s=100" width="100px;" alt="Rob Falck"/><br /><sub><b>Rob Falck</b></sub></a><br /><a href="#infra-robfalck" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#mentoring-robfalck" title="Mentoring">🧑‍🏫</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sixpearls"><img src="https://avatars.githubusercontent.com/u/1571853?v=4?s=100" width="100px;" alt="Ben Margolis"/><br /><sub><b>Ben Margolis</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=sixpearls" title="Code">💻</a> <a href="#infra-sixpearls" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/swryan"><img src="https://avatars.githubusercontent.com/u/881430?v=4?s=100" width="100px;" alt="swryan"/><br /><sub><b>swryan</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=swryan" title="Code">💻</a> <a href="#infra-swryan" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/xjjiang"><img src="https://avatars.githubusercontent.com/u/8505450?v=4?s=100" width="100px;" alt="Xun Jiang"/><br /><sub><b>Xun Jiang</b></sub></a><br /><a href="https://github.com/OpenMDAO/Aviary/commits?author=xjjiang" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
